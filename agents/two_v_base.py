"""
    Copyright (c) https://github.com/lcswillems/torch-rl
"""

from abc import ABC, abstractmethod
import torch
import numpy

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class TwoValueHeadsBase(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, exp_used_pred):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        exp_used_pred : float
            the proportion of experience used for training predictor
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.exp_used_pred = exp_used_pred

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = sum(map(len, envs)) if isinstance(envs[0], list) else len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values_ext = torch.zeros(*shape, device=self.device)
        self.values_int = torch.zeros(*shape, device=self.device)
        self.rewards_ext = torch.zeros(*shape, device=self.device)
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.advantages_ext = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values_ext[i] = value[0]
            self.values_int[i] = value[1]
            if self.reshape_reward is not None:
                self.rewards_ext[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards_ext[i] = torch.tensor(reward, device=self.device)

            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards_ext[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # ==========================================================================================
        # Define experiences: ---> for observations
        #   the whole experience is the concatenation of the experience
        #   of each process.
        #import pdb; pdb.set_trace()
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # ==========================================================================================

        # -- Calculate intrinsic return
        self.calculate_intrinsic_reward(exps, self.rewards_int)

        # Add advantage and return to experiences
        # don;t use end of episode signal for intrinsic rewards
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        #Calculate intrinsic rewards and advantages
        for i in reversed(range(self.num_frames_per_proc)):
            next_value_int = self.values_int[i + 1] if i < self.num_frames_per_proc - 1 else next_value[1]
            next_advantage_int = self.advantages_int[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards_int[i] + self.discount * next_value_int - self.values_int[i]
            self.advantages_int[i] = delta + self.discount * self.gae_lambda * next_advantage_int

        #Calculate extrinisc rewards and advantages
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask

            next_value_ext = self.values_ext[i+1] if i < self.num_frames_per_proc - 1 else next_value[0]
            next_advantage_ext = self.advantages_ext[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards_ext[i] + self.discount * next_value_ext * next_mask - self.values_ext[i]
            self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage_ext * next_mask



        # ==========================================================================================
        # @ continue Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value_ext = self.values_ext.transpose(0, 1).reshape(-1)
        exps.value_int = self.values_int.transpose(0, 1).reshape(-1)
        exps.reward_ext = self.rewards_ext.transpose(0, 1).reshape(-1)
        exps.reward_int = self.rewards_int.transpose(0, 1).reshape(-1)
        exps.advantage_ext = self.advantages_ext.transpose(0, 1).reshape(-1)
        exps.advantage_int = self.advantages_int.transpose(0, 1).reshape(-1)
        exps.returnn_ext = exps.value_ext + exps.advantage_ext
        exps.returnn_int = exps.value_int + exps.advantage_int
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        raise NotImplemented

    @abstractmethod
    def get_save_data(self):
        raise NotImplemented

    @abstractmethod
    def calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):
        raise NotImplemented
