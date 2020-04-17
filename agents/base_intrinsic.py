"""
    Dan Iulian Muntean 2020
    Trajectory collection class with extra tweaks
"""

from abc import ABC, abstractmethod
import torch
from torch_rl.utils import DictList
from agents.base import BaseAlgo


class BaseCustomIntrinsic(BaseAlgo):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 scale_ext_reward, scale_int_reward, min_stats_ep_batch=16, log_metrics_names=None, ):

        super(BaseCustomIntrinsic, self).__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
            min_stats_ep_batch=16, log_metrics_names=None)

        self.int_r_scale = scale_int_reward
        self.ext_r_scale = scale_ext_reward

        # Initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)
        self._initialize_extra_experience_values(shape)

        self._initialize_extra_log_values(log_metrics_names)

    def _initialize_extra_experience_values(self, shape):

        #Initialize extra experience values
        self.values_int = torch.zeros(*shape, device=self.device)
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)

    def _initialize_extra_log_values(self, log_metrics_names):

        # Initialize extra log values
        self.log_episode_int_return = torch.zeros(self.num_procs, device=self.device)

        self.log_return_intrinsic = [0] * self.num_procs
        self.log_full_int_return = [[] * self.num_procs]

    def collect_experiences(self):
        """Collects rollouts and computes advantages.
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

            obs, reward, done, info = self.env.step(action.cpu().numpy())

            self.collect_interactions(info)

            # Update experiences values
            self.update_experience_values(obs, action, reward, done, memory, value, i)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values:
            self.update_log_values(reward, done, i)

        # Add advantage and return to experiences

        # Make one step further to get the next value approximation
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        # Compute advantages
        self.compute_advantages(next_value)

        # Define experinces:
        exps = self.get_experiences()

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,

            # Info about intrinsic rewards
            "return_int_per_episode": self.log_return_intrinsic[-keep:],
        }

        # ==========================================================================================
        aux_logs = self.process_interactions()
        # add extra logs with agent interactions
        for k in aux_logs:
            log[k] = aux_logs[k]
        # ==========================================================================================

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_return_intrinsic = self.log_return_intrinsic[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    def update_experience_values(self, curr_obs, curr_act, curr_reward,
                                 curr_done, curr_memory, curr_value, curr_step):

        reward_ext, reward_int = curr_reward
        combined_reward = self.ext_r_scale * reward_ext + self.int_r_scale * reward_int

        # Update experiences values
        self.obss[curr_step] = self.obs
        self.obs = curr_obs  # New observations
        if self.acmodel.recurrent:
            self.memories[curr_step] = self.memory
            self.memory = curr_memory
        self.masks[curr_step] = self.mask
        self.mask = 1 - torch.tensor(curr_done, device=self.device, dtype=torch.float)
        self.actions[curr_step] = curr_act
        self.values_ext[curr_step] = curr_value
        if self.reshape_reward is not None:
            self.rewards_ext[curr_step] = torch.tensor([
                self.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(curr_obs, curr_act, reward_ext, curr_done)
            ], device=self.device)

            self.rewards[curr_step] = torch.tensor([
                self.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(curr_obs, curr_act, combined_reward, curr_done)
            ])
        else:
            self.rewards_ext[curr_step] = torch.tensor(reward_ext, device=self.device)
            self.rewards[curr_step] = torch.tensor(combined_reward, device=self.device)

        self.rewards_int[curr_step] = torch.tensor(reward_int)

    def update_log_values(self, reward, done, step):

        reward_ext, reward_int = reward

        # Update log values
        self.log_episode_return += torch.tensor(reward_ext, device=self.device, dtype=torch.float)
        self.log_episode_reshaped_return += self.rewards_ext[step]
        self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

        #Update log values for intrinsic reward
        self.log_episode_int_return += torch.tensor(reward_int, device=self.device, dtype=torch.float)

        for j, done_ in enumerate(done):
            if done_:
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return[j].item())
                self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                self.log_num_frames.append(self.log_episode_num_frames[j].item())

                self.log_return_intrinsic.append(self.log_episode_int_return[j].item())

        self.log_episode_return *= self.mask
        self.log_episode_reshaped_return *= self.mask
        self.log_episode_num_frames *= self.mask

        self.log_episode_int_return *= self.mask

    def compute_advantages(self, next_value):

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values_ext[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages_ext[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values_ext[i]
            self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

    def get_experiences(self):

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = super(BaseCustomIntrinsic, self).get_experiences()

        exps.rewards_int = self.rewards_int.transpose(0, 1).reshape(-1)
        exps.rewards = self.rewards.transpose(0, 1).reshape(-1)

        return exps

    @abstractmethod
    def update_parameters(self):
        raise NotImplemented
