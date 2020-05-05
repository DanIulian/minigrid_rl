"""
    Copyright (c) https://github.com/lcswillems/torch-rl
"""

from abc import ABC, abstractmethod
import torch
from copy import deepcopy

from agents.base import BaseAlgo
from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv
from utils.gym_wrappers import get_interactions_stats


import torch.nn.functional as F
from argparse import Namespace
from copy import deepcopy


class TwoValueHeadsBaseGeneral(BaseAlgo):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 exp_used_pred, min_stats_ep_batch=16, log_metrics_names=None, intrinsic_reward_fn = None):

        super(TwoValueHeadsBaseGeneral, self).__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda,
            entropy_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss,
            reshape_reward, min_stats_ep_batch, log_metrics_names)

        # Initialize experiences values for intrinsic rewards
        shape = (self.num_frames_per_proc, self.num_procs)
        self.exp_used_pred = exp_used_pred
        self.values_int = torch.zeros(*shape, device=self.device)
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.normadv = True

    def collect_experiences(self):
        """Collects rollouts and computes advantages. See base class for more info

        Returns
        -------
        exps : DictList
        logs : dict
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

            # Update log values
            self.update_log_values(reward, done, i)

        # ==========================================================================================
        # Define experiences: ---> for observations
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # import pdb; pdb.set_trace()
        exps = DictList()

        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # Add other data to experience buffer
        self.add_extra_experience(exps)

        # ==========================================================================================
        # -- Calculate intrinsic return
        if self.intrinsic_reward_fn:
            self.rewards_int = self.intrinsic_reward_fn(exps, self.rewards_int)

        # Add advantage and return to experiences
        # don't use end of episode signal for intrinsic rewards
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        # Compute advantages
        self.compute_advantages(next_value)

        # Add the remaining required experiences
        self.get_experiences(exps)

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        # =====================================================================================
        aux_logs = self.process_interactions()
        # add extra logs with agent interactions
        for k in aux_logs:
            log[k] = aux_logs[k]
        # =====================================================================================

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    def update_experience_values(self, curr_obs, curr_act, curr_reward,
                                 curr_done, curr_memory, curr_value, curr_step):

        super(TwoValueHeadsBaseGeneral, self).update_experience_values(
            curr_obs, curr_act, curr_reward, curr_done,
            curr_memory, curr_value[0], curr_step
        )

        self.values_int[curr_step] = curr_value[1]

    def compute_advantages(self, next_value):

        # Calculate extrinisc rewards and advantages
        super(TwoValueHeadsBaseGeneral, self).compute_advantages(next_value[0])

        # Calculate intrinsic rewards and advantages
        for i in reversed(range(self.num_frames_per_proc)):
            next_value_int = self.values_int[i + 1] if i < self.num_frames_per_proc - 1 else next_value[1]
            next_advantage_int = self.advantages_int[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards_int[i] + self.discount * next_value_int - self.values_int[i]
            self.advantages_int[i] = delta + self.discount * self.gae_lambda * next_advantage_int

    def update_log_values(self, reward, done, step):
        super(TwoValueHeadsBaseGeneral, self).update_log_values(reward, done, step)

    def get_experiences(self, exps):
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # for all tensors below, T x P -> P x T -> P * T
        exps.value_ext = self.values_ext.transpose(0, 1).reshape(-1)
        exps.value_int = self.values_int.transpose(0, 1).reshape(-1)
        exps.reward_ext = self.rewards_ext.transpose(0, 1).reshape(-1)
        exps.reward_int = self.rewards_int.transpose(0, 1).reshape(-1)
        exps.advantage_ext = self.advantages_ext.transpose(0, 1).reshape(-1)
        exps.advantage_int = self.advantages_int.transpose(0, 1).reshape(-1)
        exps.returnn_ext = exps.value_ext + exps.advantage_ext
        exps.returnn_int = exps.value_int + exps.advantage_int
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

    def augment_exp(self, exps, curiosity_model=None):

        # from exp (P * T , ** ) -> (T (rollout length), P, **)
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        env = self.env

        if curiosity_model is None:
            agworld_network = self.acmodel.curiosity_model
        else:
            agworld_network = curiosity_model

        shape = torch.Size([num_procs, num_frames_per_proc])
        frame_exp = Namespace()

        # ------------------------------------------------------------------------------------------
        # Redo in format T x P

        for k, v in exps.items():
            if k == "obs":
                continue
            setattr(frame_exp, k, v.view(shape + v.size()[1:]).transpose(0, 1).contiguous())

        def inverse_img(t, ii):
            return torch.transpose(torch.transpose(t, ii, ii+2), ii+1, ii+2).contiguous()

        frame_exp.obs_image = inverse_img(frame_exp.obs_image, 2)
        #frame_exp.states = inverse_img(frame_exp.states, 2)

        def gen_memory(ss):
            return torch.zeros(num_frames_per_proc, num_procs, ss, device=device)

        frame_exp.agworld_mems = gen_memory(agworld_network.memory_size)
        frame_exp.agworld_embs = gen_memory(agworld_network.embedding_size)

        frame_exp.actions_onehot = gen_memory(env.action_space.n)
        frame_exp.actions_onehot.scatter_(2, frame_exp.action.unsqueeze(2).long(), 1.)

        # ------------------------------------------------------------------------------------------
        # Save last frame exp

        last_frame_exp = Namespace()
        for k, v in frame_exp.__dict__.items():
            if k == "obs":
                continue
            setattr(last_frame_exp, k, v[-1].clone())

        prev_frame_exps = self.prev_frame_exps
        if self.prev_frame_exps is None:
            prev_frame_exps = deepcopy(last_frame_exp)
            for k, v in prev_frame_exps.__dict__.items():
                v.zero_()

        self.prev_frame_exps = last_frame_exp

        # ------------------------------------------------------------------------------------------
        # Fill memories with past

        frame_exp.agworld_mems[0] = prev_frame_exps.agworld_mems
        frame_exp.agworld_embs[0] = prev_frame_exps.agworld_embs

        return frame_exp, prev_frame_exps

    @staticmethod
    def flip_back_experience(exp):
        # for all tensors below, T x P -> P x T -> P * T
        for k, v in exp.__dict__.items():
            setattr(exp, k, v.transpose(0, 1).reshape(-1, *v.shape[2:]).contiguous())
        return exp

    def collect_interactions(self, info):
        super(TwoValueHeadsBaseGeneral, self).collect_interactions(info)

    def process_interactions(self):

        # process statistics about the agent's behaviour
        # in the environment
        logs = super(TwoValueHeadsBaseGeneral, self).process_interactions()
        return logs

    @abstractmethod
    def update_parameters(self):
        raise NotImplemented

    @abstractmethod
    def get_save_data(self):
        raise NotImplemented

    def add_extra_experience(self, exps: DictList):
        return
