"""
Dan Iulian Muntean 2020
Parts copied from https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/curiosity_env_wrapper.py
Wrapper around a Gym environment to add curiosity reward."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import torch
import numpy as np
from copy import deepcopy
import utils.episodic_memory as ep_mem
from torch_rl.utils import ParallelEnv
from utils.utils import RunningMeanStd


def get_observation_embedding_fn(r_model):

    def _observation_embedding_fn(x):
        r_model.eval()
        with torch.no_grad():
            embeddings = r_model.forward(x)
        r_model.train()

        return embeddings

    return _observation_embedding_fn


class CuriosityEnvWrapper(ParallelEnv):
    """Environment wrapper that adds additional curiosity reward."""
    def __init__(self,
                 envs,
                 cfg,
                 device,
                 num_frames_per_proc,
                 d_running_mean,
                 observation_preprocess_fn,
                 r_model):

        super(CuriosityEnvWrapper, self).__init__(envs)

        self._r_model = r_model
        self._device = device
        self._observation_preprocess_fn = observation_preprocess_fn
        self._observation_embedding_fn = get_observation_embedding_fn(r_model)

        self._d_running_mean = d_running_mean
        self._nr_neighbours = getattr(cfg, "number_of_neighbours", 10)
        self._eps = getattr(cfg, "pseudo_counts_constant", 0.001)
        self._cluster_distance = getattr(cfg, "cluster_distance", 0.008)
        self._kernel_eps = getattr(cfg, "kernel_eps", 0.0001)
        self._max_similarity = getattr(cfg, "max_similarity", 8)

        # Create an episodic memory for each env
        replacement_strategy = getattr(cfg, "replacement_strategy", "fifo")
        memory_capacity = getattr(cfg, "memory_capacity", 400)
        embedding_size = getattr(cfg, "embedding_size", [512])

        self._episodic_memories = [
            ep_mem.EpisodicMemory(embedding_size,
                                  self._device,
                                  self._d_running_mean,
                                  self._nr_neighbours,
                                  self._eps,
                                  self._cluster_distance,
                                  self._kernel_eps,
                                  self._max_similarity,
                                  replacement_strategy,
                                  memory_capacity)
            for _ in range(self.no_envs + 1)]

        # Total number of steps of a rollout
        self._step_count = 0

        # Keep intrinsic rewards untill a rollout is finished
        shape = (num_frames_per_proc, self.no_envs + 1)
        self.intrinsic_rewards_rollout = torch.zeros(*shape, device=self._device)

    def _compute_curiosity_reward(self, observations, infos, dones):
        """Compute intrinsic curiosity reward.
        The reward is set to 0 when the episode is finished
        """

        frames = self._observation_preprocess_fn(observations, device=self._device)
        frames = torch.transpose(torch.transpose(frames.image, 1, 3), 2, 3).contiguous()

        embedded_observations = self._observation_embedding_fn(frames)

        intrinsic_rewards = [
            self._episodic_memories[k].compute_intrinsic_reward(embedded_observations[k])
            for k in range(self.no_envs + 1)
        ]

        # Updates the episodic memory of every environment.
        for k in range(self.no_envs + 1):
            # If we've reached the end of the episode, resets the memory
            # and always adds the first state of the new episode to the memory.
            if dones[k]:
                self._episodic_memories[k].reset()

            # Add all new embeddings to the episodic memory
            self._episodic_memories[k].add(embedded_observations[k].cpu(), infos[k])

        # Augment the reward with the exploration reward.
        bonus_rewards = [
            0.0 if d else ir for (ir, d) in zip(intrinsic_rewards, dones)
        ]

        self.intrinsic_rewards_rollout[self._step_count] = torch.tensor(bonus_rewards,
                                                                        dtype=torch.float,
                                                                        device=self._device)

    def reset(self):
        observations = super(CuriosityEnvWrapper, self).reset()

        frames = self._observation_preprocess_fn(observations, device=self._device)
        frames = torch.transpose(torch.transpose(frames.image, 1, 3), 2, 3).contiguous()
        embedded_observations = self._observation_embedding_fn(frames)

        for k in range(self.no_envs + 1):
            self._episodic_memories[k].reset()
            self._episodic_memories[k].add(embedded_observations[k].cpu(), None)

        return observations

    def step(self, actions):
        obs, rewards, dones, infos = super(CuriosityEnvWrapper, self).step(actions)

        self._compute_curiosity_reward(obs, infos, dones)
        self._step_count += 1

        return obs, rewards, dones, infos

    def get_episodic_memeories(self):
        return self._episodic_memories

    def get_episodic_memory(self, k):
        """Returns the episodic memory for the k-th environment."""
        return self._episodic_memories[k]

    def get_rollout_intrinsic_rewards(self):
        """ Returns the intrinsic rewards computed for the current rollout"""
        self._step_count = 0

        i_rew = self.intrinsic_rewards_rollout.clone()
        self.intrinsic_rewards_rollout = torch.zeros_like(self.intrinsic_rewards_rollout, device=self._device)
        return i_rew

    def render(self):
        raise NotImplementedError

    def __len__(self):
        return self.no_envs + 1

    def __getitem__(self, item):
        return self.envs[item]

