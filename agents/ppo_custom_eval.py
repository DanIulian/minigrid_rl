"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy
import torch
import torch.nn.functional as F
import collections

from agents.ppo_replica import PPO


class PPOCustomEval(PPO):
    def init_evaluator(self, envs):
        from torch_rl.utils import ParallelEnv
        device = self.device
        acmodel = self.acmodel

        all_envs = [itt for sublist in envs for itt in sublist]
        nenvs = len(all_envs)
        self._train_num_envs = train_num_envs = nenvs // 2

        print(f"TOTAL NUM ENVS eval split. Train {train_num_envs} "
              f"Test {nenvs-self._train_num_envs}")

        for i in range(train_num_envs, nenvs):
            all_envs[i].unwrapped.train = False

        eval_envs = ParallelEnv(envs)
        obs = eval_envs.reset()

        if self.acmodel.recurrent:
            self.eval_memory = torch.zeros(len(obs), acmodel.memory_size, device=device)

        self.eval_mask = torch.ones(len(obs), device=device)
        self.eval_rewards = torch.zeros(len(obs), device=device)

        return eval_envs

    def evaluate(self, eval_key=None):
        env = self.eval_envs
        eval_episodes = self.eval_episodes
        train_envs = self._train_num_envs

        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.acmodel.recurrent
        acmodel = self.acmodel
        reshape_reward = self.reshape_reward
        rewards = self.eval_rewards

        memory = None
        obs = env.reset()
        if recurrent:
            memory = self.eval_memory
            memory.zero_()

        rewards.zero_()
        log_episode_reshaped_return = torch.zeros_like(rewards)

        mask = self.eval_mask.fill_(1).unsqueeze(1)
        num_envs_results = [0, 0]
        log_reshaped_return = [[], []]

        while min(num_envs_results) < eval_episodes:

            preprocessed_obs = preprocess_obss(obs, device=device)

            with torch.no_grad():
                if recurrent:
                    dist, value, memory = acmodel(preprocessed_obs, memory * mask)
                else:
                    dist, value = acmodel(preprocessed_obs)
                action = dist.sample()

            next_obs, reward, done, info = env.step(action.cpu().numpy())

            mask = (1 - torch.tensor(done, device=device, dtype=torch.float)).unsqueeze(1)

            if reshape_reward is not None:
                rewards = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=device)
            else:
                rewards = torch.tensor(reward, device=self.device)

            log_episode_reshaped_return += rewards

            for j, done_ in enumerate(done):
                if done_:
                    eval_type = 0 if j < train_envs else 1

                    num_envs_results[eval_type] += 1
                    if eval_key is None:
                        log_reshaped_return[eval_type].append(log_episode_reshaped_return[j].item())
                    else:
                        log_reshaped_return[eval_type].append(info[j][eval_key])

            log_episode_reshaped_return *= mask.squeeze(1)

            obs = next_obs

        return {
            "train_r": numpy.mean(log_reshaped_return[0][:eval_episodes]),
            "eval_r": numpy.mean(log_reshaped_return[1][:eval_episodes])
        }
