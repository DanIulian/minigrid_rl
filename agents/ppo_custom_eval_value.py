import numpy
import torch
import copy
from torch.distributions.categorical import Categorical

from agents.ppo_replica import PPO


class PPOCustomEvalValue(PPO):
    def init_evaluator(self, envs):
        from torch_rl.utils import ParallelEnv
        device = self.device
        acmodel = self.acmodel

        env0 = envs[0][0]
        self.num_train_goals = num_train_goals = len(env0.train_goals)
        self.num_test_goals = num_test_goals = len(env0.test_goals)

        print(f"Possible goals count: train-{num_train_goals} | test-{num_test_goals}")

        all_envs = [itt for sublist in envs for itt in sublist]
        nenvs = len(all_envs)
        assert nenvs == num_train_goals + num_test_goals
        self._train_num_envs = train_num_envs = num_train_goals  # nenvs  # nenvs // 2

        print(f"TOTAL NUM ENVS eval split. Train {train_num_envs} "
              f"Test {nenvs-self._train_num_envs}")

        for i in range(0, train_num_envs):
            all_envs[i].unwrapped.eval_id = i

        for i in range(num_train_goals, nenvs):
            all_envs[i].unwrapped.train = False
            all_envs[i].unwrapped.eval_id = i - num_train_goals

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

        # ==========================================================================================
        # TODO HARDCODED
        # preprocessed_obs = preprocess_obss(obs, device=device)
        # assert preprocessed_obs.image.max() == 1., "Not correct norm"

        device = rewards.device
        norm_value = 10.
        ag_id = 10 / norm_value
        empty_id = 1 / norm_value
        ag_repr = torch.tensor([ag_id, 0., 0.], device=device)
        empty_repr = torch.tensor([empty_id, 0., 0.], device=device)
        move_actions = torch.tensor([
            [-1, 0],
            [0, -1],
            [1, 0],
            [0, 1],
        ], device=device)
        # ==========================================================================================
        data = dict()

        for meth in ["policy", "value"]:
            run_value = meth == "value"
            memory = None
            obs = env.reset()
            if recurrent:
                memory = self.eval_memory
                memory.zero_()

            rewards.zero_()
            log_episode_reshaped_return = torch.zeros_like(rewards).float()

            mask = self.eval_mask.fill_(1).unsqueeze(1)
            num_envs_results = [[0] * self.num_train_goals, [0] * self.num_test_goals]
            log_reshaped_return = [[], []]
            scores = [[], []]

            while min(num_envs_results[0]) < eval_episodes or \
                    min(num_envs_results[1]) < eval_episodes:
                preprocessed_obs = preprocess_obss(obs, device=device)

                if run_value:
                    # Calculate observations for next step
                    obs = preprocessed_obs.image
                    next_state_obs = []
                    for action_id in range(4):
                        new_obs = obs.clone()
                        ag_pos = torch.where(new_obs[:, :, :, 0] == 1.)
                        ag_repr = new_obs[ag_pos]
                        new_obs[ag_pos] = empty_repr

                        move_with = move_actions[action_id]
                        new_ag_pos = copy.deepcopy(ag_pos)

                        new_ag_pos[1].add_(move_with[0])
                        new_ag_pos[1].clamp_(1, 14)

                        new_ag_pos[2].add_(move_with[1])
                        new_ag_pos[2].clamp_(1, 14)

                        new_obs[new_ag_pos] = ag_repr
                        next_state_obs.append(new_obs)

                    next_state_obs = torch.cat(next_state_obs, dim=0)
                    preprocessed_obs.image = next_state_obs

                with torch.no_grad():
                    if recurrent:
                        dist, value, memory = acmodel(preprocessed_obs, memory * mask)
                    else:
                        dist, value = acmodel(preprocessed_obs)

                    if run_value:
                        # # Change value in action dist =
                        act_values = value.view(4, -1)
                        dist = Categorical(logits=act_values.t())
                        action = dist.sample()
                    else:
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
                        ej = j - train_envs if eval_type else j

                        num_envs_results[eval_type][ej] += 1
                        if eval_key is None:
                            log_reshaped_return[eval_type].append(
                                log_episode_reshaped_return[j].item())
                        else:
                            log_reshaped_return[eval_type].append(info[j][eval_key])

                log_episode_reshaped_return *= mask.squeeze(1)

                obs = next_obs

            # print(f"train_r_{meth}", log_reshaped_return[0])
            # print(f"eval_r_{meth}", log_reshaped_return[1])
            data.update({
                f"train_r_{meth}": numpy.mean(log_reshaped_return[0]),
                f"eval_r_{meth}": numpy.mean(log_reshaped_return[1]),
                # "scores": str(scores)
            })
        return data

