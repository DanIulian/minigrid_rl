"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy
import torch
from agents.base_algo_v2 import BaseAlgov2
from torch_rl.utils import ParallelEnv

class PPO(BaseAlgov2):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, cfg, envs, acmodel, agent_data, **kwargs):

        # Get standard config params from config file
        num_frames_per_proc = getattr(cfg, "frames_per_proc", 128)
        discount = getattr(cfg, "discount", 0.99)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        recurrence = getattr(cfg, "recurrence", 4)
        clip_eps = getattr(cfg, "clip_eps", 0.2)
        epochs = getattr(cfg, "epochs", 4)
        batch_size = getattr(cfg, "batch_size", 256)
        log_metrics_names = getattr(cfg, "log_metrics", [])

        # Get optimizer config params from config file
        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})
        eval_envs = kwargs.get("eval_envs", [])
        eval_episodes = kwargs.get("eval_episodes", 0)

        self.out_dir = getattr(cfg, "out_dir", None)
        self.experience_dir = f"{self.out_dir}/exp"
        self.save_experience = save_experience = getattr(cfg, "save_experience", 0)

        if save_experience:
            import os
            os.mkdir(self.experience_dir)

        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)

        envs = ParallelEnv(envs)
        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, optimizer_args.lr, gae_lambda,
            entropy_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss,
            reshape_reward, log_metrics_names=log_metrics_names)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        optimizer_args = vars(optimizer_args)

        self.optimizer = getattr(torch.optim, optimizer)(self.acmodel.parameters(), **optimizer_args)

        if "optimizer" in agent_data:
            self.optimizer.load_state_dict(agent_data["optimizer"])

        self.batch_num = 0

        if len(eval_envs) > 0:
            self.eval_memory = None
            self.eval_mask = None
            self.eval_rewards = None

            self.eval_episodes = eval_episodes
            self.eval_envs = self.init_evaluator(eval_envs)

    def update_parameters(self):
        # Collect experiences
        update = self.batch_num
        exps, logs = self.collect_experiences()

        if self.save_experience > 0:
            norm_value = 10.
            # assert False, "Must set norm value"

            nstep = self.save_experience * self.num_frames_per_proc
            experience = dict()
            experience["logs"] = logs
            experience["obs_image"] = (exps.obs.image[:nstep].cpu() * norm_value).byte()
            experience["mask"] = exps.mask[:nstep]
            experience["action"] = exps.action[:nstep]
            experience["reward"] = exps.reward[:nstep]
            experience["num_procs"] = self.save_experience
            experience["frames_per_proc"] = self.num_frames_per_proc
            experience["norm_value"] = norm_value
            torch.save(experience, f"{self.experience_dir}/exp_update_{update}")

        for epoch_no in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_kl = 0

                # Initialize memory
                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    # inds is an array of batch_size // recurrence containing indexes of obs
                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage_ext
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage_ext
                    policy_loss = -torch.min(surr1, surr2).mean()

                    approx_kl = (sb.log_prob - dist.log_prob(sb.action)).mean()

                    value_clipped = sb.value_ext + torch.clamp(value - sb.value_ext, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn_ext).pow(2)
                    surr2 = (value_clipped - sb.returnn_ext).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_kl += approx_kl.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_kl /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

                #if batch_kl > 1.5 * 0.01:
                #    print("Stop optimizing, batch_kl {}".format(batch_kl))
                #    break

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        self.batch_num += 1

        # make a list of arrays, where each array contains num_indexes elements(batch_size if no recurrence)
        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({"optimizer": self.optimizer.state_dict()})

    def init_evaluator(self, envs):
        from torch_rl.utils import ParallelEnv
        device = self.device
        acmodel = self.acmodel

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
        num_envs_results = 0
        log_reshaped_return = []

        import pdb; pdb.set_trace()
        while num_envs_results < eval_episodes:

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
                    num_envs_results += 1
                    if eval_key is None:
                        log_reshaped_return.append(log_episode_reshaped_return[j].item())
                    else:
                        log_reshaped_return.append(info[j][eval_key])

            log_episode_reshaped_return *= mask.squeeze(1)

            obs = next_obs

        return {"eval_r": numpy.mean(log_reshaped_return[:eval_episodes])}
