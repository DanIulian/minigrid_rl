"""Dan Iulian Muntean 2020
"""
import numpy as np
import torch
import os

from agents.base_intrinsic import BaseCustomIntrinsic
from torch_rl.utils import DictList
from utils.format import preprocess_images
from utils.ec_curiosity_wrapper import CuriosityEnvWrapper
from utils.train_episodic_curiosity_network import RNetworkTrainer
from torch_rl.format import default_preprocess_obss


class PPOEpisodicCuriosity(BaseCustomIntrinsic):

    def __init__(self, cfg, envs, acmodel, agent_data, **kwargs):

        num_frames_per_proc = getattr(cfg, "num_frames_per_proc", 128)
        discount = getattr(cfg, "discount", 0.99)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        recurrence = getattr(cfg, "recurrence", 4)
        clip_eps = getattr(cfg, "clip_eps", 0.)
        epochs = getattr(cfg, "epochs", 4)
        batch_size = getattr(cfg, "batch_size", 256)

        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)

        scale_ext_reward = getattr(cfg, "scale_task_reward", 1.0)
        scale_int_reward = getattr(cfg, "scale_surrogate_reward", 0.03)

        # Create envs
        envs = CuriosityEnvWrapper(envs, cfg,
                                   torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                   (preprocess_obss or default_preprocess_obss),
                                   acmodel.curiosity_model)

        super().__init__(
            envs, acmodel, num_frames_per_proc,
            discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm,
            recurrence, preprocess_obss,
            reshape_reward, scale_ext_reward, scale_int_reward)


        # create the training class for RNetwork and add the callback to the CuriosityEnvWrapper
        self.curiosity_model_trainer = RNetworkTrainer(cfg, acmodel.curiosity_model,
                                                       self.device, self.preprocess_obss)
        self.env.add_observer(self.curiosity_model_trainer)

        # Info about PPO training
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_num = 0
        self.updates_cnt = 0

        # do initial training of RNetwork
        r_net_initial_budget = getattr(cfg, "r_net_initial_training_budget", 0)
        if r_net_initial_budget > 0:
            self._r_network_initial_train(r_net_initial_budget)

        # -- Prepare optimizers
        self._get_optimizers(cfg, agent_data)

        # EXTRA STUFF
        assert self.batch_size % self.recurrence == 0

        self.out_dir = getattr(cfg, "out_dir", None)
        self.experience_dir = f"{self.out_dir}/exp"
        self.save_experience = getattr(cfg, "save_experience", 0)
        if self.save_experience:
            os.mkdir(self.experience_dir)

        # get width and height of the observation space for position normalization
        self.env_width = envs[0][0].unwrapped.width
        self.env_height = envs[0][0].unwrapped.height

        # remember some log values from intrinsic rewards computation
        self.aux_logs = {}

    def _get_optimizers(self, cfg, agent_data):
        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})

        self.lr = optimizer_args.lr
        # -- Prepare optimizers
        optimizer_args = vars(optimizer_args)

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), **optimizer_args)

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.curiosity_model_trainer.get_optimizer.load_state_dict(agent_data["optimizer_agworld"])

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

        # Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        for epoch_no in range(self.epochs):
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
                        dist, value, memory = self.acmodel.policy_model(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel.policy_model(sb.obs)

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
                self.optimizer_policy.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.policy_model.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.policy_model.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

                # if batch_kl > 1.5 * 0.01:
                #    print("Stop optimizing, batch_kl {}".format(batch_kl))
                #    break

        # Log some values

        logs["entropy"] = np.mean(log_entropies)
        logs["value"] = np.mean(log_values)
        logs["policy_loss"] = np.mean(log_policy_losses)
        logs["value_loss"] = np.mean(log_value_losses)
        logs["grad_norm"] = np.mean(log_grad_norms)

        return logs

    def _get_batches_starting_indexes(self, recurrence=None, padding=0):
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
        num_frames_per_proc = self.num_frames_per_proc
        num_procs = self.num_procs

        if recurrence is None:
            recurrence = self.recurrence

        # Consider Num frames list ordered P * T
        if padding == 0:
            indexes = np.arange(0, self.num_frames, recurrence)
        else:
            # Consider Num frames list ordered P * T
            # Do not index step[:padding] and step[-padding:]
            frame_index = np.arange(padding, num_frames_per_proc-padding + 1 - recurrence, recurrence)
            indexes = np.resize(frame_index.reshape((1, -1)), (num_procs, len(frame_index)))
            indexes = indexes + np.arange(0, num_procs).reshape(-1, 1) * num_frames_per_proc
            indexes = indexes.reshape(-1)

        indexes = np.random.permutation(indexes)
        self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [
            indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes

    def _r_network_initial_train(self, num_steps):
        # Make num_steps in the env, making some initial training of r_network
        # before starting the agent full training

        print("Start initial RNetwork trainng for {} steps".format(num_steps))

        curr_obs = self.obs
        for i in range(num_steps):
            # Do one agent-environment interaction

            action = torch.randint(0, self.env.action_space.n, (self.num_procs,))  # sample uniform actions
            obs, reward, done, info = self.env.step(action.cpu().numpy())

            curr_obs = obs

        self.obs = curr_obs

    def _calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):
        pass

    def add_extra_experience(self, exps: DictList):
        # Process
        full_positions = [self.obss[i][j]["position"]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)]
        # Process
        full_states = [self.obss[i][j]["state"]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)]

        exps.states = preprocess_images(full_states, device=self.device)
        max_pos_value = max(self.env_height, self.env_width)
        exps.position = preprocess_images(full_positions,
                                          device=self.device,
                                          max_image_value=max_pos_value,
                                          normalize=False)
        exps.obs_image = exps.obs.image

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_agworld": self.curiosity_model_trainer.get_optimizer.state_dict(),
        })
