"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy
import torch
import torch.nn.functional as F

from agents.two_v_base_general_offset import TwoValueHeadsBaseGeneralOffset
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter


class PPOOrder(TwoValueHeadsBaseGeneralOffset):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, cfg, envs, acmodel, agent_data, **kwargs):
        num_frames_per_proc = getattr(cfg, "frames_per_proc", 128)
        discount = getattr(cfg, "discount", 0.99)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        recurrence = getattr(cfg, "recurrence", 4)

        clip_eps = getattr(cfg, "clip_eps", 0.)
        epochs = getattr(cfg, "epochs", 4)
        batch_size = getattr(cfg, "batch_size", 256)

        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})

        exp_used_pred = getattr(cfg, "exp_used_pred", 0.25)
        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)

        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)

        self.nminibatches = getattr(cfg, "nminibatches", 4)

        # Order Model configs
        self.experience_offset = getattr(cfg, "experience_offset", 10)

        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
            exp_used_pred, offset_exp=self.experience_offset)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        optimizer_args = vars(optimizer_args)

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), **optimizer_args)

        # Prepare intrinsic generators
        self.predictor_rms = RunningMeanStd()
        self.predictor_rff = RewardForwardFilter(gamma=self.discount)

        self.optimizer_predictor = self.acmodel.order_model.optimizer

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_predictor.load_state_dict(agent_data["optimizer_predictor"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

        self.batch_num = 0

        if self.running_norm_obs:
            self.collect_random_statistics(50)

        self.train_indexes = None

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        for epoch_no in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values_ext = []
            log_values_int = []
            log_policy_losses = []
            log_value_ext_losses = []
            log_value_int_losses = []
            log_grad_norms = []
            log_ret_int = []
            log_rew_int = []
            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_ext_value = 0
                batch_int_value = 0
                batch_policy_loss = 0
                batch_value_ext_loss = 0
                batch_value_int_loss = 0
                batch_loss = 0
                batch_ret_int = 0
                batch_rew_int = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]
                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, vvalue, memory = self.acmodel.policy_model(sb.obs, memory * sb.mask)
                    else:
                        dist, vvalue = self.acmodel.policy_model(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    adv = (self.int_coeff * sb.advantage_int + self.ext_coeff * sb.advantage_ext)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value losses
                    value_ext, value_int = vvalue

                    value_ext_clipped = sb.value_ext + torch.clamp(value_ext - sb.value_ext, -self.clip_eps, self.clip_eps)
                    surr1 = (value_ext - sb.returnn_ext).pow(2)
                    surr2 = (value_ext_clipped - sb.returnn_ext).pow(2)
                    value_ext_loss = torch.max(surr1, surr2).mean()

                    value_int_clipped = sb.value_int + torch.clamp(value_int - sb.value_int, -self.clip_eps, self.clip_eps)
                    surr1 = (value_int - sb.returnn_int).pow(2)
                    surr2 = (value_int_clipped - sb.returnn_int).pow(2)
                    value_int_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + \
                           (0.5 * self.value_loss_coef) * value_int_loss + \
                           (0.5 * self.value_loss_coef) * value_ext_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_ext_value += value_ext.mean().item()
                    batch_int_value += value_int.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_ext_loss += value_ext_loss.item()
                    batch_value_int_loss += value_int_loss.item()
                    batch_loss += loss
                    batch_ret_int += sb.returnn_int.mean().item()
                    batch_rew_int += sb.reward_int.mean().item()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_ext_value /= self.recurrence
                batch_int_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_ext_loss /= self.recurrence
                batch_value_int_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_rew_int /= self.recurrence
                batch_ret_int /= self.recurrence

                # Update actor-critic

                self.optimizer_policy.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.acmodel.policy_model.parameters()
                    if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.policy_model.parameters(),
                                               self.max_grad_norm)
                self.optimizer_policy.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values_ext.append(batch_ext_value)
                log_values_int.append(batch_int_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_ext_losses.append(batch_value_ext_loss)
                log_value_int_losses.append(batch_value_int_loss)
                log_grad_norms.append(grad_norm)
                log_ret_int.append(batch_ret_int)
                log_rew_int.append((batch_rew_int))

        # ==============================================================================
        # Update Predictor loss

        order_loss = self.acmodel.order_model.train_order(device=self.device)
        logs["order_loss"] = order_loss

        # ==============================================================================

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value_ext"] = numpy.mean(log_values_ext)
        logs["value_int"] = numpy.mean(log_values_int)
        logs["value"] = logs["value_ext"] + logs["value_int"]
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_ext_loss"] = numpy.mean(log_value_ext_losses)
        logs["value_int_loss"] = numpy.mean(log_value_int_losses)
        logs["value_loss"] = logs["value_int_loss"] + logs["value_ext_loss"]
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["return_int"] = numpy.mean(log_ret_int)
        logs["reward_int"] = numpy.mean(log_rew_int)

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
        # TODO Select only indexes from [0: train_frames_per_proc]
        if self.train_indexes is None:
            indexes = numpy.arange(0, self.num_frames, self.recurrence)
            select = (indexes % self.num_frames_per_proc) < self.train_frames_per_proc
            self.train_indexes = indexes[select]

        indexes = self.train_indexes

        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        # TODO THis is buggy!!!
        # if self.batch_num % 2 == 1:
        #     indexes = indexes[(indexes + recurrence) % num_frames_per_proc != 0]
        #     indexes += recurrence // 2

        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_predictor": self.optimizer_predictor.state_dict(),
            "predictor_rms": self.predictor_rms,
        })

    def collect_random_statistics(self, num_timesteps):
        #  initialize observation normalization with data from random agent

        self.obs_rms = RunningMeanStd(shape=(1, 7, 7, 3))

        curr_obs = self.obs
        collected_obss = [None] * (self.num_frames_per_proc * num_timesteps)
        for i in range(self.num_frames_per_proc * num_timesteps):
            # Do one agent-environment interaction

            action = torch.randint(0, self.env.action_space.n, (self.num_procs,))  # sample uniform actions
            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values
            collected_obss[i] = curr_obs
            curr_obs = obs

        self.obs = curr_obs
        exps = DictList()
        exps.obs = [collected_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc * num_timesteps)]

        images = [obs["image"] for obs in exps.obs]
        images = numpy.array(images)
        images = torch.tensor(images, dtype=torch.float)

        self.obs_rms.update(images)

    def calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):
        # exps is in (P x T x ...)
        #   - P is self.num_procs,
        #   - T is self.num_frames_per_proc,

        if self.running_norm_obs:
            obs = exps.obs.image * 15.0  # horrible harcoded normalized factor
            # normalize the observations for predictor and target networks
            norm_obs = torch.clamp(
                torch.div(
                    (obs - self.obs_rms.mean.to(exps.obs.image.device)),
                    torch.sqrt(self.obs_rms.var).to(exps.obs.image.device)),
                -5.0, 5.0)

            self.obs_rms.update(obs.cpu())  # update running mean
        else:
            # Without norm
            norm_obs = exps.obs.image

        obs = torch.transpose(torch.transpose(norm_obs, 1, 3), 2, 3)

        # obs_view = obs.view((self.num_procs, self.num_frames_per_proc) + obs.size()[1:])
        mask_view = exps.mask.view([self.num_procs, self.num_frames_per_proc])

        mask_view = mask_view.clone()
        mask_view[:, 0] = 0  # Fake each proc to start with new episode, so we can unpack
        mask_view = mask_view.view(-1)

        valid_sequences, valid_mask = self.acmodel.order_model.get_sequences(obs, mask_view)

        # Add all valid seq to order memory
        self.acmodel.order_model.add_seq_to_mem(valid_sequences)

        # ==========================================================================================
        # Evaluate each sequence

        self.acmodel.order_model.eval()

        scores = self.acmodel.order_model.eval_sequences(valid_sequences)

        # TODO should add to all positions :( Not just the first element in the batch
        scores = - scores.mean(dim=1)

        dst_intrinsic_r = dst_intrinsic_r.contiguous().view(-1)
        dst_intrinsic_r[valid_mask] = scores
        dst_intrinsic_r = dst_intrinsic_r.view([self.num_procs, self.num_frames_per_proc])
        dst_intrinsic_r = dst_intrinsic_r.transpose(0, 1).detach()

        # ==========================================================================================

        # with torch.no_grad():
        #     target = self.acmodel.random_target(obs)
        #     pred = self.acmodel.predictor_network(obs)
        #
        # diff_pred = (pred - target).pow_(2)
        #
        # # -- Calculate intrinsic & Normalize intrinsic rewards
        # int_rew = diff_pred.detach().mean(1)
        #
        # dst_intrinsic_r.copy_(int_rew.view((self.num_frames_per_proc, self.num_procs)))

        # Normalize intrinsic reward
        self.predictor_rff.reset()
        int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

        self.predictor_rms.update(int_rff.view(-1))
        # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
        dst_intrinsic_r.div_(torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device))

        dst_intrinsic_r = dst_intrinsic_r.transpose(0, 1)

        return dst_intrinsic_r


