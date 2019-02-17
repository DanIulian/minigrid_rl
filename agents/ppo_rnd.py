"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy
import torch
import torch.nn.functional as F

from agents.two_v_base import TwoValueHeadsBase


class PPORND(TwoValueHeadsBase):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, cfg, envs, acmodel, agent_data, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = getattr(cfg, "num_frames_per_proc", 128)
        discount = getattr(cfg, "discount", 0.99)
        lr = getattr(cfg, "lr", 7e-4)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        recurrence = getattr(cfg, "recurrence", 4)
        adam_eps = getattr(cfg, "adam_eps", 1e-5)
        clip_eps = getattr(cfg, "clip_eps", 0.)
        epochs = getattr(cfg, "epochs", 4)
        batch_size = getattr(cfg, "batch_size", 256)
        optimizer = getattr(cfg, "optimizer", "Adam")

        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        self.optimizer = getattr(torch.optim, optimizer)(self.acmodel.parameters(), lr,
                                                         eps=adam_eps)

        if "optimizer" in agent_data:
            self.optimizer.load_state_dict(agent_data["optimizer"])

        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values_ext = []
            log_values_int = []
            log_policy_losses = []
            log_value_ext_losses = []
            log_value_int_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_ext_value = 0
                batch_int_value = 0
                batch_policy_loss = 0
                batch_value_ext_loss = 0
                batch_value_int_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, vvalue, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, vvalue = self.acmodel(sb.obs)

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

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()
                    if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values_ext.append(batch_ext_value)
                log_values_int.append(batch_int_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_ext_losses.append(batch_value_ext_loss)
                log_value_int_losses.append(batch_value_int_loss)
                log_grad_norms.append(grad_norm)

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
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({"optimizer": self.optimizer.state_dict()})
