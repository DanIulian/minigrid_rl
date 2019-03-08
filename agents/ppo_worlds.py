"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy
import torch

from agents.two_v_base_general import TwoValueHeadsBaseGeneral
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter
from utils.format import preprocess_images
import torch_rl


class PPOWorlds(TwoValueHeadsBaseGeneral):
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
        exp_used_pred = getattr(cfg, "exp_used_pred", 0.25)

        self.recurrence_worlds = getattr(cfg, "recurrence_worlds", 16)

        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)

        self.nminibatches = getattr(cfg, "nminibatches", 4)

        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, exp_used_pred)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), lr, eps=adam_eps)

        # Prepare intrinsic generators
        # self.acmodel.random_target.eval()
        self.predictor_rms = RunningMeanStd()
        self.predictor_rff = RewardForwardFilter(gamma=self.discount)

        # self.optimizer_predictor = getattr(torch.optim, optimizer)(
        #     self.acmodel.predictor_network.parameters(), lr,eps=adam_eps)

        self.optimizer_envworld = getattr(torch.optim, optimizer)(
            self.acmodel.envworld_network.parameters(), lr, eps=adam_eps)

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            # self.optimizer_predictor.load_state_dict(agent_data["optimizer_predictor"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

        self.batch_num = 0

        if self.running_norm_obs:
            self.collect_random_statistics(50)

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        log_entropies = []
        log_values_ext = []
        log_values_int = []
        log_policy_losses = []
        log_value_ext_losses = []
        log_value_int_losses = []
        log_grad_norms = []

        for epoch_no in range(self.epochs):
            # Initialize log values

            # Loop for Policy

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

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                    # Update Predictor loss
                    #
                    # # Optimize intrinsic reward generator using only a percentage of experiences
                    # norm_obs = sb.obs.image
                    # obs = torch.transpose(torch.transpose(norm_obs, 1, 3), 2, 3)
                    #
                    # with torch.no_grad():
                    #     target = self.acmodel.random_target(obs)
                    #
                    # pred = self.acmodel.predictor_network(obs)
                    # diff_pred = (pred - target).pow_(2)
                    #
                    # # Optimize intrinsic reward generator using only a percentage of experiences
                    # loss_pred = diff_pred.mean(1)
                    # mask = torch.rand(loss_pred.shape[0])
                    # mask = (mask < self.exp_used_pred).type(torch.FloatTensor).to(loss_pred.device)
                    # loss_pred = (loss_pred * mask).sum() / torch.max(mask.sum(),
                    #                                                  torch.Tensor([1]).to(
                    #                                                      loss_pred.device))
                    #
                    # self.optimizer_predictor.zero_grad()
                    # loss_pred.backward()
                    # grad_norm = sum(
                    #     p.grad.data.norm(2).item() ** 2 for p in
                    #     self.acmodel.predictor_network.parameters()
                    #     if p.grad is not None
                    # ) ** 0.5
                    #
                    # torch.nn.utils.clip_grad_norm_(self.acmodel.predictor_network.parameters(),
                    #                                self.max_grad_norm)
                    # self.optimizer_predictor.step()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_ext_value /= self.recurrence
                batch_int_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_ext_loss /= self.recurrence
                batch_value_int_loss /= self.recurrence
                batch_loss /= self.recurrence

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

        # ------------------------------------------------------------------------------------------
        # Run worlds models

        shape = torch.Size([self.num_procs, self.num_frames_per_proc])
        estates = exps.states.view(shape + exps.states.size()[1:]).transpose(0, 1)
        obs_igms = exps.obs.image  #[:, :, :, [0]]
        eobs = obs_igms.view(shape + obs_igms.size()[1:]).transpose(0, 1)

        estates = torch.transpose(torch.transpose(estates, 2, 4), 3, 4)
        eobs = torch.transpose(torch.transpose(eobs, 2, 4), 3, 4)

        eactions = exps.action.view(shape).transpose(0, 1)
        emasks = exps.mask.view(shape + exps.mask.size()[1:]).transpose(0, 1)

        envworld_network = self.acmodel.envworld_network

        envworld_mems = torch.zeros(self.num_frames_per_proc, self.num_procs,
                                    envworld_network.memory_size, device=self.device)

        # Offset vector of actions -> action[i] == action[i+1] !!!!!
        # Make one hot vector representations for actions
        eactions_onehot = torch.zeros([self.num_frames_per_proc, self.num_procs,
                                       self.env.action_space.n], device=self.device)
        eactions_onehot.scatter_(2, eactions.unsqueeze(2).long(), 1.)

        prev_actions = torch.zeros((1, self.num_procs, self.env.action_space.n), device=self.device)
        eactions_onehot = torch.cat([prev_actions, eactions_onehot], dim=0)

        for i in range(self.num_frames_per_proc - 1):
            obs = eobs[i]
            masks = emasks[i]

            # Do one agent-environment interaction
            with torch.no_grad():
                obs_predic, envworld_mems[i+1] = envworld_network(obs, envworld_mems[i] * masks,
                                                                  eactions_onehot[i],
                                                                  eactions_onehot[i + 1])

            # TODO add somewhere next memory! to use @ next step

        loss_m = torch.nn.MSELoss()

        # for all tensors below, T x P -> P x T -> P * T
        envworld_mems = envworld_mems.transpose(0, 1).reshape(-1, *envworld_mems.shape[2:])
        eobs = torch.transpose(torch.transpose(obs_igms, 1, 3), 2, 3)
        eactions_onehot = eactions_onehot.transpose(0, 1).reshape(-1, *eactions_onehot.shape[2:])

        log_envworld_loss = []
        for inds in self._get_batches_starting_indexes(recurrence=self.recurrence_worlds):

            envworld_mem = envworld_mems[inds]

            envworld_batch_loss = 0

            for i in range(self.recurrence_worlds-1):
                sb = exps[inds + i]
                obs = eobs[inds + i]
                next_obs = eobs[inds + i + 1]
                prev_actions = eactions_onehot[inds + i]
                crt_actions = eactions_onehot[inds + i + 1]

                # Compute loss
                obs_predic, envworld_mem = envworld_network(obs, envworld_mem * sb.mask,
                                                            prev_actions, crt_actions)
                envworld_batch_loss += loss_m(obs_predic, next_obs.detach())

                # norm_obs = sb.obs.image
                # obs = torch.transpose(torch.transpose(norm_obs, 1, 3), 2, 3)
                #
                # with torch.no_grad():
                #     target = self.acmodel.random_target(obs)

                # # Update memories for next epoch
                # if self.acmodel.recurrent and i < self.recurrence - 1:
                #     exps.memory[inds + i + 1] = memory.detach()

            envworld_batch_loss /= self.recurrence_worlds

            self.optimizer_envworld.zero_grad()
            envworld_batch_loss.backward()
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in envworld_network.parameters()
                if p.grad is not None
            ) ** 0.5
            torch.nn.utils.clip_grad_norm_(envworld_network.parameters(), self.max_grad_norm)
            self.optimizer_envworld.step()

            log_envworld_loss.append(envworld_batch_loss.item())

        print("OBS___PREDICT:")
        print((obs_predic - obs).abs().sum())
        print(numpy.mean(log_envworld_loss))

        # ------------------------------------------------------------------------------------------

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

    def _get_batches_starting_indexes(self, recurrence=None):
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
        if recurrence is None:
            recurrence = self.recurrence

        indexes = numpy.arange(0, self.num_frames, recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
            indexes += recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            # "optimizer_predictor": self.optimizer_predictor.state_dict(),
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

        """
        replicate (should normalize by a running mean):
            X_r -- random target
            X_r_hat -- predictor

            self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
            self.max_feat = tf.reduce_max(tf.abs(X_r))
            self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
            self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

            targets = tf.stop_gradient(X_r)
            # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
            self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
            mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
            mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
            self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

        """
        return

        if self.running_norm_obs:
            obs = exps.obs.image * 15.0  # horrible harcoded normalized factor
            # nurmalize the observations for predictor and target networks
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

        with torch.no_grad():
            target = self.acmodel.random_target(obs)
            pred = self.acmodel.predictor_network(obs)

        diff_pred = (pred - target).pow_(2)

        # -- Calculate intrinsic & Normalize intrinsic rewards
        int_rew = diff_pred.detach().mean(1)

        dst_intrinsic_r.copy_(int_rew.view((self.num_frames_per_proc, self.num_procs)))

        # Normalize intrinsic reward
        # self.predictor_rff.reset() # do you have to rest it every time ???
        int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

        self.predictor_rms.update(int_rff.view(-1))
        # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
        dst_intrinsic_r.div_(torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device))

    def add_extra_experience(self, exps: DictList):
        # Process
        full_states = [self.obss[i][j]["state"]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)]

        exps.states = preprocess_images(full_states, device=self.device)






