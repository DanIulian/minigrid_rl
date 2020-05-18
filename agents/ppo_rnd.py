"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy as np
import torch

from agents.two_v_base_general import TwoValueHeadsBaseGeneral
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter, ActionNames
from utils.format import preprocess_images
from torch_rl.utils import ParallelEnv


class PPORND(TwoValueHeadsBaseGeneral):
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

        exp_used_pred = getattr(cfg, "exp_used_pred", 0.25)
        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)

        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)

        self.num_minibatch = getattr(cfg, "num_minibatch", 8)
        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)
        self.out_dir = getattr(cfg, "out_dir", None)

        self.save_experience_batch = getattr(cfg, "save_experience_batch", 5)

        envs = ParallelEnv(envs)
        super().__init__(envs, acmodel, num_frames_per_proc, discount,
                         gae_lambda, entropy_coef, value_loss_coef,
                         max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         exp_used_pred, intrinsic_reward_fn=self.calculate_intrinsic_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        # -- Prepare intrinsic generators
        self.acmodel.random_target.eval()
        self.predictor_rms = RunningMeanStd()
        self.predictor_rff = RewardForwardFilter(gamma=self.discount)

        # -- Prepare optimizers
        self._get_optimizers(cfg, agent_data)

        self.batch_num = 0
        self.updates_cnt = 0

        # get width and height of the observation space for position normalization
        self.env_width = envs[0][0].unwrapped.width
        self.env_height = envs[0][0].unwrapped.height

        # -- Previous batch of experiences last frame
        self.prev_frame_exps = None

        # -- Init evaluator envs

        # remember some log values from intrinsic rewards computation
        self.aux_logs = {}

        if self.running_norm_obs:
            self.collect_random_statistics(50)

    def _get_optimizers(self, cfg, agent_data):
        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})

        self.lr = optimizer_args.lr
        # -- Prepare optimizers

        optimizer_args = vars(optimizer_args)

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), **optimizer_args)

        self.optimizer_predictor = getattr(torch.optim, optimizer)(
            self.acmodel.predictor_network.parameters(), **optimizer_args)

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_predictor.load_state_dict(agent_data["optimizer_predictor"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        # Initialize log values
        log_entropies = []
        log_values_ext = []
        log_values_int = []
        log_policy_losses = []
        log_value_ext_losses = []
        log_value_int_losses = []
        log_grad_norms = []

        for epoch_no in range(self.epochs):

            for inds in self._get_batches_start_idx_v2():
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

        # Log some values
        logs["entropy"] = np.mean(log_entropies)
        logs["value_ext"] = np.mean(log_values_ext)
        logs["value_int"] = np.mean(log_values_int)
        logs["value"] = logs["value_ext"] + logs["value_int"]
        logs["policy_loss"] = np.mean(log_policy_losses)
        logs["value_ext_loss"] = np.mean(log_value_ext_losses)
        logs["value_int_loss"] = np.mean(log_value_int_losses)
        logs["value_loss"] = logs["value_int_loss"] + logs["value_ext_loss"]
        logs["grad_norm"] = np.mean(log_grad_norms)

        # add extra logs from intrinsic rewards
        for k in self.aux_logs:
            logs[k] = self.aux_logs[k]

        return logs

    def _get_batches_start_idx_v2(self, recurrence=None, padding=0):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """
        num_frames_per_proc = self.num_frames_per_proc
        num_procs = self.num_procs
        num_minibatch = self.num_minibatch

        if recurrence is None:
            recurrence = self.recurrence

        #  --Matrix of size (num_minibatches, num_processes) witch contains the first frame index
        #  --for each process (0 for proc1, 128 for proc2 ....)
        proc_frames_start_idx = np.tile(
            np.arange(0, self.num_frames, num_frames_per_proc), (num_minibatch, 1))
        #  -- Starting index for each proc
        proc_first_idx = np.random.randint(0, recurrence, size=(num_minibatch, num_procs))

        nr_batches_per_proc = (num_frames_per_proc // recurrence) - 1
        if nr_batches_per_proc < 1:
            raise ValueError("Recurrence bigger than rollout")

        #  -- First obs that can be used from each process rollout
        proc_batch_start_idx = proc_frames_start_idx + proc_first_idx

        batches_indexes = np.concatenate(
            [proc_batch_start_idx + recurrence * i for i in range(nr_batches_per_proc)],
            axis=1)

        batch_size = min(len(batches_indexes[0]), self.batch_size)
        batches_starting_indexes = [
                np.random.choice(
                    np.random.permutation(batches_indexes[i]),
                    batch_size, replace=False) for i in range(num_minibatch)]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_predictor": self.optimizer_predictor.state_dict(),
            "predictor_rms": self.predictor_rms,
        })

    def calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):

        target_network = self.acmodel.random_target
        predictor_network = self.acmodel.predictor_network
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        epochs_no = self.epochs

        #  Get observations and full states
        f, prev_frame_exps = self.augment_exp(exps, predictor_network)

        #  Compute Intrinsic rewards
        predicted_embeddings = torch.zeros(num_frames_per_proc,
                                          num_procs,
                                          predictor_network.embedding_size,
                                          device=device)
        predictor_network.eval()
        for i in range(num_frames_per_proc):
            cur_obs = f.obs_image[i]

            with torch.no_grad():
                f.agworld_embs[i] = target_network(cur_obs).detach()
                predicted_embeddings[i] = predictor_network(cur_obs).detach()

        predictor_network.train()

        dst_intrinsic_r = (predicted_embeddings - f.agworld_embs).pow_(2).mean(-1).detach()
        # Normalize intrinsic reward
        self.predictor_rff.reset()
        int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)
        for i in reversed(range(self.num_frames_per_proc)):
            int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

        self.predictor_rms.update(int_rff.view(-1))
        dst_intrinsic_r.div_(torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device))

        # -- Log info about intrinisc rewards distribution per action type
        self.log_intrinsic_reward_info(dst_intrinsic_r, f.action)

        # ----------------------------------------------------------------------------------
        # -- Optimize RND

        optimizer_predictor = self.optimizer_predictor
        max_grad_norm = self.max_grad_norm

        # _________ for all tensors below, T x P -> P x T -> P * T _______________________
        f = self.flip_back_experience(f)
        # ------------------------------------------------------------------------------------------

        loss_m_state = torch.nn.MSELoss(reduction='none')
        log_state_loss = []
        log_grad_agworld_norm = []

        for epoch_no in range(epochs_no):
            for inds in self._get_batches_start_idx_v2(recurrence=1):
                obs = f.obs_image[inds].detach()
                crt_actions = f.action[inds].long().detach()
                crt_actions_one = f.actions_onehot[inds].detach()

                crt_target_embs = f.agworld_embs[inds].detach()
                crt_predicted_embs = predictor_network(obs)

                state_batch_loss = loss_m_state(crt_predicted_embs, crt_target_embs).mean(-1)

                # Optimize intrinsic reward generator using only a percentage of experiences
                mask = torch.rand(state_batch_loss.shape[0])
                mask = (mask < self.exp_used_pred).type(torch.float).to(device)

                loss_rnd = (state_batch_loss * mask).sum() / torch.max(mask.sum(), torch.ones(1).to(device))
                log_state_loss.append(loss_rnd.cpu().item())
                # ======================================================================================
                # Do backpropagation and optimization steps

                optimizer_predictor.zero_grad()

                loss_rnd.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in predictor_network.parameters()
                    if p.grad is not None
                ) ** 0.5
                log_grad_agworld_norm.append(grad_norm)
                torch.nn.utils.clip_grad_norm_(predictor_network.parameters(), max_grad_norm)
                optimizer_predictor.step()

        # ------------------------------------------------------------------------------------------
        # Log some values
        self.aux_logs['state_loss'] = np.mean(log_state_loss)
        self.aux_logs['grad_norm_icm'] = np.mean(log_grad_agworld_norm)

        return dst_intrinsic_r

    def add_extra_experience(self, exps: DictList):
        # Process
        if "position" in self.obss[0][0].keys():
            full_positions = [self.obss[i][j]["position"]
                              for j in range(self.num_procs)
                              for i in range(self.num_frames_per_proc)]

            max_pos_value = max(self.env_height, self.env_width)
            exps.position = preprocess_images(full_positions,
                                              device=self.device,
                                              max_image_value=max_pos_value,
                                              normalize=False)
        # Process
        if "state" in self.obss[0][0].keys():
            full_states = [self.obss[i][j]["state"]
                           for j in range(self.num_procs)
                           for i in range(self.num_frames_per_proc)]

            exps.states = preprocess_images(full_states, device=self.device)

        exps.obs_image = exps.obs.image

    def log_intrinsic_reward_info(self, intrinsic_rewards, actions):

        self.aux_logs["mean_intrinsic_rewards"] = np.mean(intrinsic_rewards.cpu().numpy())
        self.aux_logs["min_intrinsic_rewards"] = np.min(intrinsic_rewards.cpu().numpy())
        self.aux_logs["max_intrinsic_rewards"] = np.max(intrinsic_rewards.cpu().numpy())
        self.aux_logs["var_intrinsic_rewards"] = np.var(intrinsic_rewards.cpu().numpy())

        # -- MOVE FORWARD INTRINSIC REWARDS
        int_r = intrinsic_rewards[actions == ActionNames.MOVE_FORWARD].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtype=np.float32)

        self.aux_logs["move_forward_mean_int_r"] = np.mean(int_r)
        self.aux_logs["move_forward_var_int_r"] = np.var(int_r)
        self.aux_logs["move_forward_max_int_r"] = np.max(int_r)
        self.aux_logs["move_forward_min_int_r"] = np.min(int_r)

        # -- TURNING INTRINSIC REWARDS
        int_r = intrinsic_rewards[(actions == ActionNames.TURN_LEFT) | (actions == ActionNames.TURN_RIGHT)].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtype=np.float32)

        self.aux_logs["turn_mean_int_r"] = np.mean(int_r)
        self.aux_logs["turn_var_int_r"] = np.var(int_r)
        self.aux_logs["turn_max_int_r"] = np.max(int_r)
        self.aux_logs["turn_min_int_r"] = np.min(int_r)

        # -- OBJECT PICKING UP / DROPPING INTRINSIC REWARDS
        int_r = intrinsic_rewards[(actions == ActionNames.PICK_UP) | (actions == ActionNames.DROP)].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtype=np.float32)

        self.aux_logs["obj_interactions_mean_int_r"] = np.mean(int_r)
        self.aux_logs["obj_interactions_var_int_r"] = np.var(int_r)
        self.aux_logs["obj_interactions_max_int_r"] = np.max(int_r)
        self.aux_logs["obj_interactions_min_int_r"] = np.min(int_r)

        # -- OBJECT TOGGLE (OPEN DOORS, BREAKING BOXES)
        int_r = intrinsic_rewards[actions == ActionNames.INTERACT].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtype=np.float32)

        self.aux_logs["obj_toggle_mean_int_r"] = np.mean(int_r)
        self.aux_logs["obj_toggle_var_int_r"] = np.var(int_r)
        self.aux_logs["obj_toggle_max_int_r"] = np.max(int_r)
        self.aux_logs["obj_toggle_min_int_r"] = np.min(int_r)

