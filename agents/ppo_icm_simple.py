"""
    DanM 2019
    inspired from https://github.com/lcswillems/torch-rl
"""
import numpy as np
import torch

from agents.two_v_base_general import TwoValueHeadsBaseGeneral
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter, ActionNames
from utils.format import preprocess_images
from torch_rl.utils import ParallelEnv


class PPOIcmSimple(TwoValueHeadsBaseGeneral):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, cfg, envs, acmodel, agent_data, **kwargs):
        num_frames_per_proc = getattr(cfg, "num_frames_per_proc", 128)
        discount = getattr(cfg, "discount", 0.99)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        max_grad_norm = getattr(cfg, "max_grad_norm", 40.0)
        recurrence = getattr(cfg, "recurrence", 1)
        clip_eps = getattr(cfg, "clip_eps", 0.)
        epochs = getattr(cfg, "epochs", 4)
        batch_size = getattr(cfg, "batch_size", 256)

        exp_used_pred = getattr(cfg, "exp_used_pred", 0.25)
        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)

        self.icm_beta_coeff = getattr(cfg, "beta_coeff", 0.2)
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

    def _get_optimizers(self, cfg, agent_data):
        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})

        self.lr = optimizer_args.lr
        # -- Prepare optimizers

        optimizer_args = vars(optimizer_args)

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), **optimizer_args)

        self.optimizer_agworld = getattr(torch.optim, optimizer)(
            self.acmodel.curiosity_model.parameters(), **optimizer_args)


        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_agworld.load_state_dict(agent_data["optimizer_agworld"])
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
        log_ret_int = []
        log_rew_int = []
        batch_ret_int = 0
        batch_rew_int = 0

        for epoch_no in range(self.epochs):

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

                if self.is_recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]
                    # Compute loss

                    if self.is_recurrent:
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

                    if self.is_recurrent and i < self.recurrence - 1:
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
                log_rew_int.append(batch_rew_int)

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
        logs["return_int"] = np.mean(log_ret_int)
        logs["reward_int"] = np.mean(log_rew_int)

        # add extra logs from intrinsic rewards
        for k in self.aux_logs:
            logs[k] = self.aux_logs[k]

        self.updates_cnt += 1
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
            frame_index = np.arange(padding, num_frames_per_proc-padding+1-recurrence, recurrence)
            indexes = np.resize(frame_index.reshape((1, -1)), (num_procs, len(frame_index)))
            indexes = indexes + np.arange(0, num_procs).reshape(-1, 1) * num_frames_per_proc
            indexes = indexes.reshape(-1)

        indexes = np.random.permutation(indexes)

        # Shift starting indexes by recurrence//2 half the time
        self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [
            indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_agworld": self.optimizer_agworld.state_dict(),
            "predictor_rms": self.predictor_rms,
        })

    def calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):

        # ------------------------------------------------------------------------------------------
        # Run worlds models & generate memories

        agworld_network = self.acmodel.curiosity_model

        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        beta = self.icm_beta_coeff # loss factor for Forward and Dynamics Models
        # ------------------------------------------------------------------------------------------
        # Get observations and full states
        f, prev_frame_exps = self.augment_exp(exps, agworld_network)

        # ------------------------------------------------------------------------------------------
        # -- Compute Intrinsic rewards

        pred_next_state = torch.zeros(num_frames_per_proc + 1, num_procs,
                                      agworld_network.embedding_size, device=device)
        embedding_states = torch.zeros(num_frames_per_proc + 1, num_procs,
                                      agworld_network.embedding_size, device=device)

        prev_actions = prev_frame_exps.actions_onehot

        agworld_network.eval()
        for i in range(num_frames_per_proc):
            cur_obs = f.obs_image[i]
            actions = f.actions_onehot[i]

            #Do one agent-environment interaction
            with torch.no_grad():
                _, embedding_states[i] = agworld_network(cur_obs)
                pred_next_state[i + 1] = agworld_network.forward_state(embedding_states[i], actions)

        agworld_network.train()
        dst_intrinsic_r = (pred_next_state[1:] - embedding_states[1:]).detach().pow(2).sum(2)
        dst_intrinsic_r[-1] = dst_intrinsic_r[:-1].mean()  # we don't have access to last next state :(

        # --Normalize intrinsic reward
        #self.predictor_rff.reset() # do you have to rest it every time ???
        int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

        self.predictor_rms.update(int_rff.view(-1))  # running mean statisics
        dst_intrinsic_r.div_(torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device))

        # -- Log info about intrinisc rewards distribution per action type
        self.log_intrinsic_reward_info(dst_intrinsic_r, f.action)

        # ------------------------------------------------------------------------------------------
        # -- Optimize ICM
        optimizer_agworld = self.optimizer_agworld
        max_grad_norm = self.max_grad_norm

        # _________ for all tensors below, T x P -> P x T -> P * T _______________________
        f = self.flip_back_experience(f)
        # ------------------------------------------------------------------------------------------

        loss_m_state = torch.nn.MSELoss()
        loss_m_act = torch.nn.CrossEntropyLoss()

        log_state_loss = []
        log_state_loss_same = []
        log_state_loss_diffs = []

        log_act_loss = []
        log_act_loss_same = []
        log_act_loss_diffs = []

        log_grad_agworld_norm = []

        for epoch_no in range(self.epochs):
            for inds in self._get_batches_starting_indexes(recurrence=1, padding=1):

                obs = f.obs_image[inds].detach()
                next_obs = f.obs_image[inds + 1].detach()
                crt_actions = f.action[inds].long().detach()
                crt_actions_one = f.actions_onehot[inds].detach()

                # take masks and convert them to 1D tensor for indexing
                # use next masks because done gives you the new game obs
                next_mask = f.mask[inds + 1].long().detach()
                next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                _, crt_state_embedding = agworld_network(obs)
                _, next_state_embedding = agworld_network(next_obs)

                pred_act = agworld_network.forward_action(crt_state_embedding, next_state_embedding)
                pred_state = agworld_network.forward_state(crt_state_embedding, crt_actions_one)

                act_batch_loss = loss_m_act(pred_act, crt_actions)
                state_batch_loss = loss_m_state(pred_state, next_state_embedding)

                log_act_loss.append(act_batch_loss.item())
                log_state_loss.append(state_batch_loss.item())
                # ======================================================================================
                # Compute state & action loss for same states and different states

                # if all episodes ends at once, can't compute same/diff losses
                if next_mask.sum() == 0:
                    continue

                next_mask_as_bool = next_mask.to(torch.bool)
                same = (obs[next_mask_as_bool] == next_obs[next_mask_as_bool]).all(1).all(1).all(1)

                s_pred_act = pred_act[next_mask_as_bool]
                s_crt_act = crt_actions[next_mask_as_bool]

                s_pred_state = pred_state[next_mask_as_bool]
                s_crt_state = (next_state_embedding.detach())[next_mask_as_bool]

                # if all are same/diff take care to empty tensors
                if same.sum() == same.shape[0]:
                    act_batch_loss_same = loss_m_act(s_pred_act[same], s_crt_act[same])
                    state_batch_loss_same = loss_m_state(s_pred_state[same], s_crt_state[same])

                elif same.sum() == 0:
                    act_batch_loss_diff = loss_m_act(s_pred_act[~same], s_crt_act[~same])
                    state_batch_loss_diffs = loss_m_state(s_pred_state[~same], s_crt_state[~same])

                else:
                    act_batch_loss_same = loss_m_act(s_pred_act[same], s_crt_act[same])
                    act_batch_loss_diff = loss_m_act(s_pred_act[~same], s_crt_act[~same])

                    state_batch_loss_same = loss_m_state(s_pred_state[same], s_crt_state[same])
                    state_batch_loss_diffs = loss_m_state(s_pred_state[~same], s_crt_state[~same])

                log_state_loss_same.append(state_batch_loss_same.item())
                log_state_loss_diffs.append(state_batch_loss_diffs.item())

                log_act_loss_same.append(act_batch_loss_same.item())
                log_act_loss_diffs.append(act_batch_loss_diff.item())
                # =======================================================================================

                # Do backpropagation and optimization steps

                total_loss = (1 - beta) * act_batch_loss + beta * state_batch_loss

                optimizer_agworld.zero_grad()

                total_loss.backward()
                grad_agworld_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in agworld_network.parameters()
                    if p.grad is not None
                ) ** 0.5
                log_grad_agworld_norm.append(grad_agworld_norm)
                torch.nn.utils.clip_grad_norm_(agworld_network.parameters(), max_grad_norm)
                optimizer_agworld.step()

        # ------------------------------------------------------------------------------------------
        # Log some values
        self.aux_logs['next_state_loss'] = np.mean(log_state_loss)
        self.aux_logs['next_action_loss'] = np.mean(log_act_loss)
        self.aux_logs['grad_norm_icm'] = np.mean(log_grad_agworld_norm)

        self.aux_logs['next_state_loss_same'] = np.mean(log_state_loss_same)
        self.aux_logs['next_state_loss_diffs'] = np.mean(log_state_loss_diffs)

        self.aux_logs['next_act_loss_same'] = np.mean(log_act_loss_same)
        self.aux_logs['next_act_loss_diffs'] = np.mean(log_act_loss_diffs)

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
            int_r = np.array([0], dtyp=np.float32)

        self.aux_logs["move_forward_mean_int_r"] = np.mean(int_r)
        self.aux_logs["move_forward_var_int_r"] = np.var(int_r)
        self.aux_logs["move_forward_max_int_r"] = np.max(int_r)
        self.aux_logs["move_forward_min_int_r"] = np.min(int_r)

        # -- TURNING INTRINSIC REWARDS
        int_r = intrinsic_rewards[(actions == ActionNames.TURN_LEFT) | (actions == ActionNames.TURN_RIGHT)].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtyp=np.float32)

        self.aux_logs["turn_mean_int_r"] = np.mean(int_r)
        self.aux_logs["turn_var_int_r"] = np.var(int_r)
        self.aux_logs["turn_max_int_r"] = np.max(int_r)
        self.aux_logs["turn_min_int_r"] = np.min(int_r)

        # -- OBJECT PICKING UP / DROPPING INTRINSIC REWARDS
        int_r = intrinsic_rewards[(actions == ActionNames.PICK_UP) | (actions == ActionNames.DROP) ].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtyp=np.float32)

        self.aux_logs["obj_interactions_mean_int_r"] = np.mean(int_r)
        self.aux_logs["obj_interactions_var_int_r"] = np.var(int_r)
        self.aux_logs["obj_interactions_max_int_r"] = np.max(int_r)
        self.aux_logs["obj_interactions_min_int_r"] = np.min(int_r)

        # -- OBJECT TOGGLE (OPEN DOORS, BREAKING BOXES)
        int_r = intrinsic_rewards[actions == ActionNames.INTERACT].cpu().numpy()
        if len(int_r) == 0:
            int_r = np.array([0], dtyp=np.float32)

        self.aux_logs["obj_toggle_mean_int_r"] = np.mean(int_r)
        self.aux_logs["obj_toggle_var_int_r"] = np.var(int_r)
        self.aux_logs["obj_toggle_max_int_r"] = np.max(int_r)
        self.aux_logs["obj_toggle_min_int_r"] = np.min(int_r)

