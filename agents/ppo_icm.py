"""
    DanM 2019
    inspired from https://github.com/lcswillems/torch-rl
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from copy import deepcopy

from agents.two_v_base_general import TwoValueHeadsBaseGeneral
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter
from utils.format import preprocess_images


class PPOIcm(TwoValueHeadsBaseGeneral):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

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

        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})

        exp_used_pred = getattr(cfg, "exp_used_pred", 0.25)
        preprocess_obss = kwargs.get("preprocess_obss", None)
        reshape_reward = kwargs.get("reshape_reward", None)
        eval_envs = kwargs.get("eval_envs", [])

        self.recurrence_worlds = getattr(cfg, "recurrence_worlds", 16)
        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)
        self.nminibatches = getattr(cfg, "nminibatches", 4)
        self.out_dir = getattr(cfg, "out_dir", None)
        self.pre_fill_memories = pre_fill_memories = getattr(cfg, "pre_fill_memories", 1)

        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, exp_used_pred)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        # -- Prepare intrinsic generators
        # self.acmodel.random_target.eval()
        self.predictor_rms = RunningMeanStd()
        self.predictor_rff = RewardForwardFilter(gamma=self.discount)

        # -- Prepare optimizers
        optimizer_args = vars(optimizer_args)

        self.optimizer_policy = getattr(torch.optim, optimizer)(
            self.acmodel.policy_model.parameters(), **optimizer_args)

        self.optimizer_agworld = getattr(torch.optim, optimizer)(
            self.acmodel.curiosity_model.parameters(), **optimizer_args)

        self.optimizer_evaluator = getattr(torch.optim, optimizer)(
            self.acmodel.evaluator_network.parameters(), **optimizer_args)

        self.optimizer_evaluator_base = getattr(torch.optim, optimizer)(
            self.acmodel.base_evaluator_network.parameters(), **optimizer_args)


        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_agworld.load_state_dict(agent_data["optimizer_agworld"])
            self.optimizer_evaluator.load_state_dict(agent_data["optimizer_evaluator"])
            self.optimizer_evaluator_base.load_state_dict(agent_data["optimizer_evaluator_base"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

        self.batch_num = 0
        self.updates_cnt = 0

        # get width and height of the observation space for position normalization
        self.env_width = envs[0][0].unwrapped.width
        self.env_height = envs[0][0].unwrapped.height

        if self.running_norm_obs:
            self.collect_random_statistics(50)

        # -- Previous batch of experiences last frame
        self.prev_frame_exps = None

        # -- Init evaluator envs
        self.eval_envs = None
        self.eval_memory = None
        self.eval_mask = None
        self.eval_icm_memory = None
        self.eval_dir = None

        if len(eval_envs) > 0:
            self.eval_envs = self.init_evaluator(eval_envs)
            self.eval_dir = os.path.join(self.out_dir, "eval")
            if not os.path.isdir(self.eval_dir):
                os.mkdir(self.eval_dir)

        # remember some log values from intrinsic rewards computation
        self.aux_logs = {}

    def init_evaluator(self, envs):
        from torch_rl.utils import ParallelEnv
        device = self.device
        acmodel = self.acmodel

        eval_envs = ParallelEnv(envs)
        obs = eval_envs.reset()

        if self.acmodel.recurrent:
            self.eval_memory = torch.zeros(len(obs), acmodel.memory_size, device=device)

        self.eval_agworld_memory = torch.zeros(len(obs), acmodel.curiosity_model.memory_size,
                                           device=device)
        self.eval_mask = torch.ones(len(obs), device=device)
        return eval_envs

    def augment_exp(self, exps):

        # from exp (P * T , ** ) -> (T, P, **)
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        env = self.env
        agworld_network = self.acmodel.curiosity_model

        shape = torch.Size([num_procs, num_frames_per_proc])
        frame_exp = Namespace()

        # ------------------------------------------------------------------------------------------
        # Redo in format T x P

        for k, v in exps.items():
            if k == "obs":
                continue
            setattr(frame_exp, k, v.view(shape + v.size()[1:]).transpose(0, 1).contiguous())

        def inverse_img(t, ii):
            return torch.transpose(torch.transpose(t, ii, ii+2), ii+1, ii+2).contiguous()

        frame_exp.obs_image = inverse_img(frame_exp.obs_image, 2)
        #frame_exp.states = inverse_img(frame_exp.states, 2)

        def gen_memory(ss):
            return torch.zeros(num_frames_per_proc, num_procs, ss, device=device)

        frame_exp.agworld_mems = gen_memory(agworld_network.memory_size)
        frame_exp.agworld_embs = gen_memory(agworld_network.embedding_size)

        frame_exp.actions_onehot = gen_memory(env.action_space.n)
        frame_exp.actions_onehot.scatter_(2, frame_exp.action.unsqueeze(2).long(), 1.)

        # ------------------------------------------------------------------------------------------
        # Save last frame exp

        last_frame_exp = Namespace()
        for k, v in frame_exp.__dict__.items():
            if k == "obs":
                continue
            setattr(last_frame_exp, k, v[-1].clone())

        prev_frame_exps = self.prev_frame_exps
        if self.prev_frame_exps is None:
            prev_frame_exps = deepcopy(last_frame_exp)
            for k, v in prev_frame_exps.__dict__.items():
                v.zero_()

        self.prev_frame_exps = last_frame_exp

        # ------------------------------------------------------------------------------------------
        # Fill memories with past

        frame_exp.agworld_mems[0] = prev_frame_exps.agworld_mems
        frame_exp.agworld_embs[0] = prev_frame_exps.agworld_embs

        return frame_exp, prev_frame_exps

    @staticmethod
    def flip_back_experience(exp):
        # for all tensors below, T x P -> P x T -> P * T
        for k, v in exp.__dict__.items():
            setattr(exp, k, v.transpose(0, 1).reshape(-1, *v.shape[2:]).contiguous())
        return exp

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
        log_ret_int = []
        log_rew_int = []
        batch_ret_int = 0
        batch_rew_int = 0

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
        # TODO Check this ; Bad fix
        if recurrence is None:
            if self.batch_num % 2 == 1:
                indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
                indexes += recurrence // 2
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
            "optimizer_evaluator": self.optimizer_evaluator.state_dict(),
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
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float)

        self.obs_rms.update(images)

    def calculate_intrinsic_reward(self, exps: DictList, dst_intrinsic_r: torch.Tensor):

        # ------------------------------------------------------------------------------------------
        # Run worlds models & generate memories

        agworld_network = self.acmodel.curiosity_model
        evaluator_network = self.acmodel.evaluator_network
        base_eval_network = self.acmodel.base_evaluator_network

        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        beta = 0.2  # loss factor for Forward and Dynamics Models

        # ------------------------------------------------------------------------------------------
        # Get observations and full states
        f, prev_frame_exps = self.augment_exp(exps)
        # ------------------------------------------------------------------------------------------
        if self.pre_fill_memories:
            prev_actions = prev_frame_exps.actions_onehot
            for i in range(num_frames_per_proc - 1):
                obs = f.obs_image[i]
                masks = f.mask[i]

                # Do one agent-environment interaction
                with torch.no_grad():
                    _, f.agworld_mems[i + 1], f.agworld_embs[i] = \
                        agworld_network(obs, f.agworld_mems[i] * masks, prev_actions)
                    prev_actions = f.actions_onehot[i]

        # ------------------------------------------------------------------------------------------
        # -- Compute Intrinsic rewards

        pred_next_state = torch.zeros(num_frames_per_proc + 1, num_procs,
                                      agworld_network.memory_size, device=device)
        next_state = torch.zeros(num_frames_per_proc + 1, num_procs,
                                 agworld_network.memory_size, device=device)

        prev_actions = prev_frame_exps.actions_onehot
        prev_memory = prev_frame_exps.agworld_mems
        for i in range(num_frames_per_proc):
            obs = f.obs_image[i]
            masks = f.mask[i]
            actions = f.actions_onehot[i]

            #Do one agent-environment interaction
            with torch.no_grad():
                _, next_state[i], embs = agworld_network(
                    obs, prev_memory * masks, prev_actions)

                pred_next_state[i + 1] = agworld_network.forward_state(
                    next_state[i], actions)

                prev_actions = actions
                prev_memory = next_state[i]

        # TODO fix the last intrinsic reward value as it is pred - 0
        dst_intrinsic_r = (pred_next_state[1:] - next_state[1:]).detach().pow(2).sum(2)

        # --Normalize intrinsic reward
        self.predictor_rff.reset() # do you have to rest it every time ???
        int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

        self.predictor_rms.update(int_rff.view(-1))  # running mean statisics
        # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
        dst_intrinsic_r.div_(torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device))

        # ------------------------------------------------------------------------------------------
        # -- Optimize ICM
        optimizer_evaluator = self.optimizer_evaluator
        optimizer_evaluator_base = self.optimizer_evaluator_base
        optimizer_agworld = self.optimizer_agworld
        recurrence_worlds = self.recurrence_worlds

        max_grad_norm = self.max_grad_norm

        # ------------------------------------------------------------------------------------------
        # _________ for all tensors below, T x P -> P x T -> P * T _______________________
        f = self.flip_back_experience(f)
        # ------------------------------------------------------------------------------------------

        loss_m_state = torch.nn.MSELoss()
        loss_m_act = torch.nn.CrossEntropyLoss()
        loss_m_eval = torch.nn.CrossEntropyLoss()
        loss_m_eval_base = torch.nn.CrossEntropyLoss()

        log_state_loss = []
        log_state_loss_same = []
        log_state_loss_diffs = []

        log_act_loss = []
        log_act_loss_same = []
        log_act_loss_diffs = []

        log_evaluator_loss = []
        log_base_evaluator_loss = []

        for inds in self._get_batches_starting_indexes(recurrence=recurrence_worlds, padding=1):

            agworld_mem = f.agworld_mems[inds].detach()
            new_agworld_emb = [None] * recurrence_worlds
            new_agworld_mem = [None] * recurrence_worlds

            state_batch_loss = 0
            state_batch_loss_same = 0
            state_batch_loss_diffs = 0

            act_batch_loss = 0
            act_batch_loss_same = 0
            act_batch_loss_diff = 0
            evaluator_batch_loss = 0
            evaluator_base_batch_loss = 0

            log_grad_agworld_norm = []
            log_grad_eval_norm = []

            # -- Agent world
            for i in range(recurrence_worlds):
                obs = f.obs_image[inds + i].detach()
                mask = f.mask[inds + i]
                prev_actions_one = f.actions_onehot[inds + i - 1].detach()
                pos = f.position[inds + i].detach()

                # Forward pass Agent Net for memory
                _, agworld_mem, new_agworld_emb[i] = agworld_network(
                    obs, agworld_mem * mask, prev_actions_one)
                new_agworld_mem[i] = agworld_mem

                # Compute loss for evaluator using cross entropy loss
                pred_pos = evaluator_network(new_agworld_mem[i].detach())
                target_pos = (self.env_width * pos[:, 0] + pos[:, 1]).type(torch.long)
                evaluator_batch_loss += loss_m_eval(pred_pos, target_pos)

                # Compute loss for a random input using cross entropy to obtain a base
                # for evaluator
                random_input = torch.rand_like(new_agworld_mem[i].detach())
                pred_pos_base = base_eval_network(random_input)
                target_pos_base = (self.env_width * pos[:, 0] + pos[:, 1]).type(torch.long)
                evaluator_base_batch_loss += loss_m_eval_base(pred_pos_base, target_pos_base)


            # Go back and predict action(t) given state(t) & embedding (t+1)
            # and predict state(t + 1) given state(t) and action(t)
            for i in range(recurrence_worlds - 1):

                obs = f.obs_image[inds + i].detach()
                next_obs = f.obs_image[inds + i + 1].detach()

                # take masks and convert them to 1D tensor for indexing
                # use next masks because done gives you the new game obs
                next_mask = f.mask[inds + i + 1].long().detach()
                next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                crt_actions = f.action[inds + i].long().detach()
                crt_actions_one = f.actions_onehot[inds + i].detach()

                pred_act = agworld_network.forward_action(new_agworld_mem[i], new_agworld_emb[i+1])

                pred_state = agworld_network.forward_state(new_agworld_mem[i], crt_actions_one)

                act_batch_loss += loss_m_act(pred_act, crt_actions)
                state_batch_loss += loss_m_state(pred_state, new_agworld_mem[i + 1].detach())

                # if all episodes ends at once, can't compute same/diff losses
                if next_mask.sum() == 0:
                    continue

                same = (obs[next_mask] == next_obs[next_mask]).all(1).all(1).all(1)

                s_pred_act = pred_act[next_mask]
                s_crt_act = crt_actions[next_mask]

                s_pred_state = pred_state[next_mask]
                s_crt_state = (new_agworld_mem[i + 1].detach())[next_mask]

                # if all are same/diff take care to empty tensors
                if same.sum() == same.shape[0]:
                    act_batch_loss_same += loss_m_act(s_pred_act[same], s_crt_act[same])
                    state_batch_loss_same += loss_m_state(s_pred_state[same], s_crt_state[same])

                elif same.sum() == 0:
                    act_batch_loss_diff += loss_m_act(s_pred_act[~same], s_crt_act[~same])
                    state_batch_loss_diffs += loss_m_state(s_pred_state[~same], s_crt_state[~same])

                else:
                    act_batch_loss_same += loss_m_act(s_pred_act[same], s_crt_act[same])
                    act_batch_loss_diff += loss_m_act(s_pred_act[~same], s_crt_act[~same])

                    state_batch_loss_same += loss_m_state(s_pred_state[same], s_crt_state[same])
                    state_batch_loss_diffs += loss_m_state(s_pred_state[~same], s_crt_state[~same])

            # -- Optimize models
            act_batch_loss /= (recurrence_worlds - 1)
            state_batch_loss /= (recurrence_worlds - 1)
            evaluator_batch_loss /= recurrence_worlds
            evaluator_base_batch_loss /= recurrence_worlds

            act_batch_loss_same /= (recurrence_worlds - 1)
            act_batch_loss_diff /= (recurrence_worlds - 1)

            state_batch_loss_same /= (recurrence_worlds - 1)
            state_batch_loss_diffs /= (recurrence_worlds - 1)

            ag_loss = (1 - beta) * act_batch_loss + beta * state_batch_loss

            optimizer_agworld.zero_grad()
            optimizer_evaluator.zero_grad()
            optimizer_evaluator_base.zero_grad()

            ag_loss.backward()
            evaluator_batch_loss.backward()
            evaluator_base_batch_loss.backward()

            grad_agworld_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in agworld_network.parameters()
                if p.grad is not None
            ) ** 0.5
            grad_eval_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in evaluator_network.parameters()
                if p.grad is not None
            ) ** 0.5

            torch.nn.utils.clip_grad_norm_(evaluator_network.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(agworld_network.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(base_eval_network.parameters(), max_grad_norm)

            #log some shit
            log_act_loss.append(act_batch_loss.item())
            log_state_loss.append(state_batch_loss.item())
            log_evaluator_loss.append(evaluator_batch_loss.item())
            log_base_evaluator_loss.append(evaluator_base_batch_loss.item())

            log_grad_agworld_norm.append(grad_agworld_norm)
            log_grad_eval_norm.append(grad_eval_norm)

            log_act_loss_diffs.append(act_batch_loss_diff.item())
            log_act_loss_same.append(act_batch_loss_same.item())

            log_state_loss_same.append(state_batch_loss_same.item())
            log_state_loss_diffs.append(state_batch_loss_diffs.item())

            optimizer_evaluator.step()
            optimizer_agworld.step()
            optimizer_evaluator_base.step()

        # ------------------------------------------------------------------------------------------
        # Log some values
        self.aux_logs['next_state_loss'] = np.mean(log_state_loss)
        self.aux_logs['next_action_loss'] = np.mean(log_act_loss)
        self.aux_logs['position_loss'] = np.mean(log_evaluator_loss)
        self.aux_logs['base_positon_loss'] = np.mean(log_base_evaluator_loss)
        self.aux_logs['grad_norm_icm'] = np.mean(log_grad_agworld_norm)
        self.aux_logs['grad_norm_pos'] = np.mean(log_grad_eval_norm)

        self.aux_logs['next_state_loss_same'] = np.mean(log_state_loss_same)
        self.aux_logs['next_state_loss_diffs'] = np.mean(log_state_loss_diffs)

        self.aux_logs['next_act_loss_same'] = np.mean(log_act_loss_same)
        self.aux_logs['next_act_loss_diffs'] = np.mean(log_act_loss_diffs)

        return dst_intrinsic_r

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

    def evaluate(self):
        # set networks in eval mode
        self.acmodel.eval()

        out_dir = self.eval_dir
        env = self.eval_envs
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.acmodel.recurrent
        acmodel = self.acmodel
        evaluator_network = acmodel.evaluator_network
        agworld_network = acmodel.curiosity_model
        updates_cnt = self.updates_cnt

        obs = env.reset()
        if recurrent:
            memory = self.eval_memory

        mask = self.eval_mask.fill_(1).unsqueeze(1)
        eval_ag_memory = self.eval_agworld_memory.zero_()

        prev_actions_a = torch.zeros((len(obs), env.action_space.n), device=device)
        crt_actions = torch.zeros((len(obs), env.action_space.n), device=device)
        pred_act = torch.zeros((len(obs), env.action_space.n), device=device)

        new_agworld_emb = None
        prev_agworld_mem = None
        obs_batch = None

        loss_m_nstate = torch.nn.MSELoss()

        transitions = []
        steps = 200

        for i in range(steps):

            prev_obs = obs_batch
            preprocessed_obs = preprocess_obss(obs, device=device)
            obs_batch = torch.transpose(torch.transpose(preprocessed_obs.image, 1, 3), 2, 3)

            pos_batch = preprocess_images(
                [obs[i]['position'] for i in  range(len(obs))],
                device=device,
                normalize=False
            )

            with torch.no_grad():
                if recurrent:
                    dist, value, memory = acmodel(preprocessed_obs, memory * mask)
                else:
                    dist, value = acmodel(preprocessed_obs)

                action = dist.sample()
                crt_actions.zero_()
                crt_actions.scatter_(1, action.long().unsqueeze(1), 1.)  # transform to one_hot

                # Agent world
                _, eval_ag_memory, new_agworld_emb = agworld_network(obs_batch, eval_ag_memory * mask,
                                                                     prev_actions_a)

                if prev_agworld_mem is not None:
                    pred_act = agworld_network.forward_action(prev_agworld_mem, new_agworld_emb)

                pred_state = agworld_network.forward_state(eval_ag_memory, crt_actions)

                # Evaluation network
                pred_position = F.softmax(
                    evaluator_network(eval_ag_memory.detach()),
                    dim=1)
                prev_agworld_mem = eval_ag_memory

            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            mask = (1 - torch.tensor(done, device=device, dtype=torch.float)).unsqueeze(1)

            prev_actions_a.copy_(crt_actions)

            transitions.append((obs, action.cpu(), reward, done, next_obs, dist.probs.cpu(),
                                pos_batch.cpu(), pred_state.cpu(),  pred_position.cpu(),
                                eval_ag_memory.cpu(), new_agworld_emb.cpu(), pred_act.cpu(),
                                obs_batch.cpu()))
            obs = next_obs

        if out_dir is not None:
            np.save(f"{out_dir}/eval_{updates_cnt}",
                    {"transitions": transitions,
                     "columns": ["obs", "action", "reward", "done", "next_obs", "probs",
                                 "pos_batch", "pred_emb_state", "pos_predict","eval_ag_memory",
                                 "new_agworld_emb", "pred_act", "obs_batch"]})

        self.acmodel.train()

        return None