"""
    inspired from https://github.com/lcswillems/torch-rl
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_rl
from argparse import Namespace
from copy import deepcopy

from agents.two_v_base_general import TwoValueHeadsBaseGeneral
from torch_rl.utils import DictList
from utils.utils import RunningMeanStd, RewardForwardFilter
from utils.format import preprocess_images
from torch.distributions.categorical import Categorical


class HLoss(torch.nn.Module):
    # entropy loss function
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class RandomPolicy(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, crt_policy, no_actions, device):
        super().__init__()
        self.no_actions = no_actions
        self.device = device
        self._memory_size = crt_policy.memory_size

        self.aux = nn.Linear(10, 10)

    @property
    def memory_size(self):
        return self._memory_size

    def forward(self, obs, memory):
        no_actions = self.no_actions
        device = self.device
        batch = obs.image.size(0)

        dist = Categorical(probs=torch.rand(batch, no_actions).to(device))
        vpred_ext = torch.rand(batch).to(device)
        vpred_int = torch.rand(batch).to(device)

        return dist, (vpred_ext, vpred_int), memory


class PPOWorlds(TwoValueHeadsBaseGeneral):
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

        self.connect_worlds = getattr(cfg, "connect_worlds", True)
        self.env_world_prev_act = getattr(cfg, "env_world_prev_act", False)
        self.recurrence_worlds = getattr(cfg, "recurrence_worlds", 16)
        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)
        self.nminibatches = getattr(cfg, "nminibatches", 4)
        self.out_dir = getattr(cfg, "out_dir", None)
        self.play_heuristic = play_heuristic = getattr(cfg, "play_heuristic", 0)
        self.pre_fill_memories = pre_fill_memories = getattr(cfg, "pre_fill_memories", 0)
        self.agworld_use_emb = agworld_use_emb = getattr(cfg, "agworld_use_emb", 0)

        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, exp_used_pred)

        if play_heuristic == 1:
            no_actions = self.env.action_space.n
            self.acmodel.policy_model = RandomPolicy(self.acmodel.policy_model, no_actions,
                                                     self.device)

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

        self.optimizer_envworld = getattr(torch.optim, optimizer)(
            self.acmodel.envworld_network.parameters(), **optimizer_args)

        self.optimizer_agworld = getattr(torch.optim, optimizer)(
            self.acmodel.agworld_network.parameters(), **optimizer_args)

        self.optimizer_evaluator = getattr(torch.optim, optimizer)(
            self.acmodel.evaluator_network.parameters(), **optimizer_args)

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_envworld.load_state_dict(agent_data["optimizer_envworld"])
            self.optimizer_agworld.load_state_dict(agent_data["optimizer_agworld"])
            self.optimizer_evaluator.load_state_dict(agent_data["optimizer_evaluator"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

        self.batch_num = 0
        self.updates_cnt = 0

        if self.running_norm_obs:
            self.collect_random_statistics(50)

        # -- Previous batch of experiences last frame
        self.prev_frame_exps = None

        # -- Init evaluator envs
        self.eval_envs = None
        self.eval_memory = None
        self.eval_mask = None
        self.eval_env_memory = None
        self.eval_ag_memory = None
        self.eval_dir = None

        if len(eval_envs) > 0:
            self.eval_envs = self.init_evaluator(eval_envs)
            self.eval_dir = os.path.join(self.out_dir, "eval")
            if not os.path.isdir(self.eval_dir):
                os.mkdir(self.eval_dir)

    def init_evaluator(self, envs):
        from torch_rl.utils import ParallelEnv
        device = self.device
        acmodel = self.acmodel

        eval_envs = ParallelEnv(envs)
        obs = eval_envs.reset()

        if self.acmodel.recurrent:
            self.eval_memory = torch.zeros(len(obs), acmodel.memory_size, device=device)

        self.eval_env_memory = torch.zeros(len(obs), acmodel.envworld_network.memory_size,
                                           device=device)
        self.eval_ag_memory = torch.zeros(len(obs), acmodel.agworld_network.memory_size,
                                          device=device)
        self.eval_mask = torch.ones(len(obs), device=device)

        return eval_envs

    def augment_exp(self, exps):
        # from exp (P * T , ** ) -> (T, P, **)
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        env = self.env
        envworld_network = self.acmodel.envworld_network
        agworld_network = self.acmodel.agworld_network
        evaluator_network = self.acmodel.evaluator_network

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
        frame_exp.states = inverse_img(frame_exp.states, 2)

        def gen_memory(ss):
            return torch.zeros(num_frames_per_proc, num_procs, ss, device=device)

        frame_exp.envworld_mems = gen_memory(envworld_network.memory_size)
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

        frame_exp.envworld_mems[0] = prev_frame_exps.envworld_mems
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
                    # norm_obs = sb.obs_image
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
                grad_norm = 0
                self.optimizer_policy.zero_grad()
                # batch_loss.backward()
                # grad_norm = sum(
                #     p.grad.data.norm(2).item() ** 2 for p in self.acmodel.policy_model.parameters()
                #     if p.grad is not None
                # ) ** 0.5
                # torch.nn.utils.clip_grad_norm_(self.acmodel.policy_model.parameters(),
                #                                self.max_grad_norm)
                # self.optimizer_policy.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values_ext.append(batch_ext_value)
                log_values_int.append(batch_int_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_ext_losses.append(batch_value_ext_loss)
                log_value_int_losses.append(batch_value_int_loss)
                log_grad_norms.append(grad_norm)

        # ------------------------------------------------------------------------------------------
        # Run worlds models & generate memories

        envworld_network = self.acmodel.envworld_network
        agworld_network = self.acmodel.agworld_network
        evaluator_network = self.acmodel.evaluator_network
        connect_worlds = self.connect_worlds
        env_world_prev_act = self.env_world_prev_act

        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        env = self.env

        # ------------------------------------------------------------------------------------------
        # Get observations and full states
        shape = torch.Size([num_procs, num_frames_per_proc])
        f, prev_frame_exps = self.augment_exp(exps)
        # ------------------------------------------------------------------------------------------
        envworld_loss_fwd_pass = -1
        if self.pre_fill_memories:
            prev_actions = prev_frame_exps.actions_onehot
            for i in range(num_frames_per_proc - 1):
                obs = f.obs_image[i]
                masks = f.mask[i]

                # Do one agent-environment interaction
                with torch.no_grad():
                    _, f.agworld_mems[i + 1], f.agworld_embs[i] = \
                        agworld_network(obs, f.agworld_mems[i] * masks, prev_actions,
                                        f.actions_onehot[i])
                    prev_actions = f.actions_onehot[i]

            envworld_batch_loss = 0
            loss_m_eworld = torch.nn.MSELoss()
            prev_actions = prev_frame_exps.actions_onehot

            for i in range(num_frames_per_proc - 1 - int(connect_worlds)):
                obs = f.obs_image[i]
                masks = f.mask[i]
                next_mask = f.mask[i+1].long().detach()

                # Do one agent-environment interaction
                # communicate agent memmory
                with torch.no_grad():
                    action_emb = f.agworld_mems[i + 2].detach() if connect_worlds \
                        else f.actions_onehot[i]

                    obs_predict, f.envworld_mems[i + 1] = \
                        envworld_network(obs, f.envworld_mems[i] * masks, prev_actions,
                                         action_emb)

                    # predict differece in obs
                    next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                    diff_obs = f.obs_image[i+1][next_mask] - obs[next_mask]
                    envworld_batch_loss += loss_m_eworld(obs_predict[next_mask], diff_obs)

                if env_world_prev_act:
                    prev_actions = f.actions_onehot[i].detach()

            envworld_loss_fwd_pass = (envworld_batch_loss /
                                      (num_frames_per_proc - 1 - int(connect_worlds))).item()

        # TODO add somewhere next memory! to use @ next step

        # ------------------------------------------------------------------------------------------
        # -- Optimize worlds

        optimizer_evaluator = self.optimizer_evaluator
        optimizer_envworld = self.optimizer_envworld
        optimizer_agworld = self.optimizer_agworld
        recurrence_worlds = self.recurrence_worlds
        max_grad_norm = self.max_grad_norm

        # ------------------------------------------------------------------------------------------
        # _________ for all tensors below, T x P -> P x T -> P * T _______________________
        f = self.flip_back_experience(f)
        # ------------------------------------------------------------------------------------------

        loss_m_eworld = torch.nn.MSELoss()
        loss_m_aworld = torch.nn.CrossEntropyLoss()
        loss_m_aworld_emb = torch.nn.MSELoss()
        loss_m_ah = HLoss()
        loss_m_ah_eval = torch.nn.CrossEntropyLoss()
        loss_m_eval = torch.nn.MSELoss()

        log_envworld_loss = []
        envworld_batch_loss_sames = []
        envworld_batch_loss_diffs = []
        agworld_batch_loss_sames = []
        agworld_batch_loss_diffs = []
        log_agworld_loss = []
        log_agworld_emb_loss = []
        log_agh_loss = []
        log_agh_eval_loss = []
        log_evaluator_loss = []

        btch = -1
        grad_envworld_norm = []
        grad_agworld_norm = []
        grad_eval_norm = []

        for inds in self._get_batches_starting_indexes(recurrence=recurrence_worlds, padding=1):
            btch += 1

            envworld_mem = f.envworld_mems[inds].detach()
            agworld_mem = f.agworld_mems[inds].detach()
            agworld_emb = f.agworld_embs[inds].detach()
            new_agworld_emb = [None] * recurrence_worlds
            new_agworld_mem = [None] * recurrence_worlds

            envworld_batch_loss = 0
            agworld_batch_loss_same = 0
            agworld_batch_loss_diff = 0
            envworld_batch_loss_same = 0
            envworld_batch_loss_diff = 0
            agworld_batch_loss = 0
            agworld_emb_batch_loss = 0
            agh_batch_loss = torch.zeros(1)[0]
            agh_eval_batch_loss = torch.zeros(1)[0]
            evaluator_batch_loss = 0

            # TODO not all recurrence is done ?! Must need next state + obs
            # -- Agent world
            for i in range(recurrence_worlds):
                obs = f.obs_image[inds + i].detach()
                mask = f.mask[inds + i]
                prev_actions_one = f.actions_onehot[inds + i - 1].detach()
                crt_actions = f.actions_onehot[inds + i].detach()

                # Forward pass Agent Net for memory
                _, agworld_mem, new_agworld_emb[i] = \
                    agworld_network(obs, agworld_mem * mask, prev_actions_one, crt_actions)
                new_agworld_mem[i] = agworld_mem

            # Go back and predict action given embeddint(t) & embedding (t+1)
            for i in range(recurrence_worlds-1):
                obs = f.obs_image[inds + i].detach()
                next_obs = f.obs_image[inds + i + 1].detach()

                next_mask = f.mask[inds + i + 1].long().detach()
                next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                crt_actions_one_hot = f.actions_onehot[inds + i].detach()

                crt_actions = f.action[inds + i].long().detach()

                # Predict agent action
                pred_act = agworld_network.forward_action(new_agworld_emb[i], new_agworld_emb[i+1])
                agworld_batch_loss += loss_m_aworld(pred_act[next_mask],
                                                    crt_actions[next_mask].detach())

                same = (obs[next_mask] == next_obs[next_mask]).all(1).all(1).all(1)
                s_act_predict = pred_act[next_mask]
                s_crt_actions = crt_actions[next_mask]

                agworld_batch_loss_same += loss_m_aworld(s_act_predict[same], s_crt_actions[same])
                agworld_batch_loss_diff += loss_m_aworld(s_act_predict[~same], s_crt_actions[~same])

                # Predict agent embedding
                pred_ag_emb = agworld_network.forward_action_eval(new_agworld_mem[i],
                                                                  crt_actions_one_hot)

                agworld_emb_batch_loss += loss_m_aworld_emb(pred_ag_emb[next_mask],
                                                            new_agworld_emb[i + 1].detach()[next_mask])

                # pred_act = agworld_network.forward_action(new_agworld_emb[i].detach(),
                #                                           new_agworld_emb[i+1])
                # # agh_batch_loss += loss_m_aworld(pred_act, crt_actions.detach())
                # agh_batch_loss += loss_m_ah(pred_act)
                #
                # pred_act = agworld_network.forward_action_eval(new_agworld_emb[i+1].detach(),
                #                                                new_agworld_emb[i+1].detach())
                # agh_eval_batch_loss += loss_m_ah_eval(pred_act, crt_actions.detach())

            # -- ENV world
            prev_actions_one = prev_frame_exps.actions_onehot

            for i in range(recurrence_worlds - int(connect_worlds)):
                mask = f.mask[inds + i]
                obs = f.obs_image[inds + i].detach()
                crt_full_state = f.states[inds + i].detach()
                next_obs = f.obs_image[inds + i + 1].detach()
                next_mask = f.mask[inds + i + 1].long().detach()

                if env_world_prev_act:
                    prev_actions_one = f.actions_onehot[inds + i - 1].detach()
                crt_actions = f.actions_onehot[inds + i].detach()

                # communicate agent memmory
                action_emb = new_agworld_mem[i + 1].detach() if connect_worlds else crt_actions

                # Compute loss for env world
                obs_predict, envworld_mem = \
                    envworld_network(obs, envworld_mem * mask, prev_actions_one,
                                     action_emb)

                # select only states with future state
                next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                diff_obs = (next_obs[next_mask] - obs[next_mask]).detach()
                s_obs_predict = obs_predict[next_mask]
                envworld_batch_loss += loss_m_eworld(s_obs_predict, diff_obs)

                same = (obs[next_mask] == next_obs[next_mask]).all(1).all(1).all(1)

                envworld_batch_loss_same += loss_m_eworld(s_obs_predict[same], diff_obs[same])
                envworld_batch_loss_diff += loss_m_eworld(s_obs_predict[~same], diff_obs[~same])

                # TODO Update memories for next epoch
                # # Update memories for next epoch
                # if self.acmodel.recurrent and i < self.recurrence - 1:
                #     exps.memory[inds + i + 1] = memory.detach()

                # Compute loss for evaluator
                pred_full_state = evaluator_network(envworld_mem.detach())
                evaluator_batch_loss += loss_m_eval(pred_full_state, crt_full_state)

            # -- Optimize models
            envworld_batch_loss /= recurrence_worlds
            envworld_batch_loss_same /= recurrence_worlds
            envworld_batch_loss_diff /= recurrence_worlds
            agworld_batch_loss_same /= recurrence_worlds
            agworld_batch_loss_diff /= recurrence_worlds
            agworld_batch_loss /= recurrence_worlds
            agworld_emb_batch_loss /= recurrence_worlds
            # agh_batch_loss /= recurrence_worlds
            # agh_eval_batch_loss /= recurrence_worlds
            evaluator_batch_loss /= recurrence_worlds

            # ag_loss = (agworld_batch_loss - agh_batch_loss)
            # w_loss = agworld_batch_loss - agh_batch_loss * 1.0 \
            #          + envworld_batch_loss * 10 + agh_eval_batch_loss
            w_loss = agworld_batch_loss + envworld_batch_loss * 1 + agworld_emb_batch_loss

            optimizer_agworld.zero_grad()
            optimizer_envworld.zero_grad()
            optimizer_evaluator.zero_grad()

            # ag_loss.backward(retain_graph=connect_worlds)
            # envworld_batch_loss.backward()
            w_loss.backward()
            evaluator_batch_loss.backward()

            grad_envworld_norm.append(sum(
                p.grad.data.norm(2).item() ** 2 for p in envworld_network.parameters()
                if p.grad is not None
            ) ** 0.5)
            grad_agworld_norm.append(sum(
                p.grad.data.norm(2).item() ** 2 for p in agworld_network.parameters()
                if p.grad is not None
            ) ** 0.5)
            grad_eval_norm.append(sum(
                p.grad.data.norm(2).item() ** 2 for p in evaluator_network.parameters()
                if p.grad is not None
            ) ** 0.5)

            torch.nn.utils.clip_grad_norm_(envworld_network.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(evaluator_network.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(agworld_network.parameters(), max_grad_norm)

            log_agworld_loss.append(agworld_batch_loss.item())
            log_agworld_emb_loss.append(agworld_emb_batch_loss.item())
            log_agh_loss.append(agh_batch_loss.item())
            log_agh_eval_loss.append(agh_eval_batch_loss.item())
            log_envworld_loss.append(envworld_batch_loss.item())
            envworld_batch_loss_sames.append(envworld_batch_loss_same.item())
            envworld_batch_loss_diffs.append(envworld_batch_loss_diff.item())
            agworld_batch_loss_sames.append(agworld_batch_loss_same.item())
            agworld_batch_loss_diffs.append(agworld_batch_loss_diff.item())
            log_evaluator_loss.append(evaluator_batch_loss.item())

            optimizer_envworld.step()
            optimizer_evaluator.step()
            optimizer_agworld.step()

        # ------------------------------------------------------------------------------------------
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

        logs["envworld_loss"] = np.mean(log_envworld_loss)

        logs["agworld_loss"] = np.mean(log_agworld_loss)
        logs["log_agworld_emb_loss"] = np.mean(log_agworld_emb_loss)
        logs["agh_loss"] = np.mean(log_agh_loss)
        logs["agh_eval_loss"] = np.mean(log_agh_eval_loss)
        logs["evaluator_loss"] = np.mean(log_evaluator_loss)

        # Gradient log
        logs["envworld_grad_mean"] = np.mean(grad_envworld_norm)
        logs["envworld_grad_max"] = np.max(grad_envworld_norm)
        logs["envworld_grad_std"] = np.std(grad_envworld_norm)
        logs["agworld_grad_mean"] = np.mean(grad_agworld_norm)
        logs["agworld_grad_max"] = np.max(grad_agworld_norm)
        logs["agworld_grad_std"] = np.std(grad_agworld_norm)
        logs["eval_grad_mean"] = np.mean(grad_eval_norm)
        logs["eval_grad_max"] = np.max(grad_eval_norm)
        logs["eval_grad_std"] = np.std(grad_eval_norm)

        # Split losses
        logs["envworld_loss_same"] = np.mean(envworld_batch_loss_sames)
        logs["envworld_loss_diff"] = np.mean(envworld_batch_loss_diffs)
        logs["agworld_loss_same"] = np.mean(agworld_batch_loss_sames)
        logs["agworld_loss_diff"] = np.mean(agworld_batch_loss_diffs)

        logs["envworld_loss_fwd_pass"] = envworld_loss_fwd_pass

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
            "optimizer_envworld": self.optimizer_envworld.state_dict(),
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
            obs = exps.obs_image * 15.0  # horrible harcoded normalized factor
            # normalize the observations for predictor and target networks
            norm_obs = torch.clamp(
                torch.div(
                    (obs - self.obs_rms.mean.to(exps.obs_image.device)),
                    torch.sqrt(self.obs_rms.var).to(exps.obs_image.device)),
                -5.0, 5.0)

            self.obs_rms.update(obs.cpu())  # update running mean
        else:
            # Without norm
            norm_obs = exps.obs_image

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
        exps.obs_image = exps.obs.image

    def evaluate(self):
        out_dir = self.eval_dir
        env = self.eval_envs
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.acmodel.recurrent
        acmodel = self.acmodel
        evaluator_network = self.acmodel.evaluator_network
        envworld_network = self.acmodel.envworld_network
        agworld_network = self.acmodel.agworld_network
        connect_worlds = self.connect_worlds
        env_world_prev_act = self.env_world_prev_act

        updates_cnt = self.updates_cnt

        obs = env.reset()
        if recurrent:
            memory = self.eval_memory

        mask = self.eval_mask.fill_(1).unsqueeze(1)
        envworld_mem = self.eval_env_memory.zero_()
        eval_ag_memory = self.eval_ag_memory.zero_()
        agworld_use_emb = self.agworld_use_emb

        prev_actions_e = torch.zeros((len(obs), env.action_space.n), device=device)
        prev_actions_a = torch.zeros((len(obs), env.action_space.n), device=device)
        crt_actions = torch.zeros((len(obs), env.action_space.n), device=device)
        pred_act = torch.zeros((len(obs), env.action_space.n), device=device)

        new_agworld_emb = None
        prev_agworld_mem = None
        obs_predict = None
        obs_batch = None
        loss_m_eworld = torch.nn.MSELoss()

        transitions = []
        envworld_batch_loss = 0
        steps = 200

        for i in range(steps):

            prev_obs = obs_batch
            preprocessed_obs = preprocess_obss(obs, device=device)
            obs_batch = torch.transpose(torch.transpose(preprocessed_obs.image, 1, 3), 2, 3)

            if obs_predict is not None:
                m = mask.squeeze(1).type(torch.ByteTensor)
                diff_obs = obs_batch[m] - prev_obs[m]
                envworld_batch_loss += loss_m_eworld(obs_predict[m], diff_obs)

            with torch.no_grad():
            # if True:
                # Policy

                if recurrent:
                    dist, value, memory = acmodel(preprocessed_obs, memory * mask)
                else:
                    dist, value = acmodel(preprocessed_obs)

                action = dist.sample()
                crt_actions.zero_()
                crt_actions.scatter_(1, action.long().unsqueeze(1), 1.)

                # Env world
                action_embedding = eval_ag_memory if connect_worlds else crt_actions

                obs_predict, envworld_mem = envworld_network(obs_batch, envworld_mem * mask,
                                                             prev_actions_e, action_embedding)

                # Evaluation network
                pred_full_state = evaluator_network(envworld_mem.detach())

                # Agent world
                _, eval_ag_memory, new_agworld_emb = agworld_network(obs_batch,
                                                                     eval_ag_memory * mask,
                                                                     prev_actions_a, crt_actions)
                if prev_agworld_mem is not None:
                    pred_act = agworld_network.forward_action(prev_agworld_mem, new_agworld_emb)

                pred_ag_emb = agworld_network.forward_action_eval(eval_ag_memory, crt_actions)

                prev_agworld_mem = new_agworld_emb if agworld_use_emb else eval_ag_memory

            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            mask = (1 - torch.tensor(done, device=device, dtype=torch.float)).unsqueeze(1)

            if env_world_prev_act:
                prev_actions_e.copy_(crt_actions)

            prev_actions_a.copy_(crt_actions)

            transitions.append((obs, action.cpu(), reward, done, next_obs, dist.probs.cpu(),
                                pred_full_state.cpu(), obs_predict.cpu(), envworld_mem.cpu(),
                                eval_ag_memory.cpu(), new_agworld_emb.cpu(), pred_act.cpu(),
                                obs_batch.cpu(), pred_ag_emb.cpu(), action_embedding.cpu()))


            obs = next_obs

        print(f"[evalop] {(envworld_batch_loss/steps).item():.6f}")

        if out_dir is not None:
            np.save(f"{out_dir}/eval_{updates_cnt}",
                    {"transitions": transitions,
                     "columns": ["obs", "action", "reward", "done", "next_obs", "probs",
                                 "pred_full_state", "obs_predict", "envworld_mem",
                                 "eval_ag_memory", "new_agworld_emb", "pred_act",
                                 "obs_batch", "pred_ag_emb", "action_embedding"]})

        return None





