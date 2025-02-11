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
from utils.losses_siamese import TripletLoss


def get_stats(tensor: torch.Tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    mmax = tensor.max().item()
    return mean, std, mmax


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
        eval_envs = kwargs.get("eval_envs", [])

        self.connect_worlds = getattr(cfg, "connect_worlds", True)
        assert self.connect_worlds is False, "Must be false for this config"

        self.env_world_prev_act = getattr(cfg, "env_world_prev_act", False)
        self.running_norm_obs = getattr(cfg, "running_norm_obs", False)
        self.nminibatches = getattr(cfg, "nminibatches", 4)
        self.out_dir = getattr(cfg, "out_dir", None)
        self.play_heuristic = play_heuristic = getattr(cfg, "play_heuristic", 0)
        self.pre_fill_memories = pre_fill_memories = getattr(cfg, "pre_fill_memories", 0)
        self.agworld_use_emb = agworld_use_emb = getattr(cfg, "agworld_use_emb", 0)

        self.recurrence_worlds = getattr(cfg, "recurrence_worlds", 4)
        # self.max_pred_gap = max_pred_gap = getattr(cfg, "max_pred_gap", 5)
        # self.pred_gap_factor = pred_gap_factor = getattr(cfg, "pred_gap_factor", 4)

        # Agent state exploration configs
        self.train_distance_triplets = getattr(cfg, "train_distance_triplets", 0)
        self.distance_margin = getattr(cfg, "distance_margin", 1.)

        self.warmup_steps = getattr(cfg, "warmup_steps", 5)
        self.train_ap_cross_e_gap = getattr(cfg, "train_ap_cross_e_gap", False)
        self.pred_state_rnn_mode = getattr(cfg, "pred_state_rnn_mode", True)
        self.intrinsic_norm_action = getattr(cfg, "intrinsic_norm_action", True)
        self.intrinsic_norm_gap = getattr(cfg, "intrinsic_norm_gap", True)
        self.intrinsic_gaps = getattr(cfg, "intrinsic_gaps", [1, 2, 3])
        self.save_experience_batch = getattr(cfg, "save_experience_batch", 0)
        self.action_pred_factor = getattr(cfg, "action_pred_factor", False)
        self.calc_act_pred = getattr(cfg, "calc_act_pred", False)
        self.norm_iR_rnd_style = getattr(cfg, "norm_iR_rnd_style", False)
        self.predict_state_bck = getattr(cfg, "predict_state_bck", True)
        self.use_agstate_embedding = getattr(cfg, "use_agstate_embedding", True)

        # gap_prob = torch.flip(torch.arange(1, max_pred_gap + 1), (0,)) ** pred_gap_factor
        # gap_prob = torch.ones(max_pred_gap)
        # gap_prob = gap_prob.float()/gap_prob.sum()
        # self.gap_distribution = [torch.distributions.Categorical(probs=gap_prob)]
        self.max_pred_gap = max(self.intrinsic_gaps)
        self.gap_distribution = []

        for gap_no in self.intrinsic_gaps:
            self.gap_distribution.append(lambda : torch.Tensor([gap_no]).int()[0])

        if self.train_ap_cross_e_gap:
            assert self.max_pred_gap == 1, "Cannot train cross entropy on action prediction with " \
                                           "gap > 1"
        super().__init__(
            envs, acmodel, num_frames_per_proc, discount, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, exp_used_pred)

        self.no_actions = no_actions = self.env.action_space.n

        if play_heuristic == 1:
            self.acmodel.policy_model = RandomPolicy(self.acmodel.policy_model, no_actions,
                                                     self.device)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coeff = cfg.int_coeff
        self.ext_coeff = cfg.ext_coeff

        assert self.batch_size % self.recurrence == 0

        assert self.num_frames_per_proc % self.recurrence == 0
        assert self.num_frames_per_proc % self.recurrence_worlds == 0

        # For batching from proc batch
        assert (self.batch_size / self.recurrence) % self.num_procs == 0
        assert (self.batch_size / self.recurrence_worlds) % self.num_procs == 0

        assert (self.batch_size / self.recurrence) >= self.num_procs
        assert (self.batch_size / self.recurrence_worlds) >= self.num_procs

        # -- Prepare intrinsic generators
        # self.acmodel.random_target.eval()
        if self.intrinsic_norm_action:
            self.predictor_rms_a = [RunningMeanStd() for _ in range(no_actions)]
            self.predictor_rff_a = [RewardForwardFilter(gamma=self.discount) for _ in range(
                no_actions)]

        if self.intrinsic_norm_gap:
            ng = len(self.intrinsic_gaps)
            self.predictor_rms_g = [RunningMeanStd() for _ in range(ng)]
            self.predictor_rff_g = [RewardForwardFilter(gamma=self.discount) for _ in range(ng)]

            self.predictor_rms_a = [RunningMeanStd() for _ in range(ng)]
            self.predictor_rff_a = [RewardForwardFilter(gamma=self.discount) for _ in range(ng)]

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

        self.optimizer_agstate = getattr(torch.optim, optimizer)(
            self.acmodel.agstate_network.parameters(), **optimizer_args)

        self.optimizer_evaluator = getattr(torch.optim, optimizer)(
            self.acmodel.evaluator_network.parameters(), **optimizer_args)

        if "optimizer_policy" in agent_data:
            self.optimizer_policy.load_state_dict(agent_data["optimizer_policy"])
            self.optimizer_envworld.load_state_dict(agent_data["optimizer_envworld"])
            self.optimizer_agworld.load_state_dict(agent_data["optimizer_agworld"])
            self.optimizer_agstate.load_state_dict(agent_data["optimizer_agstate"])
            self.optimizer_evaluator.load_state_dict(agent_data["optimizer_evaluator"])
            self.predictor_rms = agent_data["predictor_rms"]  # type: RunningMeanStd

        self.batch_num = 0
        self.updates_cnt = 0

        if self.running_norm_obs:
            self.collect_random_statistics(50)

        # -- Init current experience holder (processed during extrinsic reward calc)
        self.f = None
        self.prev_frame_exps = None

        # -- Previous batch of experiences last frame
        self.prev_frame_exps = None

        # -- Init evaluator envs
        self.eval_envs = None
        self.eval_memory = None
        self.eval_mask = None
        self.eval_env_memory = None
        self.eval_ag_memory = None
        self.eval_dir = None

        """ [GOALS]
        # -- Init goal buffers
        self.goal_buffer_size = self.env.max_steps // self.num_frames_per_proc + 2
        self.goal_size = self.acmodel.agstate_network.memory_size
        
        self.goals = None
        self.set_goals = None
        self.init_goals()
        """

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
        self.eval_ag_s_memory = torch.zeros(len(obs), acmodel.agstate_network.memory_size,
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
        agstate_network = self.acmodel.agstate_network
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
        frame_exp.agstate_mems = gen_memory(agstate_network.memory_size)
        frame_exp.agworld_mems = gen_memory(agworld_network.memory_size)
        frame_exp.agstate_embs = gen_memory(agworld_network.embedding_size)

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
        # Fill memories with past # TODO bad !!! bad

        # frame_exp.envworld_mems[0] = prev_frame_exps.envworld_mems
        # frame_exp.agworld_mems[0] = prev_frame_exps.agworld_mems
        # frame_exp.agstate_mems[0] = prev_frame_exps.agstate_mems
        # frame_exp.agstate_embs[0] = prev_frame_exps.agstate_embs

        return frame_exp, prev_frame_exps

    @staticmethod
    def flip_back_experience(exp):
        # for all tensors below, T x P -> P x T -> P * T
        for k, v in exp.__dict__.items():
            setattr(exp, k, v.transpose(0, 1).reshape(-1, *v.shape[2:]).contiguous())
        return exp

    def flip_continuous_f(self, exps):
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        shape = torch.Size([num_procs, num_frames_per_proc])

        # ------------------------------------------------------------------------------------------
        # Redo in format P * T -> T x P

        for k, v in exps.__dict__.items():
            if k == "obs":
                continue
            setattr(exps, k, v.view(shape + v.size()[1:]).transpose(0, 1).contiguous())

        return exps

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
        # Run worlds models & generate memories

        envworld_network = self.acmodel.envworld_network
        agworld_network = self.acmodel.agworld_network
        agstate_network = self.acmodel.agstate_network
        evaluator_network = self.acmodel.evaluator_network
        connect_worlds = self.connect_worlds
        env_world_prev_act = self.env_world_prev_act

        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        device = self.device
        env = self.env

        # ------------------------------------------------------------------------------------------
        # Get observations and full states
        f = self.f
        prev_frame_exps = self.prev_frame_exps

        # ------------------------------------------------------------------------------------------
        # -- Optimize worlds

        optimizer_evaluator = self.optimizer_evaluator
        optimizer_envworld = self.optimizer_envworld
        optimizer_agworld = self.optimizer_agworld
        optimizer_agstate = self.optimizer_agstate
        recurrence_worlds = self.recurrence_worlds
        max_grad_norm = self.max_grad_norm
        distance_margin = self.distance_margin

        # ------------------------------------------------------------------------------------------
        # _________ for all tensors below, T x P -> P x T -> P * T _______________________
        f = self.flip_back_experience(f)
        # ------------------------------------------------------------------------------------------

        loss_m_eworld = torch.nn.MSELoss()
        loss_m_eval = torch.nn.MSELoss()
        loss_m_ag_ap_cross = torch.nn.CrossEntropyLoss()
        loss_m_agstate = torch.nn.BCEWithLogitsLoss()  # torch.nn.MSELoss()
        loss_m_agstate_p = torch.nn.MSELoss()
        loss_m_triplet = TripletLoss(distance_margin)

        log_envworld_loss = []
        envworld_batch_loss_sames = []
        envworld_batch_loss_diffs = []
        log_evaluator_loss = []

        agworld_batch_loss_sames = []
        agworld_batch_loss_diffs = []

        log_agworld_loss = []
        log_agstate_ap_loss = []
        log_agstate_sp_loss = []
        log_distance_loss = []

        grad_envworld_norm = []
        grad_agworld_norm = []
        grad_agstate_norm = []
        grad_eval_norm = []

        no_actions = self.env.action_space.n
        max_action_hist = self.max_pred_gap
        train_distance_triplets = self.train_distance_triplets

        # ------------------------------------------------------------------------------------------
        # Train agent worlds with gaps - F memory already has processed memory and embeddings

        for inds, no_steps in zip(*self._get_batches_starting_indexes_gap(recurrence_worlds,
                                                                          padding=1)):

            gap_size = no_steps - recurrence_worlds

            agworld_mem = f.agworld_mems[inds + no_steps].detach()

            agstate_mem = f.agstate_mems[inds - 1].detach()
            agstate_emb = f.agstate_embs[inds - 1].detach()

            agworld_embs = [None] * no_steps
            agworld_mems = [None] * no_steps
            agstate_mems = [None] * no_steps

            agworld_batch_loss_same = torch.zeros(1, device=device)[0]
            agworld_batch_loss_diff = torch.zeros(1, device=device)[0]
            agworld_batch_loss = torch.zeros(1, device=device)[0]
            agworld_emb_batch_loss = torch.zeros(1, device=device)[0]

            mem_distance_batch_loss = torch.zeros(1, device=device)[0]

            ags_ap_batch_loss = torch.zeros(1, device=device)[0]
            ags_sp_batch_loss = torch.zeros(1, device=device)[0]

            agh_batch_loss = torch.zeros(1, device=device)[0]
            agh_eval_batch_loss = torch.zeros(1, device=device)[0]

            # --------------------------------------------------------------------------------------
            # -- Extract common embeddings for all models
            use_agstate_embedding = self.use_agstate_embedding
            crt_embeddings = None

            if use_agstate_embedding:
                crt_embeddings = [None] * no_steps

                for i in range(no_steps):
                    obs = f.obs_image[inds + i]
                    crt_embeddings[i] = agstate_network.extract_embedding(obs)

            # --------------------------------------------------------------------------------------
            # -- Backward ICM

            for i in range(gap_size, no_steps)[::-1]:  # Could start from gap_size

                obs = crt_embeddings[i] if use_agstate_embedding else f.obs_image[inds + i]
                next_mask = f.mask[inds + i + 1]  # Next state mask
                next_a_one = f.actions_onehot[inds + i + 1]

                # Forward pass Agent Net for memory
                _, agworld_mem = agworld_network(obs, agworld_mem * next_mask, next_a_one, None)
                agworld_mems[i] = agworld_mem

                # Update agent state in mem
                # f.agstate_mems[inds + i].copy_(agworld_mem.detach())

            # --------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------
            # -- Go fwd agent state and predict action given state_mem(i) &  world_mem (i+gap_size)
            # batch actions and masks
            actions_onehot = torch.stack([f.actions_onehot[inds + i] for i in range(no_steps)])
            masks = torch.stack([f.mask[inds + i] for i in range(no_steps+1)]).squeeze(2).type(
                torch.cuda.ByteTensor)
            if self.predict_state_bck:
                target_state = agworld_mems
            else:
                target_state = [f.agstate_mems[inds + i] for i in range(no_steps)]

            for i in range(no_steps - gap_size):
                obs = crt_embeddings[i] if use_agstate_embedding else f.obs_image[inds + i]
                mask = f.mask[inds + i]
                prev_a_one = f.actions_onehot[inds + i - 1]

                _, agstate_mem, _ = agstate_network(obs, agstate_mem * mask, prev_a_one, None,
                                                    embedding=use_agstate_embedding)
                agstate_mems[i] = agstate_mem

                if self.train_ap_cross_e_gap:
                    crt_a = f.action[inds + i].long()

                    next_mask = f.mask[inds + i + 1].long()
                    next_mask = next_mask.squeeze(1).type(torch.ByteTensor)

                    pred_a = agstate_network.forward_action(agstate_mem, agworld_mems[i + gap_size])

                    ags_ap_batch_loss += loss_m_ag_ap_cross(pred_a[next_mask], crt_a[next_mask])

                    # Let's try some cross entropy
                    # Check loss for same / diff obs
                    next_obs = f.obs_image[inds + i + 1]
                    obs = f.obs_image[inds + i]

                    same = (obs[next_mask] == next_obs[next_mask]).all(1).all(1).all(1)
                    s_act_predict = pred_a[next_mask]
                    s_crt_actions = crt_a[next_mask]
                    s_act_predict_s = s_act_predict[same]
                    s_act_predict_d = s_act_predict[~same]
                    if s_act_predict_d.size(0) != 0 and s_act_predict_s.size(0) != 0:
                        agworld_batch_loss_same += loss_m_ag_ap_cross(s_act_predict_s, s_crt_actions[same])
                        agworld_batch_loss_diff += loss_m_ag_ap_cross(s_act_predict_d, s_crt_actions[~same])

                    pred_state = agstate_network.predict_state(agstate_mem, f.actions_onehot[inds + i])
                    ags_sp_batch_loss += loss_m_agstate_p(pred_state[next_mask],
                                                          target_state[i + gap_size][next_mask])

                else:
                    action_hist = actions_onehot[i: i+gap_size].sum(dim=0)

                    # normalize
                    action_hist.div_(action_hist.max(dim=1)[0].unsqueeze(1))  # Normalize by max
                    # action_hist.div_(max_action_hist)
                    action_hist = action_hist.to(device)

                    # TODO --- again slow :(
                    mask = masks[i+1: i+1+gap_size].all(dim=0).type(torch.ByteTensor)
                    if mask.any():
                        pred_act_hist = agstate_network.forward_action(agstate_mem,
                                                                       agworld_mems[i + gap_size])

                        ags_ap_batch_loss += loss_m_agstate(pred_act_hist[mask], action_hist[mask])

                        # --------------------------------------------------------------------------
                        if gap_size == 1:
                            next_obs = f.obs_image[inds + i + 1]
                            obs = f.obs_image[inds + i]

                            same = (obs[mask] == next_obs[mask]).all(1).all(1).all(1)
                            s_act_predict = pred_act_hist[mask]
                            s_crt_actions = action_hist[mask]
                            s_act_predict_s = s_act_predict[same]
                            s_act_predict_d = s_act_predict[~same]
                            if s_act_predict_d.size(0) != 0 and s_act_predict_s.size(0) != 0:
                                agworld_batch_loss_same += loss_m_agstate(s_act_predict_s,
                                                                         s_crt_actions[same])
                                agworld_batch_loss_diff += loss_m_agstate(s_act_predict_d,
                                                                         s_crt_actions[~same])

                        # --------------------------------------------------------------------------
                        # agstate_batch_loss += (pred_act_hist[mask] - action_hist[mask]).abs().mean()

                        # agstate_batch_loss += loss_m_agstate(pred_act_hist, action_hist)

                        # TODO -- Should be backprop through agworld mem?
                        rnn_mode = self.pred_state_rnn_mode
                        if rnn_mode:
                            selected_actions = actions_onehot[:, mask]
                            obs = f.obs_image[inds + i]

                            empty_obs = torch.zeros_like(obs)[mask]
                            empty_hist = torch.zeros_like(action_hist)[mask]
                            new_agstate_mem = agstate_mem[mask]

                            for ii in range(gap_size):
                                _, new_agstate_mem, _ = agstate_network(empty_obs,
                                                                        new_agstate_mem,
                                                                        selected_actions[ii],
                                                                        None)

                            pred_state = agstate_network.predict_state(new_agstate_mem, empty_hist)
                            ags_sp_batch_loss += loss_m_agstate_p(pred_state,
                                                                  target_state[i + gap_size][mask])

                        else:
                            pred_state = agstate_network.predict_state(agstate_mem, action_hist)
                            ags_sp_batch_loss += loss_m_agstate_p(pred_state[mask],
                                                                  target_state[i + gap_size][mask])
                            # agstate_p_batch_loss += loss_m_agstate_p(pred_state[mask],
                            #                                          f.agstate_mems[inds + i + gap_size][
                            #                                              mask].detach())

            for _ in range(train_distance_triplets):
                triplet = torch.randperm(recurrence_worlds)[:3].sort()[0].to(device)
                mem_distance_batch_loss += loss_m_triplet(agstate_mems[triplet[0].item()],
                                                          agstate_mems[triplet[1].item()],
                                                          agstate_mems[triplet[2].item()])

            # -- Optimize models
            agworld_batch_loss_same /= recurrence_worlds
            agworld_batch_loss_diff /= recurrence_worlds

            # agworld_batch_loss /= recurrence_worlds
            ags_ap_batch_loss /= recurrence_worlds
            ags_sp_batch_loss /= recurrence_worlds

            mem_distance_batch_loss /= recurrence_worlds

            w_loss = ags_ap_batch_loss + mem_distance_batch_loss + ags_sp_batch_loss

            optimizer_agworld.zero_grad()
            optimizer_agstate.zero_grad()

            w_loss.backward()

            grad_agworld_norm.append(sum(
                p.grad.data.norm(2).item() ** 2 for p in agworld_network.parameters()
                if p.grad is not None
            ) ** 0.5)
            grad_agstate_norm.append(sum(
                p.grad.data.norm(2).item() ** 2 for p in agstate_network.parameters()
                if p.grad is not None
            ) ** 0.5)

            torch.nn.utils.clip_grad_norm_(agworld_network.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(agstate_network.parameters(), max_grad_norm)

            # log_agworld_loss.append(agworld_batch_loss.item())
            # log_agworld_emb_loss.append(agworld_emb_batch_loss.item())

            agworld_batch_loss_sames.append(agworld_batch_loss_same.item())
            agworld_batch_loss_diffs.append(agworld_batch_loss_diff.item())

            log_agstate_ap_loss.append(ags_ap_batch_loss.item())
            log_agstate_sp_loss.append(ags_sp_batch_loss.item())

            log_distance_loss.append(mem_distance_batch_loss.item())

            optimizer_agworld.step()
            optimizer_agstate.step()

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

        # logs["envworld_loss"] = np.mean(log_envworld_loss)
        # logs["agworld_loss"] = np.mean(log_agworld_loss)
        logs["agstate_ap_loss"] = np.mean(log_agstate_ap_loss)
        logs["agstate_sp_loss"] = np.mean(log_agstate_sp_loss)
        logs["agstate_distance_loss"] = np.mean(log_distance_loss)

        # logs["log_agworld_emb_loss"] = np.mean(log_agworld_emb_loss)
        # logs["evaluator_loss"] = np.mean(log_evaluator_loss)

        # Gradient log
        # logs["envworld_grad_mean"] = np.mean(grad_envworld_norm)
        # logs["envworld_grad_max"] = np.max(grad_envworld_norm)
        # logs["envworld_grad_std"] = np.std(grad_envworld_norm)

        logs["agworld_grad_mean"] = np.mean(grad_agworld_norm)
        logs["agworld_grad_max"] = np.max(grad_agworld_norm)
        logs["agworld_grad_std"] = np.std(grad_agworld_norm)

        logs["agworld_grad_mean"] = np.mean(grad_agstate_norm)
        logs["agworld_grad_max"] = np.max(grad_agstate_norm)
        logs["agworld_grad_std"] = np.std(grad_agstate_norm)

        # logs["eval_grad_mean"] = np.mean(grad_eval_norm)
        # logs["eval_grad_max"] = np.max(grad_eval_norm)
        # logs["eval_grad_std"] = np.std(grad_eval_norm)

        # Split losses
        # logs["envworld_loss_same"] = np.mean(envworld_batch_loss_sames)
        # logs["envworld_loss_diff"] = np.mean(envworld_batch_loss_diffs)
        logs["agworld_loss_same"] = np.mean(agworld_batch_loss_sames)
        logs["agworld_loss_diff"] = np.mean(agworld_batch_loss_diffs)

        # logs["envworld_loss_fwd_pass"] = envworld_loss_fwd_pass

        self.updates_cnt += 1

        return logs

    def fwd_pass_networks(self, f, prev_frame_exps):
        num_frames_per_proc = self.num_frames_per_proc
        agworld_network = self.acmodel.agworld_network
        agstate_network = self.acmodel.agstate_network
        envworld_network = self.acmodel.envworld_network
        use_agstate_embedding = self.use_agstate_embedding

        if self.pre_fill_memories:
            # --------------------------------------------------------------------------------------
            #  -- FWD Agent state
            # Get past memories from experience ->
            # TODO should get them after the previous model update!
            prev_actions = prev_frame_exps.actions_onehot
            agstate_mem = prev_frame_exps.agstate_mems

            for i in range(num_frames_per_proc):
                obs = f.obs_image[i]
                masks = f.mask[i]

                # Do one agent-environment interaction
                with torch.no_grad():
                    _, f.agstate_mems[i], f.agstate_embs[i] = \
                        agstate_network(obs, agstate_mem * masks, prev_actions, None)

                    prev_actions = f.actions_onehot[i]
                    agstate_mem = f.agstate_mems[i]
            # --------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------
            # inverse ICM
            # TODO bad that first step starts with 0 memory and 0 actions

            agworld_mem = torch.zeros_like(prev_frame_exps.agworld_mems)
            masks = torch.zeros_like(f.mask[0])
            next_actions = torch.zeros_like(f.actions_onehot[0])

            for i in range(num_frames_per_proc)[::-1]:
                obs = f.agstate_embs[i] if use_agstate_embedding else f.obs_image[i]

                # Do one agent-environment interaction
                with torch.no_grad():
                    _, f.agworld_mems[i] = agworld_network(obs, agworld_mem * masks,
                                                           next_actions, None)

                    # Going in reverse so we need next transition mask, action
                    agworld_mem = f.agworld_mems[i]
                    masks = f.mask[i]
                    next_actions = f.actions_onehot[i]

        # TODO add somewhere next memory! to use @ next step

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
            self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [
            indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes

    def _get_batches_starting_indexes_gap(self, recurrence, padding=0):
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
        gap_distribution = self.gap_distribution

        # Batch size is
        rec_batch_size = self.batch_size // recurrence

        # No envs necessary per batch - to get a batch of same no steps
        envs_factor = int(rec_batch_size // num_procs)

        batches_starting_indexes = []
        batches_step_size = []

        for sample_method in gap_distribution:
            split_no_steps = []
            no_frames = padding
            max_frames = num_frames_per_proc - padding
            while True:
                q = sample_method()
                steps = q + 1 + recurrence
                exp_steps = steps * envs_factor
                if no_frames + exp_steps > max_frames:
                    break
                no_frames += exp_steps
                split_no_steps += [steps] * envs_factor

            if (no_frames + recurrence + 1) * envs_factor <= max_frames:
                steps = (max_frames - no_frames) // envs_factor
                split_no_steps += [steps] * envs_factor

            split_no_steps = torch.stack(split_no_steps)
            split_no_steps = np.random.permutation(split_no_steps)
            frame_idx = np.hstack([np.array([padding]), split_no_steps])
            frame_idx = np.cumsum(frame_idx)[:-1]

            indexes = np.resize(frame_idx.reshape((1, -1)), (num_procs, len(frame_idx)))
            split_no_steps_i = np.resize(split_no_steps.reshape((1, -1)),
                                         (num_procs, len(split_no_steps)))

            split_no_steps_i = split_no_steps_i.reshape(-1)
            split_no_steps_i_s = split_no_steps_i.argsort()

            indexes = indexes + np.arange(0, num_procs).reshape(-1, 1) * num_frames_per_proc
            indexes = indexes.reshape(-1)

            # Should shuffle and pick envs_factor factor splits of same size
            for i in range(0, len(indexes), rec_batch_size):
                batches_starting_indexes.append(indexes[split_no_steps_i_s[i:i+rec_batch_size]])
                batches_step_size.append(split_no_steps_i[split_no_steps_i_s[i]])

        # test = [
        #     split_no_steps_i[split_no_steps_i_s[i:i+rec_batch_size]]
        # for i in range(0, len(indexes), rec_batch_size)
        # ]
        # print(test)

        return batches_starting_indexes, batches_step_size

    def get_save_data(self):
        return dict({
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_envworld": self.optimizer_envworld.state_dict(),
            "optimizer_agworld": self.optimizer_agworld.state_dict(),
            "optimizer_agstate": self.optimizer_agstate.state_dict(),
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

    def init_goals(self):
        num_frames_per_proc = self.num_frames_per_proc
        num_procs = self.num_procs
        goal_buffer_size = self.goal_buffer_size
        goal_size = self.goal_size

        self.frames_goal_size = (num_frames_per_proc, num_procs, goal_buffer_size, goal_size)
        self.goals = torch.zeros(1, num_procs, goal_buffer_size, goal_size, device=self.device)
        self.set_goals = torch.zeros(1, num_procs, goal_buffer_size, device=self.device)

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
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc

        # ------------------------------------------------------------------------------------------
        # Get augmented and processed experiences

        shape = torch.Size([num_procs, num_frames_per_proc])
        f, prev_frame_exps = self.augment_exp(exps)
        self.f = f
        self.prev_frame_exps = prev_frame_exps

        # ------------------------------------------------------------------------------------------
        # FWD pass worlds
        self.fwd_pass_networks(f, prev_frame_exps)

        # ------------------------------------------------------------------------------------------
        # Intrinsic R based on next state prediction
        no_actions = self.env.action_space.n
        max_action_hist = self.max_pred_gap / 2
        device = self.device
        agstate_network = self.acmodel.agstate_network
        # Save state
        out_dir = self.eval_dir
        updates_cnt = self.updates_cnt
        save_experience_batch = self.save_experience_batch

        save = save_experience_batch > 0 and (updates_cnt + 1) % save_experience_batch == 0

        loss_m_agstate_p = nn.MSELoss()
        loss_m_agstate = torch.nn.BCEWithLogitsLoss(reduce=False)
        agstate_mems = f.agstate_mems
        agworld_mems = f.agworld_mems

        masks = f.mask.squeeze(2).type(torch.cuda.ByteTensor)

        action_pred_factor = self.action_pred_factor
        calc_act_pred = self.calc_act_pred

        # TODO Use prev memory
        gaps = []
        dst_intrinsic_r.zero_()
        # dst_intrinsic_ra = torch.zeros_like(dst_intrinsic_r)

        if self.predict_state_bck:
            target_state = agworld_mems
        else:
            target_state = agstate_mems

        with torch.no_grad():
            dst_intrinsic_r_t = torch.zeros_like(dst_intrinsic_r)
            dst_intrinsic_r_a = torch.zeros_like(dst_intrinsic_r)

            for gap_i, gap_size_i in enumerate(self.intrinsic_gaps):
                gap_size = gap_size_i

                for i in range(0, num_frames_per_proc-gap_size_i):
                    # gap_size = gap_distribution.sample() + 1
                    # if i + gap_size >= num_frames_per_proc:
                    #     gap_size = gap_size_i

                    gaps.append(gap_size)

                    action_hist = f.actions_onehot[i: i + gap_size].sum(dim=0)

                    # normalize
                    action_hist.div_(action_hist.max(dim=1)[0].unsqueeze(1))  # Normalize by max
                    # action_hist.div_(max_action_hist)
                    action_hist = action_hist.to(device)

                    # TODO offer error in prediction between game restarts? Seems to do good :D
                    # mask2 = masks[i + 1: i + 1 + gap_size].all(dim=0)
                    if action_pred_factor or calc_act_pred:
                        pred_act_hist = agstate_network.forward_action(agstate_mems[i],
                                                                       agworld_mems[i + gap_size])
                        err_a = (pred_act_hist - action_hist).pow_(2)
                        err_a = err_a.mean(1)
                        dst_intrinsic_r_a[i + gap_size - 1].add_(err_a)
                        # dst_intrinsic_r_a[i].add_(err_a)

                    rnn_mode = self.pred_state_rnn_mode
                    if rnn_mode:
                        selected_actions = f.actions_onehot
                        empty_obs = torch.zeros_like(f.obs_image[0])
                        empty_hist = torch.zeros_like(action_hist)
                        new_agstate_mem = agstate_mems[i]

                        for ii in range(gap_size):
                            _, new_agstate_mem, _ = agstate_network(empty_obs,
                                                                    new_agstate_mem,
                                                                    selected_actions[i+ii],
                                                                    None)

                        pred_state = agstate_network.predict_state(new_agstate_mem, empty_hist)
                    else:
                        pred_state = agstate_network.predict_state(agstate_mems[i], action_hist)

                    diff_pred = (pred_state - target_state[i + gap_size]).pow_(2)
                    diff_pred = diff_pred.detach().mean(1)
                    # diff_pred.mul_(mask2.float())  # Zero prediction for fwd mask

                    # # -- Calculate prev error from t-1
                    # prev_action_hist = f.actions_onehot[i - 1] / max_action_hist
                    # action_hist.add_(prev_action_hist)
                    #
                    # mask = mask & f.mask[i].squeeze(1).type(torch.ByteTensor).to(device)
                    #
                    # pred_state = agstate_network.predict_state(agstate_mems[i - 1], action_hist)
                    # diff_pred_prev = (pred_state - agstate_mems[i + gap_size]).pow_(2)
                    # diff_pred_prev = diff_pred_prev.detach().mean(1)
                    # diff_pred_prev.mul_(mask.float())  # Zero prediction for fwd mask

                    # -- Calculate intrinsic & Normalize intrinsic rewards
                    int_rew = diff_pred  # - diff_pred_prev

                    # TODO Not sure it improves offseting reward
                    dst_intrinsic_r_t[i + gap_size - 1].add_(int_rew)
                    # dst_intrinsic_r_t[i].add_(int_rew)
                    # TODO - What about last prediction :( no intrinsic reward

                if self.intrinsic_norm_gap:
                    self.predictor_rms_g[gap_i].update(dst_intrinsic_r_t.view(-1))
                    rms = torch.sqrt(self.predictor_rms_g[gap_i].var).to(dst_intrinsic_r.device)
                    dst_intrinsic_r_t.div_(rms)

                    if save:
                        mean, std, mmax = get_stats(dst_intrinsic_r_t)
                        print(f"Gap {gap_size} SP rms: {rms.item():.6f} | "
                              f"ysM: {mean:.6f} {std:.6f} {mmax:.6f}")
                elif save:
                    if action_pred_factor:
                        mean, std, mmax = get_stats(dst_intrinsic_r_a)
                        r_a = f" | AP: ysM: {mean:.3f} {std:.3f} {mmax:.3f}"
                    else:
                        r_a = ""

                    mean, std, mmax = get_stats(dst_intrinsic_r_t)
                    print(f"Gap {gap_size} | SP ysM: {mean:.6f} {std:.6f} {mmax:.6f}" + r_a)

                if gap_size == 1 and save:
                    for i in range(no_actions):
                        actions = (f.action == i).type(torch.cuda.ByteTensor)

                        mean, std, mmax = get_stats(dst_intrinsic_r_t[actions])
                        r_t = f" | SP: ysM: {mean:.3f} {std:.3f} {mmax:.3f}"

                        mean, std, mmax = get_stats(dst_intrinsic_r_a[actions])
                        r_a = f" | AP: ysM: {mean:.3f} {std:.3f} {mmax:.3f}"
                        print(f"Gap {gap_size} [{i}] no: {actions.sum().item():4}" + r_t + r_a)

                if action_pred_factor:

                    # self.predictor_rms_a[gap_i].update(dst_intrinsic_r_a.view(-1))
                    # rms = torch.sqrt(self.predictor_rms_a[gap_i].var).to(dst_intrinsic_r_a.device)
                    # print(f"Gap {gap_size} act pred rms:", rms.item())
                    # dst_intrinsic_r_a.div_(rms)

                    # mean, std, mmax = get_stats(dst_intrinsic_r_a)
                    # print(f"Gap {gap_size} AP ysM: {mean:.6f} {std:.6f} {mmax:.6f}")

                    dst_intrinsic_r_a = dst_intrinsic_r_a.max() * 1.2 - dst_intrinsic_r_a

                    # dst_intrinsic_r_t.sub_(dst_intrinsic_r_a)
                    # dst_intrinsic_r_t.mul_(dst_intrinsic_r_a)

                    # mean, std, mmax = get_stats(dst_intrinsic_r_t)
                    # print(f"Gap {gap_size} SP ysM: {mean:.6f} {std:.6f} {mmax:.6f}")

                dst_intrinsic_r.add_(dst_intrinsic_r_t)

        # Append info for eval
        if save:
            gaps.append(0)  # adjust to have same size
            f.gaps = gaps
            f.dst_intrinsic_r_pre = dst_intrinsic_r.clone()

        if self.intrinsic_norm_action:
            no_actions = self.no_actions
            int_rff = torch.zeros((num_frames_per_proc, num_procs), device=self.device)
            for i in range(no_actions):
                int_rff.zero_()

                actions = f.action == i

                self.predictor_rms_a[i].update(dst_intrinsic_r[actions].view(-1))
                # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
                rms = torch.sqrt(self.predictor_rms_a[i].var).to(dst_intrinsic_r.device)

                actions = actions.float()

                if save:
                    print(f"Action {i} rms:", rms.item())

                rms = rms * actions + (1-actions)
                dst_intrinsic_r.div_(rms)

        # ------------------------------------------------------------------------------------------
        # Normalize intrinsic reward
        if self.norm_iR_rnd_style:
            self.predictor_rff.reset()  # do you have to rest it every time ???

            int_rff = torch.zeros((self.num_frames_per_proc, self.num_procs), device=self.device)

            for i in reversed(range(self.num_frames_per_proc)):
                int_rff[i] = self.predictor_rff.update(dst_intrinsic_r[i])

            self.predictor_rms.update(int_rff.view(-1))
            # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
            rms = torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device)
            dst_intrinsic_r.div_(rms)
        else:
            self.predictor_rms.update(dst_intrinsic_r.view(-1))
            # self.predictor_rms.update(dst_intrinsic_r.max(1)[0].view(-1))
            # dst_intrinsic_r.sub_(self.predictor_rms.mean.to(dst_intrinsic_r.device))
            rms = torch.sqrt(self.predictor_rms.var).to(dst_intrinsic_r.device)
            dst_intrinsic_r.div_(rms)

        # ------------------------------------------------------------------------------------------

        if save:
            mean, std, mmax = get_stats(dst_intrinsic_r)
            print(f"IR ysM: {mean:.6f} {std:.6f} {mmax:.6f}")

        # ------------------------------------------------------------------------------------------

        if save:
            f.dst_intrinsic_r_post = dst_intrinsic_r.clone()
            torch.save(f, f"{out_dir}/f_{updates_cnt}")

            delattr(f, "gaps")
            delattr(f, "dst_intrinsic_r_pre")
            delattr(f, "dst_intrinsic_r_post")

        return dst_intrinsic_r

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
        agstate_network = self.acmodel.agstate_network

        connect_worlds = self.connect_worlds
        env_world_prev_act = self.env_world_prev_act

        updates_cnt = self.updates_cnt

        obs = env.reset()
        if recurrent:
            memory = self.eval_memory

        mask = self.eval_mask.fill_(1).unsqueeze(1)
        envworld_mem = self.eval_env_memory.zero_()
        eval_ag_memory = self.eval_ag_memory.zero_()

        prev_actions_e = torch.zeros((len(obs), env.action_space.n), device=device)
        prev_actions_a = torch.zeros((len(obs), env.action_space.n), device=device)
        crt_actions = torch.zeros((len(obs), env.action_space.n), device=device)
        pred_act = torch.zeros((len(obs), env.action_space.n), device=device)

        new_agworld_emb = None
        prev_agworld_emb = None
        obs_predict = None
        obs_batch = None
        loss_m_eworld = torch.nn.MSELoss()

        transitions = []
        envworld_batch_loss = 0
        steps = 400

        for i in range(steps):

            prev_obs = obs_batch
            preprocessed_obs = preprocess_obss(obs, device=device)
            obs_batch = torch.transpose(torch.transpose(preprocessed_obs.image, 1, 3), 2, 3)

            if obs_predict is not None:
                m = mask.squeeze(1).type(torch.ByteTensor)
                diff_obs = obs_batch[m] - prev_obs[m]
                envworld_batch_loss += loss_m_eworld(obs_predict[m], diff_obs)

            with torch.no_grad():
                # Policy

                if recurrent:
                    dist, value, memory = acmodel(preprocessed_obs, memory * mask)
                else:
                    dist, value = acmodel(preprocessed_obs)

                action = dist.sample()
                crt_actions.zero_()
                crt_actions.scatter_(1, action.long().unsqueeze(1), 1.)

                # -- Env world
                action_embedding = eval_ag_memory if connect_worlds else crt_actions

                obs_predict, envworld_mem = envworld_network(obs_batch, envworld_mem * mask,
                                                             prev_actions_e, action_embedding)

                # -- Ag state run
                _, eval_ag_memory, _ = agstate_network(obs_batch, eval_ag_memory * mask,
                                                       prev_actions_a, None)

                # Evaluation network
                pred_full_state = evaluator_network(envworld_mem.detach())

            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            mask = (1 - torch.tensor(done, device=device, dtype=torch.float)).unsqueeze(1)

            if env_world_prev_act:
                prev_actions_e.copy_(crt_actions)

            prev_actions_a.copy_(crt_actions)

            transitions.append([obs, action.cpu(), reward, done, next_obs, dist.probs.cpu(),
                                pred_full_state.cpu(), obs_predict.cpu(), envworld_mem.cpu(),
                                eval_ag_memory, None, None,
                                obs_batch.cpu(), mask.cpu(), crt_actions.cpu()])
            obs = next_obs

        columns = ["obs", "action", "reward", "done", "next_obs", "probs",
                   "pred_full_state", "obs_predict", "envworld_mem",
                   "eval_ag_state_mem", "ag_world_mem", "pred_act",
                   "obs_batch", "mask", "crt_actions"]

        obs_idx = columns.index("obs_batch")
        mask_idx = columns.index("mask")
        act_idx = columns.index("crt_actions")

        eval_ag_memory_idx = columns.index("eval_ag_state_mem")
        new_agworld_mem_idx = columns.index("ag_world_mem")
        pred_act_idx = columns.index("pred_act")

        # Agent world
        next_actions = torch.zeros((len(obs), env.action_space.n), device=device)
        ag_w_mem = self.eval_ag_memory.zero_()
        mask = self.eval_mask.fill_(1).unsqueeze(1)
        gap_size = 1

        for i in range(gap_size, steps)[::-1]:
            obs_batch = transitions[i][obs_idx].to(device)
            prev_agstate_mem = transitions[i - gap_size][eval_ag_memory_idx].to(device)

            with torch.no_grad():

                _, ag_w_mem = agworld_network(obs_batch, ag_w_mem * mask, next_actions, None)
                mask = transitions[i][mask_idx].to(device)
                next_actions = transitions[i][act_idx].to(device)

                pred_act = agstate_network.forward_action(prev_agstate_mem, ag_w_mem)

            transitions[i][new_agworld_mem_idx] = ag_w_mem.cpu()
            transitions[i-gap_size][pred_act_idx] = pred_act.cpu()

        print(f"[evalop] {(envworld_batch_loss/steps).item():.6f}")

        if out_dir is not None:
            np.save(f"{out_dir}/eval_{updates_cnt}",
                    {"transitions": transitions,
                     "columns": columns})

        return None





