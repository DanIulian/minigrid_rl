# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import numpy as np
import math
from argparse import Namespace

from models.utils import initialize_parameters


class OrderModelsOracle(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        self.policy_model = OrderPolicyModelOracle(cfg, obs_space, action_space,
                                                   use_memory=use_memory, use_text=use_text)

        revert_obs_space = [obs_space["image"][-1]] + list(obs_space["image"][:2])

        self.order_model = OrderingModel(cfg, revert_obs_space)

        self.memory_type = self.policy_model.memory_type
        self.use_text = self.policy_model.use_text
        self.use_memory = self.policy_model.use_memory

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OrderPolicyModelOracle(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # CFG Information
        self.memory_type = memory_type = cfg.memory_type
        hidden_size = getattr(cfg, "hidden_size", 128)
        self._memory_size = memory_size = getattr(cfg, "memory_size", 128)
        kernel_sizes = getattr(cfg, "kernel_sizes", [5, 3, 3])
        strides = getattr(cfg, "strides", [3, 2, 2])

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_sizes[0], strides[0]),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_sizes[1], strides[1]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_sizes[2], strides[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        c = obs_space["image"][2]

        self.image_embedding_size = int(np.prod(self.image_conv(torch.rand(1, c, n, m)).size()))

        self.fc1 = nn.Linear(self.image_embedding_size, hidden_size)

        crt_size = hidden_size

        # Define memory
        if self.use_memory:
            if memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(crt_size, memory_size)
            else:
                self.memory_rnn = nn.GRUCell(crt_size, memory_size)

            crt_size = memory_size

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size,
                                   batch_first=True)

        # Resize image embedding
        self.embedding_size = crt_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.fc2_val = nn.Sequential(
            nn.Linear(self.embedding_size, memory_size),
            nn.ReLU(inplace=True),
        )

        self.fc2_act = nn.Sequential(
            nn.Linear(self.embedding_size, memory_size),
            nn.ReLU(inplace=True),
        )

        # Define heads
        self.vf_int = nn.Linear(memory_size, 1)
        self.vf_ext = nn.Linear(memory_size, 1)
        self.pd = nn.Linear(memory_size, action_space.n)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.memory_type == "LSTM":
            return 2*self._memory_size
        else:
            return self._memory_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        if self.use_memory:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                hidden = memory
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden
                memory = hidden
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        val = self.fc2_val(embedding)
        act = self.fc2_act(embedding)

        vpred_int = self.vf_int(val).squeeze(1)
        vpred_ext = self.vf_ext(val).squeeze(1)

        pd = self.pd(act)

        dist = Categorical(logits=F.log_softmax(pd, dim=1))

        return dist, (vpred_ext, vpred_int), memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class OrderVisualEncoder(nn.Module):
    def __init__(self, cfg, in_size, out_size):
        super(OrderVisualEncoder, self).__init__()

        assert len(in_size) == 3, "Visual encoder len(in_size) != 3"
        assert len(out_size) == 1, "Visual encoder len(out_size) != 1"

        kernel_sizes = getattr(cfg, "kernel_sizes", [5, 3, 3])
        strides = getattr(cfg, "strides", [3, 2, 2])

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_sizes[0], strides[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_sizes[1], strides[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_sizes[2], strides[2]),
            nn.ReLU(inplace=True)
        )
        out_conv_size = self.image_conv(torch.rand([1] + in_size)).size()
        out_feat_size = int(np.prod(out_conv_size))

        self.ln1 = nn.Sequential(
            nn.Linear(out_feat_size, out_size[0]),
            nn.BatchNorm1d(out_size[0], affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_conv(x)
        x = x.flatten(1)
        x = self.ln1(x)
        return x


class OneHotPositionalEncoding:
    def __init__(self):
        pass

    def __call__(self, x):
        seq_len = x.size(0)
        y = torch.eye(seq_len).unsqueeze(1).expand(seq_len, x.size(1), seq_len).to(device=x.device)
        x = torch.cat([x, y], dim=2)
        return x


def get_shuffled_sequence(sequence: torch.Tensor):
    device = sequence.device
    seq_len = sequence.size(1)
    seq_shape = sequence.size()

    # Get indexes to shuffle
    rand = torch.rand(sequence.size()[:2])
    batch_rand_perm = rand.argsort(dim=1).to(device)

    # Generate correct order
    ordered = torch.arange(seq_len).to(device).unsqueeze(0).expand_as(batch_rand_perm)
    target_order = torch.zeros_like(ordered).to(device)
    target_order.scatter_(1, batch_rand_perm, ordered)

    batch_rand_perm = batch_rand_perm[(...,) + (None,) * (len(seq_shape) - 2)].expand_as(sequence)
    shuffled_seq = sequence.gather(1, batch_rand_perm)
    return shuffled_seq, target_order


class OrderingModel(nn.Module):
    def __init__(self, cfg: Namespace, obs_space):
        super(OrderingModel, self).__init__()

        self.order_samples_eval = cfg.order_model.eval_samples
        self.train_batches_per_epoch = cfg.order_model.train_batches_per_epoch
        self.ntoken_out = cfg.order_model.ntoken_out
        self.max_eval_bath = cfg.order_model.max_eval_bath
        self.warm_up_seq_size = cfg.order_model.warm_up_seq_size

        self.seq_len = self.ntoken_out - 1

        oracle_cfg = Namespace()
        oracle_cfg.seq_len = self.seq_len
        oracle_cfg.num_buffer_unique_obs = cfg.order_model.num_buffer_unique_obs
        oracle_cfg.num_buffer_unique_seq = cfg.order_model.num_buffer_unique_seq
        oracle_cfg.obs_shape = obs_space
        oracle_cfg.max_steps = cfg.order_model.max_steps

        self.order_oracle = OracleOrder(oracle_cfg)

        optimizer_args = vars(cfg.order_model.optimizer_args)

        self.optimizer = getattr(torch.optim, cfg.order_model.optimizer)(
            self.order_network.parameters(), **optimizer_args)

        self.tgt_mask = None
        self.warm_up = True

        self.criterion_eval = nn.CrossEntropyLoss(reduction="none")
        self.criterion_train = nn.CrossEntropyLoss()

    def get_sequences(self, obs1d: torch.Tensor, mask1d: torch.Tensor):
        # Index mask for last element in sequence

        seq_len = self.seq_len

        n = len(obs1d) - seq_len
        obs_seq = torch.stack([obs1d[i:(n+i+1)] for i in range(seq_len)]).transpose(0, 1)
        mask_seq = torch.stack([mask1d[i:(n+i+1)] for i in range(seq_len)]).transpose(0, 1)

        same_ep = mask_seq[:, 1:].sum(dim=1) == (seq_len - 1)
        sequences = obs_seq[same_ep]

        last_null = torch.zeros(seq_len-1).to(same_ep.device).bool()
        same_ep = torch.cat([last_null, same_ep])
        return sequences, same_ep

    def add_seq_to_mem(self, seq):
        # add on eval
        oracle = self.order_oracle

        for i in range(seq.size(0)):
            seq_id, elem_ids = oracle.get_seq_id(seq[i])

        return None

    def train_order(self):
        return 0

    def eval_sequences(self, seq: torch.Tensor):
        seq_cnt = seq.size(0)
        oracle = self.order_oracle
        device = seq.device

        scores = torch.zeros(seq_cnt, 1).to(device)
        for i in range(seq_cnt):
            ret_score = oracle.check_new_seq(seq[i], auto_add=False)
            if ret_score is None:
                print("Seq or elem not in buffer")
            scores[i] = ret_score.to(device)

        # Normalize so as to always prefer the final reward
        return - scores / self.max_steps


class OracleOrder:
    def __init__(self, cfg):
        seq_len = cfg.seq_len
        num_buffer_unique_obs = cfg.num_buffer_unique_obs
        num_buffer_unique_seq = cfg.num_buffer_unique_seq
        obs_shape = cfg.obs_shape

        self.max_steps = cfg.max_steps

        self.unique_obs_buffer = torch.zeros((num_buffer_unique_obs,) + obs_shape)
        self.unique_obs_buffer.fill_(-1)
        self.max_idx_unique_obs = 1

        self.unique_seq_buffer = torch.zeros(num_buffer_unique_seq, seq_len).long()
        self.unique_seq_buffer.fill_(-1)
        self.max_idx_unique_seq = 1

        self.elem_seq_matrix = torch.zeros(num_buffer_unique_obs, num_buffer_unique_seq)
        self.seq_count = torch.zeros(num_buffer_unique_seq)

    # Get idx of new obs
    def get_elem_idx(self, nobs):
        unique_obs_buffer = self.unique_obs_buffer
        max_idx_unique_obs = self.max_idx_unique_obs

        idx_e = (unique_obs_buffer[:max_idx_unique_obs] == nobs).flatten(1).all(1).nonzero()
        return idx_e.item() if idx_e.numel() > 0 else None

    def get_seq_encoding(self, seq_elems, auto_add=True):
        unique_obs_buffer = self.unique_obs_buffer

        res_idx = []
        for i in range(seq_elems.size(0)):
            idx_element = self.get_elem_idx(seq_elems[i])
            if idx_element is None and auto_add:
                unique_obs_buffer[self.max_idx_unique_obs] = seq_elems[i]
                idx_element = self.max_idx_unique_obs
                self.max_idx_unique_obs += 1
            if idx_element is None:
                return None
            res_idx.append(idx_element)

        return torch.tensor(res_idx)

    def get_seq_id(self, seq, auto_add=True):
        unique_seq_buffer = self.unique_seq_buffer
        max_idx_unique_seq = self.max_idx_unique_seq

        seq_elem_encoding = self.get_seq_encoding(seq, auto_add=auto_add)
        if seq_elem_encoding is None:
            return None, None

        idx_e = (unique_seq_buffer[:max_idx_unique_seq] == seq_elem_encoding).all(1).nonzero()
        idx_e = idx_e.item() if idx_e.numel() > 0 else None
        if idx_e is None and auto_add:
            unique_seq_buffer[self.max_idx_unique_seq] = seq_elem_encoding
            idx_e = self.max_idx_unique_seq
            self.max_idx_unique_seq += 1

        return idx_e, seq_elem_encoding

    def check_new_seq(self, seq, auto_add=True):
        seq_id, seq_elems = self.get_seq_id(seq, auto_add=auto_add)
        if seq_id is None:
            return None
        elem_seq_matrix = self.elem_seq_matrix
        seq_count = self.seq_count

        uniq_elem, uniq_counts = seq_elems.unique(return_counts=True)
        uniq_elem = uniq_elem.long()
        uniq_counts = uniq_counts.float()

        if auto_add:
            elem_seq_matrix[uniq_elem, seq_id] = uniq_counts
            seq_count[seq_id] += 1

        similar_seq = elem_seq_matrix[uniq_elem]

        possible_seq_mask = (similar_seq == uniq_counts.unsqueeze(1).expand_as(similar_seq)).all(0)
        correct_seq_cnt = seq_count[seq_id]

        same_permutations = np.prod([np.prod(np.arange(1, x + 1)) for x in uniq_counts])
        domain_cnt = seq_count[possible_seq_mask].sum() * same_permutations

        prob = correct_seq_cnt / float(domain_cnt)
        return prob
