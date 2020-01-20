# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import numpy as np
from argparse import Namespace
import math
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


from models.utils import initialize_parameters


class OrderModels(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        self.policy_model = OrderPolicyModel(cfg, obs_space, action_space, use_memory=use_memory,
                                             use_text=use_text)

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


class OrderPolicyModel(nn.Module, torch_rl.RecurrentACModel):
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

        self.ln1 = nn.Linear(int(np.prod(out_conv_size)), out_size[0])

    def forward(self, x):
        x = self.image_conv(x)
        x = x.flatten(1)
        x = self.ln1(x)
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

        self.seq_len = self.ntoken_out - 1

        self.order_network = OrderNetwork(cfg.order_model, obs_space, cfg.order_model.ntoken_out,
                                          transform_obs=None)
        self.order_memory = OrderStorage(cfg.order_memory, obs_space, transform=None)

        optimizer_args = vars(cfg.order_model.optimizer_args)

        self.optimizer = getattr(torch.optim, cfg.order_model.optimizer)(
            self.order_network.parameters(), **optimizer_args)

        self.tgt_mask = None
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

    def add_seq_to_mem(self, *args, **kwargs):
        return self.order_memory.add_seq_to_mem(*args, **kwargs)

    def train_order(self, device="cpu"):
        optimizer = self.optimizer
        model = self.order_network
        ntokens_out = self.ntoken_out

        dataloader = self.order_memory.dataloader()
        max_batches = self.train_batches_per_epoch

        criterion = nn.CrossEntropyLoss()
        model.train()  # Turn on the train mode
        total_loss = 0.
        cur_loss = 0.

        for batch, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            data, targets = data.transpose(0, 1).contiguous(), targets.transpose(0, 1).contiguous()
            # -> change to shape S x B

            optimizer.zero_grad()

            # TODO Can do better than this! do not have to genarate this every time
            tgt_mask = model._generate_square_subsequent_mask(ntokens_out - 1).to(device)

            # Modify targets for input - Shift to the right and pad with Empty Token (max)
            in_target = targets.clone()
            in_target[1:, :] = targets[:-1, :]
            in_target[0, :] = ntokens_out - 1

            output = model(data, in_target, tgt_mask=tgt_mask)

            loss = criterion(output.view(-1, ntokens_out - 1), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            if max_batches > max_batches:
                break

        return total_loss / batch

    def eval_sequences(self, seq: torch.Tensor):
        num_samples = self.order_samples_eval
        seq_cnt = seq.size(0)
        seq_shape = seq.size()
        seq_len = self.seq_len
        b_size = self.max_eval_bath
        model = self.order_network

        # Expand sequence to number of samples Batch x seq_len x img_dim ...
        seq = seq.unsqueeze(1).expand((seq_cnt, num_samples) + seq_shape[1:]).contiguous()
        seq = seq.view((-1, ) + seq_shape[1:])

        # Get indexes & shuffled list to shuffle
        shuffled_seq, target_order = get_shuffled_sequence(seq)
        criterion = self.criterion_eval

        model.eval()

        with torch.no_grad():
            all_scores = []
            for i in range(0, shuffled_seq.size(0), b_size):

                scores = self.get_data_loss(shuffled_seq[i: i+b_size], target_order[i: i+b_size],
                                            criterion)

                all_scores.append(scores)

            scores = torch.cat(all_scores, dim=0).view(seq_cnt, num_samples, seq_len)

        return scores.mean(dim=1)

    def get_data_loss(self, data, target, criterion):
        ntoken_out = self.ntoken_out
        device = data.device

        # -> change to shape S x B
        data, target = data.transpose(0, 1).contiguous(), target.transpose(0, 1).contiguous()

        if self.tgt_mask is None or self.tgt_mask.size(0) != ntoken_out - 1:
            tgt_mask = self.order_network._generate_square_subsequent_mask(ntoken_out - 1)
            tgt_mask = tgt_mask.to(device)
            self.tgt_mask = tgt_mask

        tgt_mask = self.tgt_mask

        # Modify targets for input - Shift to the right and pad with Empty Token (max)
        in_target = target.clone()
        in_target[1:, :] = target[:-1, :]
        in_target[0, :] = ntoken_out - 1

        output = self.order_network(data, in_target, tgt_mask=tgt_mask)

        loss = criterion(output.view(-1, ntoken_out - 1), target.view(-1))
        if len(loss.size()) != 0:
            loss = loss.view(target.size()).transpose(0, 1)

        return loss


class OrderNetwork(nn.Module):

    def __init__(self, opts: Namespace, in_size, ntoken_out, transform_obs=None):
        super(OrderNetwork, self).__init__()

        ninp = opts.emsize
        nhead = opts.nhead
        nhid = opts.nhid
        nlayers = opts.nlayers
        dropout = opts.dropout
        self.ntoken_out = ntoken_out

        self.model_type = 'Transformer'
        self.transform = transform_obs
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.encoder_encoder = OrderVisualEncoder(opts, in_size, (ninp, ))
        self.encoder_decoder = nn.Embedding(ntoken_out, ninp)

        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken_out-1)

        self.init_weights()

        self.eval_criterion = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # mask.fill_(0)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder_decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        seq_len, batch_size = src.size()[:2]
        img_size = src.size()[2:]

        visual_embedding = self.encoder_encoder(src.view((-1, ) + img_size))

        src = visual_embedding.view((seq_len, batch_size, -1)) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        tgt = self.encoder_decoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        memory = self.transformer_encoder(src, mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

        # tgt_mask = tgt_mask[:6, :6].unsqueeze(0).expand(len(src), 6, 6)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        output = self.decoder(output)
        return output


class OrderStorage(Dataset):
    """Shuffle the order of numbers"""

    def __init__(self, cfg: Namespace, img_shape: Tuple[int], transform=None):

        batch_seq_len = cfg.batch_seq_len
        experience_offset = cfg.experience_offset
        memory_size = cfg.memory_size
        self.batch_size = cfg.batch_size

        assert (batch_seq_len - 1) == experience_offset, f"Incorrect offset experience_offset " \
                                                         f"should be batch_seq_len - 1 " \
                                                         f"({(batch_seq_len - 1)})"

        self.batch_seq_len = batch_seq_len
        self.transform = transform
        self.img_shape = tuple(img_shape)

        self.exp_pos = 0  # experience start pos
        self.full_memory = False
        self.exp = torch.zeros([memory_size, batch_seq_len] + img_shape)

        self.prev_obs = None
        self.prev_masks = None

        assert len(img_shape) == 3, "Image shape is not 3 dimensional"

    def dataloader(self, device=None):
        # Should fix this, overhead for creating dataloader each time
        batch_size = self.batch_size
        idxs = torch.randperm(len(self))
        exp = self.exp
        device = exp.device if device is None else device

        for i in range(0, idxs.size(0), batch_size):
            batch = exp[idxs[i: i + batch_size]].to()
            shuffled_seq, target_order = get_shuffled_sequence(batch)

            if self.transform:
                raise NotImplemented
            yield shuffled_seq, target_order

    def __len__(self):
        return self.exp.size(0) if self.full_memory else self.exp_pos

    def add_to_mem(self, obss, masks: torch.Tensor):
        batch_seq_len = self.batch_seq_len
        exp = self.exp
        exp_pos = self.exp_pos
        exp_size = exp.size(0)

        if self.prev_obs is not None:
            # TODO fix sequence at the border :(
            # prev_obs = self.prev_obs
            # prev_masks = self.prev_masks
            prev_obs = obss
            prev_masks = masks

            # We will not consider overlapping sequences
            for row in range(prev_masks.size(0)):
                for i in range(0, prev_masks.size(1), batch_seq_len):
                    if prev_masks[row, i + 1: i + batch_seq_len].all():
                        exp[exp_pos] = prev_obs[row, i: i + batch_seq_len]
                        exp_pos = (exp_pos + 1) % exp_size
                    else:
                        # Include last sequence in episode
                        z = i + 1
                        while prev_masks[row, z]:
                            z += 1

                        exp[exp_pos] = prev_obs[row, z - batch_seq_len: z]
                        exp_pos = (exp_pos + 1) % exp_size

                        # TODO Add extra sequence with an empty image for ending (sim Joker token)

            self.exp_pos = exp_pos

        self.prev_obs = obss
        self.prev_masks = masks

    def add_seq_to_mem(self, sequences: torch.Tensor):
        exp = self.exp
        exp_pos = self.exp_pos
        exp_size = exp.size(0)
        no_seq = sequences.size(0)
        if no_seq > exp_size - exp_pos:
            first_batch = exp_size - exp_pos
            exp[exp_pos:] = sequences[:first_batch]
            sec_batch = (no_seq - first_batch)
            exp[:sec_batch] = sequences[first_batch:]
            self.exp_pos = sec_batch
            self.full_memory = True
        else:
            self.exp_pos = exp_pos + no_seq
            exp[exp_pos: self.exp_pos] = sequences

    def __getitem__(self, idx):
        seq_len = self.batch_seq_len
        exp = self.exp

        seq = exp[idx]
        seq_order = torch.randperm(seq_len)
        seq = seq[seq_order]

        act_seq_order = torch.zeros_like(seq_order)
        act_seq_order[seq_order] = torch.arange(seq_len)
        seq = seq.float()

        if self.transform:
            seq = self.transform(seq)

        return seq, act_seq_order

