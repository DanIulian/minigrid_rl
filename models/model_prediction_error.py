# DanM, 2019
# parts from https://github.com/lcswillems/torch-rl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from models.utils import initialize_parameters


class PEModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        self.policy_model = WorldsPolicyModel(cfg, obs_space, action_space, use_memory=use_memory,
                                              use_text=use_text)

        self.curiosity_model = CuriosityModel(cfg, obs_space, action_space)

        self.memory_type = self.policy_model.memory_type
        self.use_text = self.policy_model.use_text
        self.use_memory = self.policy_model.use_memory

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class WorldsPolicyModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # CFG Information
        self.memory_type = memory_type = cfg.memory_type
        hidden_size = getattr(cfg, "hidden_size", 128)
        self._memory_size = memory_size = getattr(cfg, "memory_size", 128)

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2)*64

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


class CuriosityModel(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(CuriosityModel, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self.obs_channel1 = [11, 7, 7]
        self.obs_channel2 = [6, 7, 7]
        self.obs_channel3 = [3, 7, 7]


        hidden_size = getattr(cfg, "hidden_size", 256)
        self._memory_size = memory_size = getattr(cfg, "memory_size", 256)
        channels = 3
        self.action_space = torch.Size((action_space.n, ))

        out_size = n * m * channels

        self.image_conv = nn.Sequential(
            nn.Conv2d(channels, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, (2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, (2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2) * 64

        # Consider embedding out of fc1
        self.fc1 = nn.Sequential(
            nn.Linear(image_embedding_size, hidden_size),
        )

        self.memory_rnn = nn.GRUCell(hidden_size + action_space.n, memory_size)

        # Next state prediction
        self._embedding_size = embedding_size = hidden_size  # memory_size

        self.pe = nn.Sequential(
            nn.Linear(memory_size + action_space.n, memory_size),
            nn.ReLU(inplace=True),
        )

        self.next_obs1 = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.ReLU(inplace=True),
            nn.Linear(memory_size, np.prod(self.obs_channel1)),
        )

        self.next_obs2 = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.ReLU(inplace=True),
            nn.Linear(memory_size, np.prod(self.obs_channel2)),
        )

        self.next_obs3 = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.ReLU(inplace=True),
            nn.Linear(memory_size, np.prod(self.obs_channel3)),
        )


    @property
    def memory_size(self):
        return self._memory_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, x, memory, action_prev):
        b_size = x.size()

        x = self.image_conv(x)
        x = x.view(b_size[0], -1)

        x = local_embedding = self.fc1(x)

        local_embedding = nn.ReLU()(local_embedding)

        x = torch.cat([x, action_prev], dim=1)

        x = memory = self.memory_rnn(x, memory)

        return None, memory, local_embedding

    def forward_state(self, embedding, action_next):
        x = torch.cat([embedding, action_next], dim=1)

        next_obs1 = self.next_obs1(self.pe(x))
        next_obs2 = self.next_obs2(self.pe(x))
        next_obs3 = self.next_obs3(self.pe(x))

        next_obs1 = next_obs1.view(x.size()[0], self.obs_channel1[0], self.obs_channel1[1], self.obs_channel1[2])
        next_obs2 = next_obs2.view(x.size()[0], self.obs_channel2[0], self.obs_channel2[1], self.obs_channel2[2])
        next_obs3 = next_obs3.view(x.size()[0], self.obs_channel3[0], self.obs_channel3[1], self.obs_channel3[2])

        return next_obs1, next_obs2, next_obs3

