# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from typing import Optional, Tuple

from models.utils import initialize_parameters


class Model(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False):
        ''' Standard CNN model used for PPO

        :param cfg: A parsed config file
        :param obs_space: The dimensions of the observation space 3D vector
        :param action_space: Number of actions the agent can take
        :param use_memory: True if the agent uses LSTM/GRU
        '''

        super().__init__()

        # extract necessary info from config file
        self.memory_type = cfg.memory_type
        self.mem_size = getattr(cfg, "memory_size", 128)

        hidden_size = getattr(cfg, "hidden_size", 128)  # feature size after CNN processing
        k_sizes = getattr(cfg, "k_sizes", [5, 5, 3])    # kernel size for each layer
        s_sizes = getattr(cfg, "s_sizes", [3, 3, 1])    # stride size for each layer

        # Decide which components are enabled
        self.use_memory = use_memory

        # experiment used model
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, k_sizes[0], s_sizes[0]),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, k_sizes[1], s_sizes[1]),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, k_sizes[2], s_sizes[2]),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        print(f"OBS space {obs_space}")
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        out_conv_size = self.image_conv(torch.rand((1, obs_space["image"][2], n, m))).size()
        out_feat_size = int(np.prod(out_conv_size))
        self.image_embedding_size = out_feat_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.image_embedding_size, hidden_size),
            # nn.ReLU(inplace=True),
        )

        crt_size = hidden_size

        # Define memory
        if self.use_memory:
            if self.memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(crt_size, self.mem_size)
            else:
                self.memory_rnn = nn.GRUCell(crt_size, self.mem_size)

            crt_size = self.mem_size

        # Resize image embedding
        self.embedding_size = crt_size

        self.fc2_val = nn.Sequential(
            nn.Linear(self.embedding_size, self.mem_size),
            nn.ReLU(inplace=True),
        )
        self.fc2_act = nn.Sequential(
            nn.Linear(self.embedding_size, self.mem_size),
            nn.ReLU(inplace=True),
        )

        # Define action and value heads
        self.vf = nn.Linear(self.mem_size, 1)
        self.pd = nn.Linear(self.mem_size, action_space.n)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.memory_type == "LSTM":
            return 2 * self.mem_size
        else:
            return self.mem_size

    @property
    def semi_memory_size(self):
        return self.embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        if self.use_memory:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)  # type: Tuple[torch.Tensor]
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                hidden = memory  # type: Optional[torch.Tensor]
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden
                memory = hidden
        else:
            embedding = x

        val = self.fc2_val(embedding)
        act = self.fc2_act(embedding)

        # Value function head
        vpred = self.vf(val).squeeze(1)

        # Action head
        pd = self.pd(act)
        dist = Categorical(logits=F.log_softmax(pd, dim=1))

        return dist, vpred, memory
