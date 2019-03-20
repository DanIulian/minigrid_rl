# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import numpy as np

from models.utils import initialize_parameters
from torch_rl.utils import DictList, ParallelEnv
from utils.format import preprocess_images


class WorldsModels(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        connect_worlds = getattr(cfg, "connect_worlds", True)

        self.policy_model = WorldsPolicyModel(cfg, obs_space, action_space, use_memory=use_memory,
                                              use_text=use_text)

        self.agworld_network = AgentWorld(cfg, obs_space, action_space)

        # Configure action embedding size for connected worlds
        if connect_worlds:
            cfg.env_action_emb_size = self.agworld_network.memory_size

        self.envworld_network = EnvWorld(cfg, obs_space, action_space)

        self.evaluator_network = EvaluationNet(cfg,
                                               self.envworld_network.memory_size,
                                               cfg.full_state_size)

        # self.random_target = RandomNetwork(cfg, obs_space, action_space)

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
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)-2)*((m-1)-2)*64

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
            nn.ReLU(),
        )

        self.fc2_act = nn.Sequential(
            nn.Linear(self.embedding_size, memory_size),
            nn.ReLU(),
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


class EnvWorld(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(EnvWorld, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        action_emb_size = getattr(cfg, "env_action_emb_size", action_space.n)

        hidden_size = 512  # getattr(cfg, "hidden_size", 256)
        self._memory_size = memory_size = 512  # getattr(cfg, "memory_size", 256)
        channels = 3
        self.action_space = torch.Size((channels, n, m))

        out_size = n * m * channels

        self.image_conv = nn.Sequential(
            nn.Conv2d(channels, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(image_embedding_size, hidden_size),
        )

        self.memory_rnn = nn.GRUCell(hidden_size + action_space.n, memory_size)

        # # Next state prediction
        self.fc2 = nn.Sequential(
            nn.Linear(memory_size + action_emb_size, memory_size),
            # nn.BatchNorm1d(memory_size),
            nn.LeakyReLU(),
            nn.Linear(memory_size, memory_size),
            # nn.BatchNorm1d(memory_size),
            nn.LeakyReLU(),
            nn.Linear(memory_size, out_size),

        )

        # self.fc2 = nn.Sequential(
        #     nn.ConvTranspose2d(memory_size + action_emb_size, memory_size, (1, 1)),
        #     # nn.BatchNorm2d(memory_size),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(memory_size, memory_size, (2, 2)),
        #     # nn.BatchNorm2d(memory_size),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(memory_size, 3, (6, 6)),
        #     # nn.BatchNorm2d(64),
        #     # nn.LeakyReLU(),
        # )

    @property
    def memory_size(self):
        return self._memory_size

    def forward(self, x, memory, action_prev, action_next):

        b_size = x.size()

        x = self.image_conv(x)
        x = x.view(b_size[0], -1)

        x = self.fc1(x)

        x = torch.cat([x, action_prev], dim=1)

        x = memory = self.memory_rnn(x, memory)

        x = torch.cat([x, action_next], dim=1)

        # x = x.unsqueeze(2).unsqueeze(2)

        x = self.fc2(x)
        return x.view(b_size[:1] + self.action_space), memory


class AgentWorld(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(AgentWorld, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        hidden_size = 512  # getattr(cfg, "hidden_size", 256)
        self._memory_size = memory_size = 256  # getattr(cfg, "memory_size", 256)
        channels = 3
        self.action_space = torch.Size((action_space.n, ))

        out_size = n * m * channels

        self.image_conv = nn.Sequential(
            nn.Conv2d(channels, 16, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.LeakyReLU(),
        )

        image_embedding_size = ((n - 1) - 2) * ((m - 1) - 2) * 64

        # Consider embedding out of fc1
        self.fc1 = nn.Sequential(
            nn.Linear(image_embedding_size, hidden_size),
        )

        self.memory_rnn = nn.GRUCell(hidden_size + action_space.n, memory_size)

        # Next state prediction
        self._embedding_size = embedding_size = hidden_size  # memory_size

        self.fc2 = nn.Sequential(
            nn.Linear(memory_size + embedding_size, memory_size),
            # nn.Linear(hidden_size + hidden_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, action_space.n),
            # nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(memory_size + embedding_size, memory_size),
            # nn.Linear(hidden_size + hidden_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, action_space.n),
            # nn.ReLU()
        )


    @property
    def memory_size(self):
        return self._memory_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, x, memory, action_prev, action_next):
        b_size = x.size()

        x = self.image_conv(x)
        x = x.view(b_size[0], -1)

        x = local_embedding = self.fc1(x)

        local_embedding = nn.ReLU()(local_embedding)

        x = torch.cat([x, action_prev], dim=1)

        x = memory = self.memory_rnn(x, memory)

        # local_embedding = memory

        return None, memory, local_embedding

    def forward_action(self, x, embedding):
        x = torch.cat([x, embedding], dim=1)

        x = self.fc2(x)
        return x

    def forward_action_eval(self, x, embedding):
        x = torch.cat([x, embedding], dim=1)

        x = self.fc3(x)
        return x


class EvaluationNet(nn.Module):
    def __init__(self, cfg, obs_space, out_space):
        super(EvaluationNet, self).__init__()

        hidden_size = getattr(cfg, "hidden_size", 256)

        self.out_space = torch.Size([out_space[2], *out_space[:2]])
        out_space = np.prod(out_space)

        self.fc1 = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_space),
            nn.ReLU()
        )

    def forward(self, x):
        b_size = x.size(0)

        x = self.fc1(x)

        return x.view(b_size, *self.out_space)


