# DanMuntean, 2020

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
from models.utils import initialize_parameters, initialize_parameters2
from models.model_replica import Model


class RNDICMPretrainModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False):
        super().__init__()

        self.policy_model = WorldsPolicyModel(cfg, obs_space, action_space, use_memory)
        self.curiosity_model = ICMPretrainModel(cfg, obs_space, action_space)

        self.predictor_network = PredictionNetwork(cfg, self.curiosity_model.embedding_size)

        self.random_target = RandomNetwork(cfg, self.curiosity_model.embedding_size)

        self.memory_type = self.policy_model.memory_type
        self.use_memory = self.policy_model.use_memory

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class WorldsPolicyModel(Model):
    ''' The same model as the one used for PPO
    For details see replica_model.Model
    '''
    def __init__(self, cfg, obs_space, action_space, use_memory):

        super(WorldsPolicyModel, self).__init__(cfg, obs_space, action_space, use_memory)

        # Define value heads
        self.vf_int = nn.Linear(self.mem_size, 1)
        self.vf_ext = self.vf

        # Initialize parameters correctly
        initialize_parameters2(self.vf_int)

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
        vpred_int = self.vf_int(val).squeeze(1)
        vpred_ext = self.vf_ext(val).squeeze(1)

        # Action head
        pd = self.pd(act)
        dist = Categorical(logits=F.log_softmax(pd, dim=1))

        return dist, (vpred_ext, vpred_int), memory


class PredictionNetwork(nn.Module):
    def __init__(self, cfg, input_space):
        super(PredictionNetwork, self).__init__()

        self._input_space = input_space
        self._embedding_sizes = getattr(cfg, "embedding_sizes", [256, 128, 128])

        self.fc = nn.Sequential(
            nn.Linear(self._input_space, self._embedding_sizes[0]),
            nn.BatchNorm1d(self._embedding_sizes[0]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self._embedding_sizes[0], self._embedding_sizes[1]),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(self._embedding_sizes[1]),
            nn.Linear(self._embedding_sizes[1], self._embedding_sizes[2]),
        )
        self.apply(initialize_parameters2)

    @property
    def memory_size(self):
        return self._input_space

    @property
    def embedding_size(self):
        return self._input_space

    def forward(self, x):
        x = self.fc(x)
        return x


class RandomNetwork(nn.Module):
    def __init__(self, cfg, input_space):
        super(RandomNetwork, self).__init__()

        self._input_space = input_space
        self._embedding_sizes = getattr(cfg, "embedding_sizes", [256, 128])

        self.fc = nn.Sequential(
            nn.Linear(input_space, self._embedding_sizes[0]),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self._embedding_sizes[0], self._embedding_sizes[1]),
            nn.LeakyReLU(inplace=True),
        )
        self.apply(initialize_parameters2)

    def forward(self, x):
        x = self.fc(x)
        return x


class ICMPretrainModel(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(ICMPretrainModel, self).__init__()

        print(f"OBS space {obs_space}")
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self._embedding_size = getattr(cfg, "embedding_size", 256)
        self.action_space = torch.Size((action_space.n, ))
        k_sizes = getattr(cfg, "k_sizes", [3, 2, 2])    # kernel size for each layer
        s_sizes = getattr(cfg, "s_sizes", [1, 1, 1])    # stride size for each layer

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, k_sizes[0], s_sizes[0]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 32, k_sizes[1], s_sizes[1]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, k_sizes[2], s_sizes[2]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        with torch.no_grad():
            out_conv_size = self.image_conv(torch.rand(1, obs_space["image"][2], n, m)).size()

        self._image_embedding_size = int(np.prod(out_conv_size))

        # Consider embedding out of fc1
        self.fc1 = nn.Sequential(
            nn.Linear(self._image_embedding_size, self._embedding_size),
            nn.LeakyReLU(inplace=True)
        )

        # Inverse dynamics prediction
        # given the phi(s_t), phi(s_t+1) return p(a_t| phi(s_t), phi(s_t+1))
        self.fc_beta = nn.Sequential(
            nn.Linear(self._embedding_size + self._embedding_size, self._embedding_size),  # phi(s_t) + phi(s_t+1)
            nn.LeakyReLU(inplace=True),

            nn.Linear(self._embedding_size, action_space.n),
        )

        # Forward dynamics prediction
        # given at and bt return bt+1
        self.fc_alpha = nn.Sequential(
            nn.Linear(self._embedding_size + action_space.n, self._embedding_size),  # phi(s_t) + a_t
            nn.LeakyReLU(inplace=True),

            nn.Linear(self._embedding_size, self._embedding_size)
        )


        self.apply(initialize_parameters2)

    @property
    def memory_size(self):
        return self._embedding_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, x):
        b_size = x.size()

        x = self.image_conv(x)
        x = x.reshape(b_size[0], -1)
        x = self.fc1(x)

        return x

    def forward_action(self, curr_state_embedding, next_state_embedding):
        z = torch.cat([curr_state_embedding, next_state_embedding], dim=1)
        z = self.fc_beta(z)
        return z

    def forward_state(self, embedding, action_next):
        x = torch.cat([embedding, action_next], dim=1)
        x = self.fc_alpha(x)
        return x

