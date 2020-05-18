# DanM, 2020

import numpy as np
import torch
import torch.nn as nn
import torch_rl
from models.model_icm_like import WorldsPolicyModel
from models.utils import initialize_parameters2


class DisagreementModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False):
        super().__init__()

        # Create the policy model
        self.policy_model = WorldsPolicyModel(cfg, obs_space, action_space, use_memory)
        self.memory_type = self.policy_model.memory_type
        self.use_memory = self.policy_model.use_memory

        self.feature_extractor = RandomFeatureExtractor(cfg, obs_space, action_space)
        # Create dynamics list
        nr_heads = getattr(cfg, "nr_dyn", 5)
        self.dyn_list = nn.ModuleList([DynamicsModel(cfg, obs_space, action_space) for _ in range(nr_heads)])

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class RandomFeatureExtractor(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(RandomFeatureExtractor, self).__init__()

        print(f"OBS space {obs_space}")
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self._embedding_size = getattr(cfg, "embedding_size", 512)
        k_sizes = getattr(cfg, "k_sizes", [3, 2, 2])    # kernel size for each layer
        s_sizes = getattr(cfg, "s_sizes", [1, 1, 1])    # stride size for each layer

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, k_sizes[0], s_sizes[0]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, k_sizes[1], s_sizes[1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, k_sizes[2], s_sizes[2]),
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

        self.apply(initialize_parameters2)

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def memory_size(self):
        return self._embedding_size

    @property
    def network_type(self):
        return "random"

    def forward(self, x):
        b_size = x.size()

        x = self.image_conv(x)
        x = x.reshape(b_size[0], -1)

        local_embedding = self.fc1(x)

        return local_embedding


class DynamicsModel(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(DynamicsModel, self).__init__()

        self._embedding_size = getattr(cfg, "embedding_size", 512)
        self._residual_size = getattr(cfg, "residual_size", 512)
        self._nr_residuals = getattr(cfg, "nr_res_blocks", 4)

        self.fc1 = nn.Sequential(
            nn.Linear(self._embedding_size + action_space.n, self._residual_size),
            nn.LeakyReLU(inplace=True),
        )

        def _get_residual_bloc():
            return nn.Sequential(
                nn.Linear(self._residual_size + action_space.n, self._residual_size),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self._residual_size, self._residual_size),
                nn.LeakyReLU(inplace=True),
            )

        self.res_blocks = nn.ModuleList()

        for _ in range(self._nr_residuals):
            self.res_blocks.append(_get_residual_bloc())

        self.fc2 = nn.Linear(self._residual_size, self._embedding_size)

        self.apply(initialize_parameters2)

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def residual_size(self):
        return self._residual_size

    def forward(self, x, action_prev):
        x = torch.cat([x, action_prev], dim=1)
        x = self.fc1(x)

        for block in self.res_blocks:
            local_embedding = x
            x = torch.cat([x, action_prev], dim=1)
            x = block(x)
            x = x + local_embedding

        x = self.fc2(x)

        return x
