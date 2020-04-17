# Dan Iulian Muntean 2020
# inspired from https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/r_network.py


"""R-network and some related functions to train R-networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl

from typing import Optional, Tuple
from models.utils import initialize_parameters2, initialize_parameters_ec
from models.model_replica import Model


class EpisodicCuriosityModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False):
        super().__init__()

        self.policy_model = WorldsPolicyModel(cfg, obs_space, action_space, use_memory=use_memory)

        self.curiosity_model = RNetwork(cfg, obs_space)

        self.memory_type = self.policy_model.memory_type
        self.use_memory = self.policy_model.use_memory

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class WorldsPolicyModel(Model):
    '''The same model as the one used for PPO
    For details see replica_model.Model
    '''
    def __init__(self, cfg, obs_space, action_space, use_memory=False):
        super(WorldsPolicyModel, self).__init__(cfg, obs_space, action_space, use_memory)

        # Define value heads
        #self.vf_int = nn.Linear(self.mem_size, 1)
        #self.vf_ext = self.vf

        # Initialize parameters correctly
        #initialize_parameters2(self.vf_int)

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
        #vpred_int = self.vf_int(val).squeeze(1)
        vpred_ext = self.vf(val).squeeze(1)

        # Action head
        pd = self.pd(act)
        dist = Categorical(logits=F.log_softmax(pd, dim=1))

        return dist, vpred_ext, memory


class RNetwork(nn.Module):
    """The R network architecture. While the paper uses a ResNet-18 for embedding extraction,
       I tried using a much simpler architecture given that our observations have dimensions 7x7x3
    """

    def __init__(self, cfg, obs_space):
        super(RNetwork, self).__init__()

        print(f"OBS space {obs_space}")
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        # set the similarity function to be used
        self._similarity = getattr(cfg, "similarity_computation", "simple")
        if self._similarity == "simple":
            self.forward_similarity = self._forward_similarity_simple
        else:
            self.forward_similarity = self._forward_similarity_complex


        self._embedding_size = getattr(cfg, "r_net_embedding_size", 512) # embedding size after CNN processing
        k_sizes = getattr(cfg, "r_net_k_sizes", [2, 2, 2])  # kernel size for each layer
        s_sizes = getattr(cfg, "r_net_s_sizes", [1, 1, 1])  # stride size for each layer
        comp_sizes = getattr(cfg, "r_net_comp_sizes", [512, 512, 256]) # number of units for comparator


        # experiment used model
        self.image_embedding = nn.Sequential(
            nn.Conv2d(3, 32, k_sizes[0], s_sizes[0]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, k_sizes[1], s_sizes[1]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, k_sizes[2], s_sizes[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            out_conv_size = self.image_embedding(torch.rand((1, obs_space["image"][2], n, m))).size()
        self.image_embedding_size = int(np.prod(out_conv_size))

        self.fc1 = nn.Sequential(
            nn.Linear(self.image_embedding_size, self._embedding_size),
            nn.ReLU(inplace=True),
        )

        self.comparator = nn.Sequential(
            nn.Linear(2 * self._embedding_size, comp_sizes[0]),
            nn.BatchNorm1d(comp_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(comp_sizes[0], comp_sizes[1]),
            nn.BatchNorm1d(comp_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(comp_sizes[1], comp_sizes[2]),
            nn.BatchNorm1d(comp_sizes[2]),
            nn.ReLU(inplace=True),
            # return two classes for the probability of reaching one obs from another
            nn.Linear(comp_sizes[2], 2),
        )

        self.apply(initialize_parameters_ec)

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, obs):
        # get the observation embedding
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()
        x = self.image_embedding(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def forward_similarity(self, emb1, emb2):
        return None

    def _forward_similarity_complex(self, emb1, emb2):
        x = torch.cat([emb1, emb2], dim=1)
        return self.comparator(x)

    def _forward_similarity_simple(self, emb1, emb2):
        """A simple top network that basically computes sigmoid(dot_product(x1, x2)).
        """
        dot_product = torch.sum(emb1 * emb2, axis=1)
        sigm = torch.sigmoid(dot_product)
        return torch.stack((1 - sigm, sigm), axis=1)

