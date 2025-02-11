# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl


from models.utils import initialize_parameters


class RNDModels(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        self.policy_model = RNDPredModel(cfg, obs_space, action_space, use_memory=use_memory,
                                         use_text=use_text)

        self.predictor_network = PredictionNetwork(cfg, obs_space, action_space)

        self.random_target = RandomNetwork(cfg, obs_space, action_space)

        self.memory_type = self.policy_model.memory_type
        self.use_text = self.policy_model.use_text
        self.use_memory = self.policy_model.use_memory

    @property
    def memory_size(self):
        return self.policy_model.memory_size

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class RNDPredModel(nn.Module, torch_rl.RecurrentACModel):
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
        #self.image_conv = nn.Sequential(
        #    nn.Conv2d(3, 16, (2, 2)),
        #    nn.ReLU(),
        #    nn.Conv2d(16, 32, (2, 2)),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 64, (2, 2)),
        #    nn.ReLU()
        #)

        #experiments used model
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

        #self.image_embedding_size = ((n-1)-2)*((m-1)-2)*64
        self.image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2) * 64

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


class PredictionNetwork(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(PredictionNetwork, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        hidden_size = getattr(cfg, "hidden_size", 256)

        #self.image_conv = nn.Sequential(
        #    nn.Conv2d(3, 16, (2, 2)),
        #    nn.LeakyReLU(),
        #    nn.Conv2d(16, 32, (2, 2)),
        #    nn.LeakyReLU(),
        #    nn.Conv2d(32, 64, (2, 2)),
        #    nn.LeakyReLU()
        #)

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

        image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(image_embedding_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
        )

    def forward(self, x):
        b_size = x.size(0)

        x = self.image_conv(x)
        x = x.view(b_size, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class RandomNetwork(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super(RandomNetwork, self).__init__()
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        hidden_size = getattr(cfg, "hidden_size", 256)

        #self.image_conv = nn.Sequential(
        #    nn.Conv2d(3, 16, (2, 2)),
        #    nn.LeakyReLU(),
        #    nn.Conv2d(16, 32, (2, 2)),
        #    nn.LeakyReLU(),
        #    nn.Conv2d(32, 64, (2, 2)),
        #    nn.LeakyReLU()
        #)

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (2, 2)),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (2, 2)),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        image_embedding_size = ((n - 2) - 2) * ((m - 2) - 2) * 64

        #image_embedding_size = ((n - 1) - 2) * ((m - 1) - 2) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(image_embedding_size, hidden_size),
            #nn.ReLU(),
        )

    def forward(self, x):
        b_size = x.size(0)
        x = self.image_conv(x)
        x = x.view(b_size, -1)

        x = self.fc1(x)
        return x
