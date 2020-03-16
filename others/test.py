from typing import List
import torch
from scipy.signal import convolve2d
import numpy as np

"""
CTR + P - info about method
CTR + SPACE - autocomplete
CTR + Click (on method/variable) = got to definition
CTR + (left/right arrow) - move with word not character
CTR - Shift - E = search everywhere
ALT - Shift - e - run line/ block
CTR - / - comment  line / block (same for uncomment)
Shift - F6 = rename variable
CTR - D = duplicate line
CTR + Shift + sagetute = move line/ block

CTR - Alt - e : in console search history

import pdb; pdb.set_trace() = debug


---- in terminal
source activate env

-> install (! within env) all libraries 
->  INSTALL with (from terminal)
        conda install package
        pip install package
        
"""

import torch
import time


no_tests = 10


def get_shuffle_seq(seq):
    seq_len = seq.size(0)

    seq_order = torch.randperm(seq_len)
    seq = seq[seq_order]

    act_seq_order = torch.zeros_like(seq_order)
    act_seq_order[seq_order] = torch.arange(seq_len)
    seq = seq.float()

    return seq, act_seq_order.to(seq.device)


data = torch.rand(1500, 6, 3, 4).cuda()

st = time.time()
for _ in range(no_tests):
    s, so = [], []
    for i in range(data.size(0)):
        _s, _so = get_shuffle_seq(data[i])
        s.append(_s)
        so.append(_so)
s = torch.stack(s)
so = torch.stack(so).cuda()

end = time.time() - st
print(end)

st = time.time()
for _ in range(no_tests):
    rand = torch.rand(data.size()[:2], device=data.device)
    batch_rand_perm = rand.argsort(dim=1)
    batch_rand_perm = batch_rand_perm[(..., ) + (None, ) * (len(data.size())-2)].expand_as(data)
    shuffled = data.gather(1, batch_rand_perm)
end = time.time() - st
print(end)

data = torch.arange(6).unsqueeze(0).expand(10, 6)

rand = torch.rand(data.size()[:2], device=data.device)
batch_rand_perm = rand.argsort(dim=1)
batch_rand_perm = batch_rand_perm[(...,) + (None,) * (len(data.size()) - 2)].expand_as(data)
shuffled = data.gather(1, batch_rand_perm)

ordered = torch.arange(data.size(1)).unsqueeze(0).expand(data.size()[:2])
correct_order = torch.zeros_like(ordered)
correct_order.scatter_(1, batch_rand_perm, ordered)


def test(a: int, b: List[int]) -> str:
    c = [x + a for x in b]
    return str(c[0])


def game():
    img = np.zeros((10, 10))
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    # kernel = np.ones((3, 3))
    # kernel[1, 1] = 0
    next_img = np.zeros_like(img)

    neighb = convolve2d(img, kernel, mode='same', )

    next_img[:] = (~((neighb < 3) & (img == 1)))


# ==================================================================================================

input_data = [np.rand(1,1,10,10) for _ in range(10000)]
target_data = [game(x) for x in range(100000)]


from torch.utils.data import Dataset, DataLoader
from torch import optim

def generate_data(img):
    return img


class GameOfLifeData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_size = 128
        no_ex = 10

        self.x = [(torch.rand(img_size, img_size) < 0.5).float() for i in range(no_ex)]
        self.y = [generate_data(img) for img in self.x]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


class GameNet(torch.nn.Module):
    def __init__(self):
        super(GameNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 1, (3, 3), stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = self.conv3(x)
        return x


model = GameNet()

transformed_dataset = GameOfLifeData()

dataloader = DataLoader(transformed_dataset, batch_size=6,
                        shuffle=True, num_workers=4)

optimizer = optim.RMSprop(model.parameters(), lr=0.001)
loss = torch.nn.BCEWithLogitsLoss()

no_epochs = 100
for ep in range(no_epochs):
    for i, (img, target) in enumerate(dataloader):
        out = model(img)

        loss = loss(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


import cv2

t = out[0,0].int().numpy()
cv2.imshow("test", t)
cv2.waitKey(0)

imgt = torch.from_numpy(img)
imgt = imgt.unsqueeze(0).unsqueeze(0).float()
# imgt = torch.zeros((1,1, 10, 10))

model = GameNet()

out = model(imgt)
print(f"IMG size {out.size()}")

# ==================================================================================================

class Test:
    def __init__(self):
        pass

    def what(self):
        for i in range(0, 10, 2):
            yield i

t = Test()
d = t.what()

for ii, i in enumerate(d):
    print(i)

class Ana:
    def __init__(self):
        self.aa = 2  # type: int
        self.c = 3  # type: int

    def test(self):
        t = self.aa

        a = 3 + 4 + 3 * 2
        t += 1


if __name__ == "__main__":
    a = Ana()
    a.test()
# ==================================================================================================
import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper
import numpy as np
import cv2

env_key = "MiniGrid-FourRooms-v0"
seed = 0

env = gym.make(env_key, agent_pos = (1,1), goal_pos = None, doors = True)
env.max_steps = 400
env = FullyObsWrapper(env)
env.seed(seed)

obs = env.reset()["image"]


while True:
    act = np.random.randint(3)
    obs, r, done, info = env.step(act)

    img = obs["image"] * 15
    img = cv2.resize(img, (0, 0), fx=20, fy=20)
    cv2.imshow("test", img)
    cv2.waitKey(1)
    if done:
        env.reset()
        print("RESET")

act = np.random.randint(3)
