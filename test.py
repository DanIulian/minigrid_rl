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

