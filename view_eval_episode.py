
import time
import argparse
import numpy as np
import os
import sys
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
from natsort import natsorted

def show_episode(path):
    window = Window('Gym Minigrid')

    images_list = os.listdir(path)
    sorted_images = natsorted(images_list)
    for img_path in sorted_images:
        img = plt.imread(f"{path}/{img_path}")
        window.show_img(img)

        time.sleep(0.1)

if __name__ == '__main__':
    path = sys.argv[1]
    show_episode(path)

