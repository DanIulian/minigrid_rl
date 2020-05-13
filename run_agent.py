''' Dan Iulian Muntean 2020
    Generic agent for policy evaluation
'''

from gym_minigrid.wrappers import *
import torch
import torch.nn

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tqdm
import gym

import utils
from gym_minigrid.window import Window
from models import get_model
from agents import get_agent
from utils import gym_wrappers
from argparse import Namespace

try:
    import gym_minigrid
except ImportError:
    print("Can not import MiniGrid-ENV")
    exit(0)



class RGBImgWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['rendered_image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'rendered_image': rgb_img
        }


class RGBImgPartialWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['rendered_image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'rendered_image': rgb_img_partial
        }


class EvalAgent(object):

    def __init__(self,
                 env_name,
                 path_to_checkpoint,
                 nr_runs=2,
                 argmax=False,
                 view_type="FullView",
                 max_steps=400):

        self._env_name = env_name
        self._path_to_checkpoint = path_to_checkpoint
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nr_runs = nr_runs
        self._view_type = view_type
        self._argmax = argmax
        self._max_steps = max_steps
        self._window = Window(self._env_name)
        self._make_env()
        _, self._obs_preprocess_fn = self.obs_preprocess()
        self.load_model()

        if self._model.recurrent:
            self._memories = torch.zeros(1, self._model.memory_size, device=self._device)

        self._step_count = 0

        print("Argmax - {}".format(self._argmax))

    def _make_env(self):
        # create a new env
        self._eval_env = gym.make(self._env_name)
        self._eval_env.max_steps = self._max_steps
        self._eval_env.seed(np.random.randint(1, 10000))

        if self._view_type == "FullView":
            self._eval_env = RGBImgWrapper(self._eval_env)
        elif self._view_type == "AgentView":
            self._eval_env = RGBImgPartialWrapper(self._eval_env)
        else:
            raise ValueError("Incorrect view name: {}".format(self._view_type))

    def obs_preprocess(self) -> tuple:
        # Define obss preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(self._env_name,
                                                                 self._eval_env.observation_space,
                                                                 '.',
                                                                 max_image_value=15.0,
                                                                 normalize=True,
                                                                 permute=False,
                                                                 obs_type="compact",
                                                                 type=None)
        return obs_space, preprocess_obss

    def load_model(self):
        print("Checkpoint is in {}".format(self._path_to_checkpoint))
        try:
            status = utils.load_status(self._path_to_checkpoint)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        saver = utils.SaveData(self._path_to_checkpoint, save_best=True, save_all=True)
        self._model, self._agent_data, self._other_data = None, dict(), dict()
        try:
            # Continue from last point
            self._model, self._agent_data, self._other_data = saver.load_training_data(best=False)
            print("Training data exists & loaded successfully\n")
        except OSError:
            print("Could not load training data\n")

    def run_episode(self):

        print("Evaluating agent after for {} episodes".format(self._nr_runs))
        self._model.eval()

        for i in tqdm.tqdm(range(self._nr_runs)):
            obs = self._eval_env.reset()
            human_obs = obs["rendered_image"]
            episode_obs = [human_obs]

            while True:
                #preprocess observation
                preprocessed_obs = self._obs_preprocess_fn([obs], device=self._device)

                with torch.no_grad():
                    if self._model.recurrent:
                        dist, _, self._memories = self._model(preprocessed_obs, self._memories)
                    else:
                        dis, _ = self._model(preprocessed_obs)

                if self._argmax:
                    _, action, = dist.probs.max(1, keepdim=True)
                else:
                    action = dist.sample()

                print(action.cpu().numpy(), dist.probs)
                obs, reward, done, info = self._eval_env.step(action.cpu().numpy())
                self._window.show_img(obs["rendered_image"])
                if done:
                    break

        self._model.train()


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parse = ArgumentParser()
    arg_parse.add_argument(
        '--env_name',
        default="MiniGrid-DoorKey16x16-v0"
    )
    arg_parse.add_argument(
        '--path_to_checkpoint',
        default=None
    )
    arg_parse.add_argument(
        '--nr_runs',
        default=2,
        type=int
    )
    arg_parse.add_argument(
        '--max_steps',
        default=400,
        type=int
    )
    arg_parse.add_argument(
        '--view_type',
        default="FullView"
    )

    args = arg_parse.parse_args()

    eval_agent = EvalAgent(args.env_name,
                           args.path_to_checkpoint,
                           nr_runs=args.nr_runs,
                           view_type=args.view_type,
                           max_steps=args.max_steps)
    eval_agent.run_episode()

