''' Dan Iulian Muntean 2020
    Generic agent for policy evaluation
'''

from gym_minigrid.wrappers import *
import torch
import torch.nn
import numpy as np
import os
import tqdm
import gym

import utils as ut
from gym_minigrid.window import Window
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

try:
    import gym_minigrid

except ImportError:
    print("Can not import MiniGrid-ENV")
    exit(0)

from gym_minigrid.minigrid import *
from gym_minigrid.envs.multiroom import Room



ROOMS_LIST = {
    "MultiRoomN11S8_v1": [
        Room((11, 1),  (8, 5), (11, 1),  None),
        Room((10, 5),  (8, 8), (16, 5),  None),
        Room((17, 6),  (4, 4), (17, 7),  None),
        Room((20, 3),  (5, 8), (20, 8),  None),
        Room((19, 10), (6, 6), (21, 10), None),
        Room((19, 15), (5, 8), (21, 15), None),
        Room((15, 16), (5, 6), (19, 18), None),
        Room((8, 14),  (8, 7), (15, 17), None),
        Room((4, 15),  (5, 8), (8, 17),  None),
        Room((2, 9),   (5, 7), (5, 15),  None),
        Room((1, 6),   (6, 4), (5, 9),   None),
    ],
    "MultiRoomN11S8_v2": [
        Room((5, 9),   (4, 5), (5, 9),   None),
        Room((8, 7),   (4, 5), (8, 10),  None),
        Room((7, 2),   (6, 6), (10, 7),  None),
        Room((12, 4),  (4, 4), (12, 5),  None),
        Room((15, 4),  (5, 5), (15, 5),  None),
        Room((19, 5),  (4, 5), (19, 6),  None),
        Room((19, 9),  (5, 7), (20, 9),  None),
        Room((19, 15), (5, 8), (21, 15), None),
        Room((16, 18), (4, 5), (19, 21), None),
        Room((12, 19), (5, 4), (16, 20), None),
        Room((8, 15),  (5, 8), (12, 20), None),
    ],
    "MultiRoomN11S8_v3": [
        Room((12, 4),  (4, 5), (12, 4),  None),
        Room((15, 4),  (6, 7), (15, 7),  None),
        Room((17, 10), (5, 4), (18, 10), None),
        Room((19, 13), (4, 6), (20, 13), None),
        Room((16, 16), (4, 4), (19, 17), None),
        Room((10, 16), (7, 8), (16, 18), None),
        Room((7, 18),  (4, 4), (10, 20), None),
        Room((3, 18),  (5, 4), (7, 20),  None),
        Room((0, 11),  (6, 8), (4, 18),  None),
        Room((1, 7),   (6, 5), (3, 11),  None),
        Room((6, 7),   (4, 6), (6, 8),   None),
    ],
    "MultiRoomN11S8_v4": [
        Room((18, 3),  (6, 5), (18, 3),  None),
        Room((16, 7),  (8, 6), (19, 7),  None),
        Room((14, 12), (7, 4), (19, 12), None),
        Room((17, 15), (4, 4), (19, 15), None),
        Room((17, 18), (4, 5), (19, 18), None),
        Room((12, 19), (6, 4), (17, 21), None),
        Room((6, 19),  (7, 4), (12, 21), None),
        Room((0, 15),  (7, 7), (6, 20),  None),
        Room((1, 11),  (4, 5), (2, 15),  None),
        Room((1, 5),   (4, 7), (3, 11),  None),
        Room((1, 1),   (4, 5), (3, 5),  None),
    ],
    "MultiRoomN11S8_v5": [
        Room((16, 15), (8, 8), (16, 15), None),
        Room((19, 10), (6, 6), (21, 15), None),
        Room((15, 9),  (5, 5), (19, 11), None),
        Room((12, 9),  (4, 4), (15, 10), None),
        Room((9, 10),  (4, 4), (12, 11), None),
        Room((5, 13),  (7, 5), (10, 13), None),
        Room((1, 10),  (5, 7), (5, 14),  None),
        Room((0, 5),   (4, 6), (2, 10),  None),
        Room((3, 1),   (5, 8), (3, 7),   None),
        Room((7, 2),   (8, 4), (7, 4),   None),
        Room((14, 1),  (6, 5), (14, 4),  None),
    ]
}

class StaticMultiRoomEnvHeatmap():
    """
    Environment with multiple rooms (subgoals)
    """
    def __init__(self,
                 numRooms,
                 maxRoomSize=10):

        assert maxRoomSize >= 4
        self.numRooms = numRooms

        self.minNumRooms = numRooms
        self.maxNumRooms = numRooms

        self.maxRoomSize = maxRoomSize

        self.rooms = ROOMS_LIST["MultiRoomN11S8_v5"]

        self._gen_grid(25, 25)

    def _gen_grid(self, width, height):

        roomList = self.rooms
        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColor = np.array([255, 255, 255], dtype=np.uint8)
                entryDoor = Door('yellow')
                self.grid.set(*room.entryDoorPos, entryDoor)


class HeatMapCell(object):
    def __init__(self, pos, color):
        self.position = pos
        self.color = color

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return self.position


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
                 path_to_checkpoints,
                 nr_runs=2,
                 argmax=False,
                 view_type="FullView",
                 max_steps=400,
                 index=None):

        self._env_name = env_name
        self._path_to_checkpoints = path_to_checkpoints
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nr_runs = nr_runs
        self._view_type = view_type
        self._argmax = argmax
        self._max_steps = max_steps
        self._index = index
        self._window = Window(self._env_name)
        self._make_env()
        _, self._obs_preprocess_fn = self.obs_preprocess()

        self._models = []
        self._agents_data = []

        for path in self._path_to_checkpoints:
            self.load_model(path)

        if self._models[0].recurrent:
            self._memories = torch.zeros(1, self._models[0].memory_size, device=self._device)

        self._step_count = 0

        self._visited_pos = {}
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
        obs_space, preprocess_obss = ut.get_obss_preprocessor(self._env_name,
                                                              self._eval_env.observation_space,
                                                              '.',
                                                              max_image_value=15.0,
                                                              normalize=True,
                                                              permute=False,
                                                              obs_type="compact",
                                                              type=None)
        return obs_space, preprocess_obss

    def load_model(self, path):
        print("Checkpoint is in {}".format(path))
        try:
            status = ut.load_status(path)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        saver = ut.SaveData(path, save_best=True, save_all=True)
        cur_model, cur_agent_data, cur_other_data = None, dict(), dict()
        try:
            # Continue from last point
            cur_model, cur_agent_data, cur_other_data = saver.load_training_data(best=False, index=self._index)
            print("Training data exists & loaded successfully\n")
        except OSError:
            print("Could not load training data\n")

        self._models.append(cur_model)
        self._agents_data.append(cur_agent_data)

    def run_episodes(self):
        print("Evaluating agent after for {} episodes".format(self._nr_runs))
        for model in self._models:
            self._memories = torch.zeros(1, model.memory_size, device=self._device)
            curr_runs = self._nr_runs // len(self._models)
            self.run_model(model, curr_runs)

    def run_model(self, model, nr_episodes):

        model.eval()
        for _ in tqdm.tqdm(range(nr_episodes)):
            obs = self._eval_env.reset()
            human_obs = obs["rendered_image"]
            episode_obs = [human_obs]

            while True:
                #  Record current agent's position
                ag_pos = tuple(self._eval_env.unwrapped.agent_pos)
                if ag_pos not in self._visited_pos:
                    self._visited_pos[ag_pos] = 0
                self._visited_pos[ag_pos] += 1

                # preprocess observation
                preprocessed_obs = self._obs_preprocess_fn([obs], device=self._device)

                with torch.no_grad():
                    if model.recurrent:
                        dist, _, self._memories = model(preprocessed_obs, self._memories)
                    else:
                        dis, _ = model(preprocessed_obs)

                if self._argmax:
                    _, action, = dist.probs.max(1, keepdim=True)
                else:
                    action = dist.sample()

                #print(action.cpu().numpy(), dist.probs)
                obs, reward, done, info = self._eval_env.step(action.cpu().numpy())
                # self._window.show_img(obs["rendered_image"])
                if done:
                    break
        model.train()

    def show_heatmap(self):
        self._visited_pos[(-1, -1)] = 0  # Non visited color

        nr_max_visited = float(max(self._visited_pos.values()))
        visited_frequency = list(np.unique(list(self._visited_pos.values())))

        # Make color-map
        no_colors = len(np.unique(list(self._visited_pos.values())))
        colors = cm.Reds(np.linspace(0, 1, no_colors))

        #  Remove A from RGBA and make them Integers
        np_colors = (np.delete(colors, -1, 1) * 255).astype(np.uint8)
        colors_map = {x: np_colors[visited_frequency.index(self._visited_pos[x])] for x in self._visited_pos}

        #  Create the colorbar for heatmap
        clrs = [colors[i] for i in range(0, no_colors, no_colors // 10)]
        labels = [visited_frequency[i] / self._nr_runs for i in range(0, no_colors, no_colors // 10)]
        cmap, norm = matplotlib.colors.from_levels_and_colors([0] + labels,clrs )
        ncolors = len(labels)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors + 0.5)
        colorbar = plt.colorbar(mappable,  orientation='vertical')
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels([f"{i:.1f}" for i in labels])

        env_heatmap = StaticMultiRoomEnvHeatmap(11, 8)
        env_heatmap_grid = env_heatmap.grid

        # Render the grid
        for j in range(0, env_heatmap_grid.height):
            for i in range(0, env_heatmap_grid.width):

                cell = env_heatmap_grid.get(i, j)
                if (i, j) in colors_map.keys():
                    env_heatmap_grid.set(i, j, HeatMapCell((i, j), colors_map[(i, j)]))
                elif cell is None:
                    env_heatmap_grid.set(i, j, HeatMapCell((i, j), colors_map[(-1, -1)]))


        img_heatmap = env_heatmap_grid.render(10)
        plt.imshow(img_heatmap)
        plt.show()
        #self._window.show_img(img_heatmap)
        #self._window.show(block=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parse = ArgumentParser()
    arg_parse.add_argument(
        '--env_name',
        default="MiniGrid-DoorKey16x16-v0"
    )
    arg_parse.add_argument(
        '--path_to_checkpoint',
        nargs='+',
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

    arg_parse.add_argument(
        '--model_index',
        default=None
    )

    args = arg_parse.parse_args()

    eval_agent = EvalAgent(args.env_name,
                           args.path_to_checkpoint,
                           nr_runs=args.nr_runs,
                           view_type=args.view_type,
                           max_steps=args.max_steps,
                           index=args.model_index)
    eval_agent.run_episodes()
    eval_agent.show_heatmap()

