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

try:
    import gym_minigrid

except ImportError:
    print("Can not import MiniGrid-ENV")
    exit(0)

from gym_minigrid.minigrid import *
from gym_minigrid.envs.multiroom import Room


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

        self.rooms = [
            Room((10, 1), (4, 4), (10, 1), None),
            Room((10, 4), (4, 4), (11, 4), None),
            Room((11, 7), (4, 4), (12, 7), None),
            Room((14, 6), (4, 4), (14, 8), None),
            Room((15, 3), (4, 4), (16, 6), None),
            Room((18, 4), (4, 4), (18, 5), None),
            Room((19, 7), (4, 4), (20, 7), None)]

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
                 path_to_checkpoint,
                 nr_runs=2,
                 argmax=False,
                 view_type="FullView",
                 max_steps=400,
                 index=None):

        self._env_name = env_name
        self._path_to_checkpoint = path_to_checkpoint
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nr_runs = nr_runs
        self._view_type = view_type
        self._argmax = argmax
        self._max_steps = max_steps
        self._index = int(index)
        self._window = Window(self._env_name)
        self._make_env()
        _, self._obs_preprocess_fn = self.obs_preprocess()
        self.load_model()

        if self._model.recurrent:
            self._memories = torch.zeros(1, self._model.memory_size, device=self._device)

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

    def load_model(self):
        print("Checkpoint is in {}".format(self._path_to_checkpoint))
        try:
            status = ut.load_status(self._path_to_checkpoint)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        saver = ut.SaveData(self._path_to_checkpoint, save_best=True, save_all=True)
        self._model, self._agent_data, self._other_data = None, dict(), dict()
        try:
            # Continue from last point
            self._model, self._agent_data, self._other_data = saver.load_training_data(best=False, index=self._index)
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

                #  Record current agent's position
                ag_pos = tuple(self._eval_env.unwrapped.agent_pos)
                if ag_pos not in self._visited_pos:
                    self._visited_pos[ag_pos] = 0
                self._visited_pos[ag_pos] += 1

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
                #self._window.show_img(obs["rendered_image"])
                if done:
                    break
        self._model.train()

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

        env_heatmap = StaticMultiRoomEnvHeatmap(7, 4)
        env_heatmap_grid = env_heatmap.grid

        # Render the grid
        for j in range(0, env_heatmap_grid.height):
            for i in range(0, env_heatmap_grid.width):

                cell = env_heatmap_grid.get(i, j)

                if (i, j) in colors_map.keys():
                    env_heatmap_grid.set(i, j, HeatMapCell((i, j), colors_map[(i, j)]))

        img_heatmap = env_heatmap_grid.render(10)
        self._window.show_img(img_heatmap)
        self._window.show(block=True)

        '''
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'
    '''




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
    eval_agent.run_episode()
    eval_agent.show_heatmap()

