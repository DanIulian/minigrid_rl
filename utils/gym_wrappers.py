# Dan Muntean 2019
# script for computing agent's beaviour during learning

import gym
import numpy as np
from copy import deepcopy
from gym import Wrapper
from optparse import OptionParser
import time
import math
import collections
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import cv2
from gym_minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper, \
    StateBonus, ActionBonus
import torch
from gym import error, spaces, utils

try:
    import gym_minigrid
except ImportError:
    pass


def state_bonus(env):
    return StateBonus(env)


def action_bonus(env):
    return ActionBonus(env)


def include_full_state(env):
    return RecordFullState(env)


def just_move(env):
    return JustMove(env)


def constant_reward(env):
    return ConstantReward(env)


def full_state_rgb_train(env):
    return RGBImgObsWrapper(env, tile_size=6)


def partial_rgb_train(env):
    return RGBImgPartialObsWrapper(env, tile_size=6)


def full_state_train(env):
    return FullyObsWrapper(env)


def full_state_train_dir(env):
    return FullyObsWrapperDirDiff(env)


def get_action_bonus_only(env):
    return ActionBonus(env, only_bonus=True)


def get_action_bonus(env):
    return ActionBonus(env, only_bonus=False)


def include_position(env):
    return RecordPosition(env)


def get_interactions(env):
    return GetImportantInteractions(env)


def add_env_id(env):
    return AddIDasText(env)


def occupancy_stats(env):
    return OccupancyMap(env)


def tensor_out(env):
    return TensorOut(env)


class FullyObsWrapperDirDiff(FullyObsWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def observation(self, obs):
        env = self.unwrapped

        obs = super().observation(obs)

        full_grid = obs["image"] + env.agent_dir

        return {
            'mission': obs['mission'],
            'image': full_grid
        }


class TensorOut(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        obs["image"] = torch.from_numpy(obs["image"])
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["image"] = torch.from_numpy(obs["image"])
        return obs

    def seed(self, seed=None):
        self.unwrapped.seed(seed=seed)


class ConstantReward(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env._reward = self._reward

    def _reward(self):
        return 1


class JustMove(gym.core.Wrapper):
    _move_actions = np.array([
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ])

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        env = self.unwrapped
        env.step_count += 1

        reward = 0
        done = False

        # Move action
        if 0 <= action < 4:
            # Get the new possible position for the action
            fwd_pos = self.agent_pos + self._move_actions[action]

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell == None or fwd_cell.can_overlap():
                env.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = env._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(*fwd_pos, None)
        #
        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(*fwd_pos, self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None
        #
        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)
        #
        # # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass
        #
        # else:
        #     assert False, "unknown action"

        if env.step_count >= env.max_steps:
            done = True

        obs = env.gen_obs()
        return obs, reward, done, {}


def rotate_img(img, cw=True):
    if cw:
        # rotate cw
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=1)
    else:
        # rotate ccw
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=0)
    return out


class RecordingBehaviour(Wrapper):
    '''
    When finished collecting information call get_behaviour()
    The function returns a dictionary with the following fields:
        -initial_status - initial information about the agent's position
        -actions - the actions taken by the agent during episode
        -positions - the position of the agent at each time-step
        -orientations - the orientation of the agent at each time-step
        -full_states - (MAX_NR_STEPS, HEIGHT, WIDTH, 3) np array with all info about the grid
    '''
    def __init__(self, env):
        super(RecordingBehaviour, self).__init__(env)

        self.actions_taken = None
        self.agent_pos = None
        self.agent_orient = None
        self.agent_init_status = None
        self.full_states = None

    def step(self, action):
        #get the full observation
        self.full_states[self.env.step_count] = self.env.grid.encode()

        observation, reward, done, info = self.env.step(action)

        self.actions_taken[self.env.step_count] = np.array(self.env.actions(action).value)
        self.agent_pos[self.env.step_count] = np.array(self.env.agent_pos)
        self.agent_orient[self.env.step_count] = np.array(self.env.agent_dir)

        if done:
            self.full_states[self.env.step_count] = self.env.grid.encode()

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.actions_taken = np.zeros(self.env.max_steps + 1)
        self.agent_pos = np.zeros((self.env.max_steps + 1, 2))
        self.agent_orient = np.zeros(self.env.max_steps + 1)
        self.full_states = np.zeros((self.env.max_steps + 1, self.env.width, self.env.height, 3))

        self.agent_pos[self.env.step_count] = np.array(self.env.agent_pos)
        self.agent_orient[self.env.step_count] = np.array(self.env.start_dir)

        return self.env.reset(**kwargs)

    def get_behaviour(self):
        return {
            "actions": deepcopy(self.actions_taken),
            "positions": deepcopy(self.agent_pos[:, [1, 0]]),
            "orientations": deepcopy(self.agent_orient),
            "full_states": deepcopy(self.full_states),
            "step_count": self.env.step_count,
        }


class RecordFullState(Wrapper):
    def __init__(self, env):
        super(RecordFullState, self).__init__(env)
        self._step = np.random.randint(155)

    def step(self, action):
        env = self.unwrapped
        self._step += 1

        observation, reward, done, info = env.step(action)

        # observation["image"] = observation["image"].astype(np.int)
        # observation["image"].fill((action + 1)*2)

        observation["state"] = self.get_full_state()

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._step = np.random.randint(155)
        env = self.unwrapped

        obs = env.reset(**kwargs)

        # obs["image"] = obs["image"].astype(np.int)
        # obs["image"].fill(self._step)

        obs["state"] = self.get_full_state()

        return obs

    def get_full_state(self):
        env = self.unwrapped
        full_grid = env.grid.encode()

        carying = 0 if env.carrying is None else OBJECT_TO_IDX[env.carrying.type]

        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [15, env.agent_dir, carying])
        # full_grid = full_grid.transpose(0, 1, 2)
        return full_grid

    def seed(self, seed=None):
        self.unwrapped.seed(seed=seed)


class UniqueStates(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, env.carrying.type, env.carrying.color)

        # Get the count for this (s,a) pair
        if tup in self.counts:
            self.counts[tup] += 1
        else:
            self.counts[tup] += 1

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.counts = {}
        return self.env.reset(**kwargs)


def rotate_img(img, cw=True):
    if cw:
        # rotate cw
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=1)
    else:
        # rotate ccw
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=0)
    return out


class OccupancyMap(Wrapper):
    """ Builds discovery map """
    def __init__(self, env):
        super(OccupancyMap, self).__init__(env)

        env_name = env.spec.id
        self.actions = self.env.unwrapped.actions
        self.step_count = self.env.unwrapped.step_count
        self.counts = {}

        if "MultiRoom" in env_name:
            self.get_occupancy = self.get_irregular_map
        elif "ObstructedMaze-2D" in env_name or "ObstructedMaze-1Q" in env_name:
            self.get_occupancy = self.get_obstructedmaze_3room
        elif "ObstructedMaze-2Q" in env_name:
            self.get_occupancy = self.get_obstructedmaze_6room
        else:
            self.get_occupancy = self.get_full_occupancy

        self.occupancy = None
        self.seen = None

    def step(self, action):
        env = self.unwrapped
        observation, reward, done, info = self.env.step(action)

        # Count unique states
        if not env.carrying:
            carrying_type = None
            carrying_color = None
        else:
            carrying_type = env.carrying.type
            carrying_color = env.carrying.color

        tup = (tuple(env.agent_pos), env.agent_dir, carrying_type, carrying_color)

        if tup in self.counts:
            self.counts[tup] += 1
        else:
            self.counts[tup] = 1

        # Add discovery map
        self.add_view(observation["image"])

        if done:
            possible = self.occupancy.sum()
            seen = ((self.occupancy * self.seen) > 0).sum()
            values = np.array(list(self.counts.values()))
            info['occupancy'] = {
                "possible": possible,
                "seen": seen,
                "discovered": seen/possible,
                "unique_states": len(self.counts.keys()),
                "same_state_max": values.max(),
                "same_state_mean": values.mean(),
                "same_state_std": values.std(),
            }

        return observation, reward, done, info

    def add_view(self, view):
        topX, topY, botX, botY = self.env.unwrapped.get_view_exts()
        maxX, maxY = self.occupancy.shape
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid.encode()[:,:,0]
        view = view[:, :, 0]

        # Rotate grid to map view
        if agent_dir == 0:
            view = rotate_img(view, cw=False)
        elif agent_dir == 1:
            view = rotate_img(view, cw=True)
            view = rotate_img(view, cw=True)
        elif agent_dir == 2:
            view = rotate_img(view, cw=True)

        # Select only visible grid
        d1, d2 = maxX-botX, maxY-botY
        maxX = maxX+1 if d1 >= 0 else d1
        maxY = maxY+1 if d2 >= 0 else d2
        visible_view = view[min(topX, 0)*-1: maxX, min(topY, 0)*-1: maxY]
        visible_grid = grid[max(topX,0):botX, max(topY,0):botY]

        self.seen[max(topX, 0):botX, max(topY, 0):botY] += (visible_grid == visible_view)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.occupancy = self.get_occupancy(self.env.unwrapped.grid.encode())
        self.seen = np.zeros_like(self.occupancy)
        self.add_view(obs["image"])
        self.counts = {}
        return obs

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def get_full_occupancy(self, full_grid: np.array) -> np.array:
        occupancy = np.ones_like(full_grid[:, :, 0])
        return occupancy

    def get_obstructedmaze_3room(self, full_grid: np.array) -> np.array:
        room_size = self.env.unwrapped.room_size
        occupancy = np.zeros_like(full_grid[:, :, 0])
        occupancy[:, -room_size:] = 1
        occupancy[room_size-1:room_size*2, room_size-1:] = 1
        return occupancy

    def get_obstructedmaze_6room(self, full_grid: np.array) -> np.array:
        room_size = self.env.unwrapped.room_size
        occupancy = np.zeros_like(full_grid[:, :, 0])
        occupancy[:, -room_size:] = 1
        occupancy[room_size-1:room_size*2, room_size-1:] = 1
        occupancy[-room_size:, :] = 1
        return occupancy

    @staticmethod
    def get_irregular_map(full_grid: np.array) -> np.array:
        """Get occupancy map for non-regular maze"""
        full_grid = full_grid[:, :, 0]

        walls_idx = np.array(np.where(full_grid == 2)).T
        occupancy1 = np.zeros_like(full_grid)
        row = -1
        min_c = None
        for i in range(walls_idx.shape[0]):
            if row != walls_idx[i, 0]:
                if row > 0:
                    occupancy1[row, min_c:walls_idx[i - 1, 1]] = 1
                row, min_c = walls_idx[i]

        walls_idx = np.array(np.where(full_grid.transpose(1, 0) == 2)).T
        occupancy2 = np.zeros_like(full_grid)
        cl = -1
        min_r = None
        for i in range(walls_idx.shape[0]):
            if cl != walls_idx[i, 0]:
                if cl > 0:
                    occupancy2[min_r:walls_idx[i - 1, 1], cl] = 1
                cl, min_r = walls_idx[i]

        occupancy = occupancy1 & occupancy2
        return occupancy


def bfs(grid, start, target):
    move = np.array([[+1, 0], [-1, 0], [0, +1], [0, -1]])
    height, width = grid.shape

    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == target[0] and x == target[1]:
            return path

        new = move + np.array([x, y])
        test = new[np.argsort(np.linalg.norm(new - target, axis=1))]  # Sort by distance to target
        for x2, y2 in test:
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != 1 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return None


class RecordPosition(Wrapper):
    """
    Embed into the observation the position of the agent
    on the map
    """
    def __init__(self, env):
        super(RecordPosition, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["position"] = np.array(self.env.unwrapped.agent_pos)

        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["position"] = np.array(self.env.unwrapped.agent_pos)

        return obs

    def seed(self, seed=None):
        self.env.seed(seed=seed)


class AddIDasText(Wrapper):
    def __init__(self, env):
        super(AddIDasText, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["mission"] = self.env.unwrapped._env_proc_id

        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["mission"] = self.env.unwrapped._env_proc_id

        return obs

    def seed(self, seed=None):
        self.env.seed(seed=seed)


def get_interactions_stats(ep_statistics):

    # process statistics about the agent's behaviour
    # in the environment
    logs = {
        "ep_completed": len(ep_statistics),

        "doors_opened": 0,
        "doors_closed": 0,
        "doors_interactions": 0,

        "keys_picked": 0,
        "keys_dropped": 0,
        "keys_interactions": 0,

        "balls_picked": 0,
        "balls_dropped": 0,
        "balls_interactions": 0,

        "boxes_picked": 0,
        "boxes_dropped": 0,
        "boxes_broken": 0,
        "boxes_interactions": 0,

        "objects_interactions": 0,
        "categories_interactions": 0,

        "possible": 0,
        "seen": 0,
        "discovered": 0,
        "unique_states": 0,
        "same_state_max": 0,
        "same_state_mean": 0,
        "same_state_std": 0,
    }

    if len(ep_statistics) == 0:
        return logs

    for ep_info in ep_statistics:
        if "occupancy" in ep_info:
            for k, v in ep_info["occupancy"].items():
                logs[k] += v

        ep_info = ep_info['interactions']

        global no_categories, no_objects, cat_interactions, objects_interactions
        no_categories = 0.
        no_objects = 0.
        cat_interactions = 0.
        objects_interactions = 0.

        def stats_interactions(objects, interactions):
            global no_categories, no_objects, cat_interactions, objects_interactions
            no_categories += (len(objects) > 0)
            no_objects += len(objects)
            cat_interactions += (interactions > 0)
            objects_interactions += interactions

        # count the mean number of interactions with doors
        doors_interactions, doors_opened, doors_closed = 0., 0., 0.
        for door in ep_info['doors'].values():
            doors_opened += door['nr_opened']
            doors_closed += door['nr_closed']
            if door['nr_opened'] > 0:
                doors_interactions += 1

        if len(ep_info['doors']) > 0 and doors_interactions > 0:
            logs['doors_interactions'] +=\
                float(doors_interactions) / len(ep_info['doors'])
            logs['doors_opened'] += doors_opened / doors_interactions
            logs['doors_closed'] += doors_closed / doors_interactions

        stats_interactions(ep_info['doors'], doors_interactions)

        # count the mean number of interactions with boxes
        boxes_interactions, boxes_picked, boxes_dropped, boxes_broken = 0., 0., 0., 0.
        for box in ep_info['boxes'].values():
            boxes_picked += box["nr_picked_up"]
            boxes_dropped += box["nr_put_down"]
            boxes_broken += box['broken']
            if box['nr_picked_up'] > 0 or box['broken'] > 0:
                boxes_interactions += 1

        if len(ep_info['boxes']) > 0 and boxes_interactions > 0:
            logs['boxes_interactions'] +=\
                float(boxes_interactions) / len(ep_info['boxes'])
            logs['boxes_picked'] += boxes_picked / boxes_interactions
            logs['boxes_dropped'] += boxes_dropped / boxes_interactions
            logs['boxes_broken'] += boxes_broken / boxes_interactions

        stats_interactions(ep_info['boxes'], boxes_interactions)

        # count the mean number of interactions with balls
        balls_interactions, balls_picked, balls_dropped = 0., 0., 0.
        for ball in ep_info['balls'].values():
            balls_picked += ball["nr_picked_up"]
            balls_dropped += ball["nr_put_down"]
            if ball['nr_picked_up'] > 0:
                balls_interactions += 1

        if len(ep_info['balls']) > 0 and balls_interactions > 0:
            logs['balls_interactions'] +=\
                float(balls_interactions) / len(ep_info['balls'])
            logs['balls_picked'] += balls_picked / balls_interactions
            logs['balls_dropped'] += balls_dropped / balls_interactions

        stats_interactions(ep_info['balls'], balls_interactions)

        # count the mean number of interactions with keys
        keys_interactions, keys_picked, keys_dropped = 0., 0., 0.
        for key in ep_info['keys'].values():
            keys_picked += key["nr_picked_up"]
            keys_dropped += key["nr_put_down"]
            if key['nr_picked_up'] > 0:
                keys_interactions += 1

        if len(ep_info['keys']) > 0 and keys_interactions > 0:
            logs['keys_interactions'] +=\
                float(keys_interactions) / len(ep_info['keys'])
            logs['keys_picked'] += keys_picked / keys_interactions
            logs['keys_dropped'] += keys_dropped / keys_interactions

        stats_interactions(ep_info['keys'], keys_interactions)

        # Calculate the total number of objects
        reached_goal = ep_info['reward']
        logs['objects_interactions'] += (objects_interactions + reached_goal) / (no_objects + 1)
        logs['categories_interactions'] += (cat_interactions + reached_goal) / (no_categories + 1)

    for log_key in logs:
        if log_key != 'ep_completed':
            logs[log_key] /= float(logs['ep_completed'])

    return logs


class GetImportantInteractions(Wrapper):
    '''
    Wrapper that saves important interactions with the environment
    It counts the number of pick-ups and put-downs for each object
    (keys, balls, boxes), and how many times a door is opened
    '''

    def __init__(self, env):
        super(GetImportantInteractions, self).__init__(env)

        self.actions = self.env.unwrapped.actions
        self.step_count = self.env.unwrapped.step_count

        self.grid = None
        self.doors = {}
        self.objects = {
            "ball": {},
            "box": {},
            "key": {}
        }
        self.carrying = False
        self.directions = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }

    def step(self, action):

        if action == 5:
            self.check_broken_box()
        observation, reward, done, info = self.env.step(action)

        self.check_doors()
        self.check_objects()

        if done:
            info['interactions'] = {
                "doors": deepcopy(self.doors),
                "keys": deepcopy(self.objects['key']),
                "boxes": deepcopy(self.objects['box']),
                "balls": deepcopy(self.objects['ball']),
                "reward": reward > 0,
            }

        return observation, reward, done, info

    def check_doors(self):
        '''
        Check if a door was opened or closed and log the interaction
        '''
        for door_pos in self.doors:
            door = self.grid.get(door_pos[0], door_pos[1])
            if self.doors[door_pos]["is_opened"] != door.is_open:
                self.doors[door_pos]["is_opened"] = door.is_open
                if door.is_open:
                    self.doors[door_pos]["nr_opened"] += 1
                else:
                    self.doors[door_pos]["nr_closed"] += 1

    def check_objects(self):
        '''
        Check if the agent picked up or putted down an object and
        log the interaction
        '''
        if (not self.carrying) and self.env.unwrapped.carrying:
            self.carrying = True
            obj = self.env.unwrapped.carrying
            obj_pos = tuple(obj.init_pos)
            self.objects[obj.type][obj_pos]["nr_picked_up"] += 1

        elif self.carrying and (self.env.unwrapped.carrying is None):
            self.carrying = False
            for obj_type in self.objects:
                for obj_pos in self.objects[obj_type]:
                    if self.objects[obj_type][obj_pos]["nr_picked_up"] >\
                            self.objects[obj_type][obj_pos]["nr_put_down"]:
                        self.objects[obj_type][obj_pos]["nr_put_down"] += 1

    def check_broken_box(self):
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.directions[self.env.unwrapped.agent_dir]
        front_pos = (agent_pos[0] + agent_dir[0], agent_pos[1] + agent_dir[1])
        front_obj = self.grid.get(front_pos[0], front_pos[1])
        if front_obj!= None and front_obj.type == 'box':
            obj_pos = tuple(front_obj.init_pos)
            self.objects['box'][obj_pos]['broken'] = 1


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.doors = {}
        self.objects = {
            "ball": {},
            "box": {},
            "key": {}
        }

        self.carrying = False
        self.grid = self.env.unwrapped.grid

        # get all objects from the grid
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                v = self.grid.get(i, j)

                if v:
                    if v.type == 'door':
                        self.doors[(i, j)] = {
                            "is_opened": v.is_open,
                            "nr_opened": 0,
                            "nr_closed": 0
                        }
                    elif (v.type == 'key')  or (v.type == 'ball'):
                        self.objects[v.type][(i, j)] = {
                            "color": v.color,
                            "nr_picked_up": 0,
                            "nr_put_down": 0
                        }

                        # if initial position for object is not present, add it
                        if v.init_pos is None:
                            v.init_pos = np.array([i, j])

                    elif v.type == 'box':
                        self.objects[v.type][(i, j)] = {
                            "color": v.color,
                            "nr_picked_up": 0,
                            "nr_put_down": 0,
                            "broken": 0,
                        }
                        # check if there is a key inside the box
                        if v.contains:
                            in_obj = v.contains
                            self.objects[in_obj.type][(i, j)] = {
                                "color": in_obj.color,
                                "nr_picked_up": 0,
                                "nr_put_down": 0
                            }
                            if in_obj.init_pos is None:
                                in_obj.init_pos = np.array([i, j])

                        # if initial position for object is not present, add it
                        if v.init_pos is None:
                            v.init_pos = np.array([i, j])
        return obs

    def seed(self, seed=None):
        self.env.seed(seed=seed)


class ExploreActions(gym.core.Wrapper):
    """
        Store exploration data.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}
        self.memory_len = 10

        # Keep a list of min state, action options found
        self.idxs_counts = np.zeros(self.memory_len)
        self.idxs_counts.fill(9999)  # Fill with big number
        self.min_count_idx = -1

        self.idx = [None] * self.memory_len
        self.max_min_count = 0
        self.max_action = self.unwrapped.action_space.n

        parent_class = env

    def step(self, action):
        self.add_state_count(action)

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def add_state_count(self, action):
        counts = self.counts
        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, env.carrying)

        # Update the count for this (s, a) pair
        counts = None
        if tup in counts:
            counts = counts[tup]
        else:
            counts[tup] = counts = np.zeros(self.max_action)

        counts[action] += 1

        # Update list of min state, action options
        min_count = counts.min()
        if self.max_min_count > min_count:
            idxs_counts = self.idxs_counts
            change = np.argmax(self.idxs_counts)
            idxs_counts[change] = min_count
            self.idx[change] = tup
            self.max_min_count = np.max(idxs_counts)
            self.min_count_idx = np.argmin(idxs_counts)

    def get_to_state(self, origin, target):
        origin_pos, origin_dir, orgin_carrying = origin
        target_pos, target_dir, target_carrying = origin

        if target_carrying != orgin_carrying:
            # Go to item
            target_pos = None

        if origin_pos == target_pos:
            if origin_dir != target_dir:
                return None # get action for rotation:
            else:
                return np.random.randint(self.max_action)  # You are here

        # Get to coord
        grid = None  # TODO generate occupancy grid (1 - obstacle. 0 - free)

        path = bfs(grid, origin_pos, target_pos)

    def min_state_explore(self):
        """ Get action that gets the agent to a min count state """
        env = self.unwrapped
        counts = self.counts

        crt = (tuple(env.agent_pos), env.agent_dir, env.carrying)
        if crt in counts:
            min_crt = counts[crt].min()
            min_count_idx = self.min_count_idx
            if min_crt <= self.idxs_counts[min_count_idx]:
                p = counts[crt]  # Return this state's count
            else:
                # distribution over actions that get the agent to min state count option
                min_state = self.idx[min_count_idx]
                pass
        else:
            p = np.ones(self.max_action)

        return p / p.sum()

    def get_next_action_cnt(self):
        env = self.unwrapped
        counts = self.counts
        n_act = self.max_action

        tup = (tuple(env.agent_pos), env.agent_dir, env.carrying)
        if tup in counts:
            v = counts[tup]
        else:
            v = np.zeros(n_act)

        return v

    def get_new_action_prob(self, temperature=1):
        v = self.get_next_action_cnt()
        v = (v.max() - v) ** temperature
        return v / v.sum()


class ActionBonus(gym.core.Wrapper):
    """
    source @ gym_minigrid repo
    # added carrying info
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env, only_bonus=False):
        super().__init__(env)
        self.counts = {}
        self.only_bonus = only_bonus

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, env.carrying.type, env.carrying.color,  action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        if self.only_bonus:
            reward = 0

        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.counts = {}
        return self.env.reset(**kwargs)


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-DoorKey-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
    env = RecordPosition(env)
    env = GetImportantInteractions(env)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.unwrapped.actions.left
        elif keyName == 'RIGHT':
            action = env.unwrapped.actions.right
        elif keyName == 'UP':
            action = env.unwrapped.actions.forward

        elif keyName == 'SPACE':
            action = env.unwrapped.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.unwrapped.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.unwrapped.actions.drop

        elif keyName == 'RETURN':
            action = env.unwrapped.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)
        print(env.doors)
        print(env.objects)
        print(env.unwrapped.agent_pos)

        print('step=%s, reward=%.2f' % (env.unwrapped.step_count, reward))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break


if __name__ == "__main__":
    main()
