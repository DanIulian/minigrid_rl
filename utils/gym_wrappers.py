#Dan Muntean 2019
#script for computing agent's beaviour during learning

import gym
import numpy as np
from copy import deepcopy
from gym import Wrapper
from optparse import OptionParser
import time
import math
import collections

try:
    import gym_minigrid
except ImportError:
    pass


def include_full_state(env):
    return RecordFullState(env)


def get_action_bonus_only(env):
    return ActionBonus(env, only_bonus=True)


def get_action_bonus(env):
    return ActionBonus(env, only_bonus=True)


def include_position(env):
    return RecordPosition(env)


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

        self.agent_pos[self.env.step_count] = np.array(self.env.start_pos)
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

        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [15, env.agent_dir, 0])
        full_grid = full_grid.transpose(1, 0, 2)
        return full_grid

    def seed(self, seed=None):
        self.unwrapped.seed(seed=seed)


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
    def __init__(self, env):
        super(RecordPosition, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["position"] = np.array(self.env.agent_pos)

        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["position"] = np.array(self.env.start_pos)

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
        tup = (env.agentPos, env.agentDir, env.carrying, action)

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


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
    env = RecordingBehaviour(env)

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


import cv2


img = cv2.imread("")