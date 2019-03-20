#Dan Muntean 2019
#script for computing agent's beaviour during learning

import gym
import numpy as np
from copy import deepcopy
from gym import Wrapper
from optparse import OptionParser
import time
import math

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

        self.actions_taken.append((self.env.actions(action).value,
                                   self.env.actions(action).name))
        if type(self.env.agent_pos) == tuple:
            self.agent_pos.append(self.env.agent_pos)
        else:
            self.agent_pos.append(self.env.agent_pos.tolist())
        self.agent_orient.append(self.env.agent_dir)

        if done:
            self.full_states[self.env.step_count] = self.env.grid.encode()

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.actions_taken = []
        self.agent_pos = []
        self.agent_orient = []
        self.full_states = np.zeros((self.env.max_steps, self.env.height, self.env.width, 3))
        if type(self.env.start_pos) == tuple:
            self.agent_init_status = (self.env.start_pos, self.env.start_dir)
        else:
            self.agent_init_status = (self.env.start_pos.tolist(),
                                      self.env.start_dir)

        return self.env.reset(**kwargs)

    def get_behaviour(self):
        return {
            "initial_status": self.agent_init_status,
            "actions": deepcopy(self.actions_taken),
            "positions": deepcopy(self.agent_pos),
            "orientations": deepcopy(self.agent_orient),
            "full_states": deepcopy(self.full_states)
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


class ExploreActions(gym.core.Wrapper):
    """
        Store exploration data.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        counts = self.counts
        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action, env.carrying)

        obs, reward, done, info = self.env.step(action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in counts:
            preCnt = counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        counts[tup] = newCnt

        return obs, reward, done, info

    def get_next_action_cnt(self):
        env = self.unwrapped
        counts = self.counts

        n_act = env.action_space.n
        v = np.zeros(n_act)

        for action in range(env.action_space.n):
            tup = (tuple(env.agent_pos), env.agent_dir, action, env.carrying)
            if tup in counts:
                v[action] = counts[tup]
        return v

    def get_new_action_prob(self, temperature=1):
        v = self.get_next_action_cnt()
        v = (v.max() - v) ** temperature
        return v / v.sum()


class ActionBonus(gym.core.Wrapper):
    """
    source @ gym_minigrid repo
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
        tup = (env.agentPos, env.agentDir, action)

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
