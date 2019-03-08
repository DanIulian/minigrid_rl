#Dan Muntean 2019
#script for computing agent's beaviour during learning

import gym
import numpy as np
from copy import deepcopy
from gym import Wrapper
try:
    import gym_minigrid
except ImportError:
    pass


def include_full_state(env):
    return RecordFullState(env)


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

    def step(self, action):

        observation, reward, done, info = self.env.step(action)

        full_states = self.env.grid.encode().transpose(1, 0, 2)
        observation["state"] = full_states

        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        full_states = self.env.grid.encode().transpose(1, 0, 2)
        obs["state"] = full_states

        return obs

    def seed(self, seed=None):
        self.env.seed(seed=seed)

from optparse import OptionParser
import time
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
