#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
from utils.gym_wrappers import OccupancyMap,  GetImportantInteractions, get_interactions_stats


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
    env.max_steps = 30
    env = (env)
    env = GetImportantInteractions(env)
    env = OccupancyMap(env)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    # Render partial obs
    from gym_minigrid.rendering import Renderer
    from gym_minigrid.minigrid import CELL_PIXELS

    env.unwrapped.obs_render = Renderer(
        env.unwrapped.agent_view_size * CELL_PIXELS // 2,
        env.unwrapped.agent_view_size * CELL_PIXELS // 2,
        True
    )

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)
        img = env.unwrapped.get_obs_render(obs["image"])

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print(info)
            print(get_interactions_stats([info]))
            print(env.seen)
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

    env = gym.make("MiniGrid-ObstructedMaze-2Q-v0")

    obs = env.reset()["image"]
    full_grid = env.unwrapped.grid.encode()
    topX, topY, botX, botY = env.unwrapped.get_view_exts()

    cv2.imshow("full_grid", full_grid*10)
    cv2.waitKey(1)
    cv2.imshow("obs", obs*10)
    cv2.waitKey(1)
    obs = env.step(0)["image"]
    print(env.occupancy)
    print(env.seen)



