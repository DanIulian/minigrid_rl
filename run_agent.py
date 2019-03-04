# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys
from liftoff.config import read_config
from argparse import Namespace

try:
    import gym_minigrid
except ImportError:
    pass

import utils
from models import get_model
from agents import get_agent


def post_process_args(args: NameError) -> None:
    args.mem = args.recurrence > 1


def run(full_args: Namespace) -> None:

    args = full_args.main
    agent_args = full_args.agent
    model_args = full_args.model

    if args.seed == 0:
        args.seed = full_args.run_id + 1
    max_eprews = args.max_eprews

    post_process_args(agent_args)
    post_process_args(model_args)

    model_dir = full_args.cfg_dir
    print(model_dir)

    # ==============================================================================================
    # Set seed for all randomness sources
    utils.seed(args.seed)

    # ==============================================================================================
    # Generate environment

    env = gym.make(args.env)
    env.max_steps = full_args.env_cfg.max_episode_steps
    env.seed(args.seed + 10000 * 0)


    # Define obss preprocessor
    max_image_value = full_args.env_cfg.max_image_value
    normalize_img = full_args.env_cfg.normalize
    obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, env.observation_space,
                                                             model_dir,
                                                             max_image_value=max_image_value,
                                                             normalize=normalize_img)
    # ==============================================================================================
    # Load training status
    try:
        status = utils.load_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}

    saver = utils.SaveData(model_dir, save_best=args.save_best, save_all=args.save_all)
    model, agent_data, other_data = None, dict(), None
    try:
        # Continue from last point
        model, agent_data, other_data = saver.load_training_data(best=False)
        print("Training data exists & loaded successfully\n")
    except OSError:
        print("Could not load training data\n")

    if torch.cuda.is_available():
        model.cuda()
        device = torch.device("cuda")
    else:
        model.cpu()
        device = torch.device("cpu")

    # ==============================================================================================
    # Test model

    done = True
    model.eval()

    while True:
        if done:
            obs = env.reset()
            memory = torch.zeros(1, model.policy_model.memory_size, device=device)

        time.sleep(0.1)
        renderer = env.render()

        preprocessed_obs = preprocess_obss([obs], device=device)
        if model.recurrent:
            dist, _, memory = model.policy_model(preprocessed_obs, memory)
        else:
            dist, value = model.policy_model(preprocessed_obs)


        action = dist.sample()
        obs, reward, done, _ = env.step(action.cpu().numpy())
        if renderer.window is None:
            break


def main() -> None:
    import os
    """ Read configuration from disk (the old way)"""
    # Reading args
    full_args = read_config()  # type: Args
    args = full_args.main

    if not hasattr(full_args, "run_id"):
        full_args.run_id = 0

    run(full_args)


if __name__ == "__main__":
    main()
