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

    model_dir = getattr(args, "model_dir", full_args.out_dir)
    print(model_dir)

    # ==============================================================================================
    # @ torc_rl repo original

    # Define logger, CSV writer and Tensorboard writer

    logger = utils.get_logger(model_dir)
    csv_file, csv_writer = utils.get_csv_writer(model_dir)
    if args.tb:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(model_dir)

    # Log command and all script arguments

    logger.info("{}\n".format(" ".join(sys.argv)))
    logger.info("{}\n".format(args))

    # ==============================================================================================
    # Set seed for all randomness sources
    utils.seed(args.seed)

    # ==============================================================================================
    # Generate environments

    envs = []

    actual_procs =  getattr(args, "actual_procs", None)
    if actual_procs:
        # Split envs in chunks
        env = gym.make(args.env)
        env.max_steps = full_args.env_cfg.max_episode_steps

        env.seed(args.seed + 10000 * 0)
        envs.append([env])
        for env_i in range(1, args.procs, actual_procs):
            env_chunk = []
            for i in range(env_i, min(env_i+actual_procs, args.procs)):
                env = gym.make(args.env)
                env.seed(args.seed + 10000 * i)
                env_chunk.append(env)
            envs.append(env_chunk)
        first_env = envs[0][0]
    else:
        for i in range(args.procs):
            env = gym.make(args.env)
            env.max_steps = full_args.env_cfg.max_episode_steps

            env.seed(args.seed + 10000 * i)
            envs.append(env)
        first_env = envs[0]

    # Define obss preprocessor
    max_image_value = full_args.env_cfg.max_image_value
    normalize_img = full_args.env_cfg.normalize
    obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, first_env.observation_space,
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
        logger.info("Training data exists & loaded successfully\n")
    except OSError:
        logger.info("Could not load training data\n")

    # ==============================================================================================
    # Load Model

    if model is None:
        model = get_model(model_args, obs_space, first_env.action_space,
                          use_memory=model_args.mem, use_text=model_args.text)
        logger.info(f"Model [{model_args.name}] successfully created\n")

        # Print Model info
        logger.info("{}\n".format(model))

    if torch.cuda.is_available():
        model.cuda()
    logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

    # ==============================================================================================
    # Load Agent

    algo = get_agent(full_args.agent, envs, model, agent_data,
                     preprocess_obss=preprocess_obss, reshape_reward=None)

    # ==============================================================================================
    # Train model

    crt_eprew = 0
    if "eprew" in other_data:
        crt_eprew = other_data["eprew"]
    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]

    algo.collect_random_statistics(50)

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - total_start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"]]
            data += [logs["grad_norm"], logs["value_ext"], logs["value_int"], logs["value_ext_loss"], logs["value_int_loss"]]

            logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | "
                "F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | "
                "∇ {:.3f} | VE {:.3f} | VI {:.3f} | veL {:.3f} | viL {:.3f} ".format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)
            csv_file.flush()

            if args.tb:
                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            status = {"num_frames": num_frames, "update": update}

            crt_eprew = list(rreturn_per_episode.values())[0]

        # -- Save vocabulary and model

        if args.save_interval > 0 and update % args.save_interval == 0:
            preprocess_obss.vocab.save()

            saver.save_training_data(model, algo.get_save_data(), crt_eprew)

            logger.info("Model successfully saved")

            utils.save_status(status, model_dir)

        if crt_eprew > max_eprews:
            print("Reached max return 0.9")
            exit()


def main() -> None:
    import os

    """ Read configuration from disk (the old way)"""
    # Reading args
    full_args = read_config()  # type: Args
    args = full_args.main

    if not hasattr(full_args, "run_id"):
        full_args.run_id = 0

    if hasattr(args, "model_dir"):
        # Define run dir
        os.environ["TORCH_RL_STORAGE"] = "results_dir"

        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        default_model_name = "{}_{}_seed{}_{}".format(args.env, args.algo, args.seed, suffix)
        model_name = args.model or default_model_name
        model_dir = utils.get_model_dir(model_name)

        full_args.out_dir = model_dir
        args.model_dir = model_dir

    run(full_args)


if __name__ == "__main__":
    main()
