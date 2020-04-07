# AndreiN, 2019
# parts from https://github.com/lcswillems/torch-rl

import gym
import time
import datetime
import torch
import sys
# from liftoff.config import read_config
from argparse import Namespace
import numpy as np
from typing import List


try:
    import gym_minigrid
except ImportError:
    pass

import utils
from models import get_model
from agents import get_agent
from utils import gym_wrappers

MAIN_CFG_ARGS = ["main", "env_cfg", "agent", "model"]


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
    for arg in subgroups:
        if hasattr(cfg, arg):
            setattr(getattr(cfg, arg), new_arg, new_arg_value)


def post_process_args(args: Namespace) -> None:
    args.mem = args.recurrence > args.min_mem


def extra_log_fields(header: list, log_keys: list) ->list:
    unusable_fields = ['return_per_episode', 'reshaped_return_per_episode',
                       'num_frames_per_episode', 'num_frames']
    extra_fields = []
    for field in log_keys:
        if field not in header and field not in unusable_fields:
            extra_fields.append(field)

    return extra_fields


def get_envs(full_args, env_wrapper, no_envs, n_actions=6):
    """ Minigrid action 6 is Done - useless"""
    envs = []
    args = full_args.main
    actual_procs = args.actual_procs
    add_to_cfg(full_args, MAIN_CFG_ARGS, "out_dir", full_args.out_dir)

    env_args = getattr(full_args.env_cfg, "env_args", None)
    env_args = dict() if env_args is None else vars(env_args)
    print(env_args)

    env = gym.make(args.env, **env_args)
    env.action_space.n = n_actions
    env.max_steps = full_args.env_cfg.max_episode_steps
    env.unwrapped._env_proc_id = 0
    env = env_wrapper(env)
    env.seed(args.seed + 10000 * 0)

    envs.append([env])
    chunk_size = int(np.ceil((no_envs - 1) / float(actual_procs)))
    for env_i in range(1, no_envs, chunk_size):
        env_chunk = []
        for i in range(env_i, min(env_i + chunk_size, no_envs)):
            env = gym.make(args.env, **env_args)
            env.action_space.n = n_actions
            env.max_steps = full_args.env_cfg.max_episode_steps
            env.unwrapped._env_proc_id = i
            env = env_wrapper(env)
            env.seed(args.seed + 10000 * i)

            env_chunk.append(env)
        envs.append(env_chunk)

    return envs, chunk_size


def print_keys(header: list, data: list, extra_logs: list = None) ->tuple:
    basic_keys_format = \
        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | " \
        "F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | "\
        "∇ {:.3f}"
    printable_data = data[:17]

    if extra_logs:
        for field in extra_logs:
            if field[0] in header:
                basic_keys_format += (" | " + field[1] + " {:." + field[2] + "} ")
                printable_data.append(data[header.index(field[0])])

    return basic_keys_format, printable_data


def run(full_args: Namespace, return_models: bool = False):
    if sys.argv[0].startswith("train"):
        import os
        full_args.out_dir = os.path.dirname(sys.argv[1])

    args = full_args.main
    agent_args = full_args.agent
    model_args = full_args.model
    extra_logs = getattr(full_args, "extra_logs", None)
    main_r_key = getattr(full_args, "main_r_key", None)

    if args.seed == 0:
        args.seed = full_args.run_id + 1
    max_eprews = args.max_eprews
    max_eprews_window = getattr(args, "max_eprews_window", 1)

    post_process_args(agent_args)
    post_process_args(model_args)

    model_dir = getattr(args, "model_dir", full_args.out_dir)
    print(model_dir)

    # ==============================================================================================
    # @ torc_rl repo original

    # Define logger, CSV writer and Tensorboard writer

    logger = utils.get_logger(model_dir)
    csv_file, csv_writer = utils.get_csv_writer(model_dir)
    tb_writer = None
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

    # Get env wrappers - must be a list of elements
    wrapper_method = getattr(full_args.env_cfg, "wrapper", None)
    if wrapper_method is None:
        def idem(x):
            return x
        env_wrapper = idem
    else:
        env_wrappers = [getattr(gym_wrappers, w_p) for w_p in wrapper_method]
        def env_wrapp(w_env):
            for wrapper in env_wrappers[::-1]:
                w_env = wrapper(w_env)
            return w_env

        env_wrapper = env_wrapp

    actual_procs = getattr(args, "actual_procs", None)
    no_actions = getattr(full_args.env_cfg, "no_actions", 6)

    if actual_procs:
        # Split envs in chunks
        no_envs = args.procs
        envs, chunk_size = get_envs(full_args, env_wrapper, no_envs, n_actions=no_actions)
        first_env = envs[0][0]
        print(f"NO of envs / proc: {chunk_size}; No of processes {len(envs[1:])} + Master")
    else:
        for i in range(args.procs):
            env = env_wrapper(gym.make(args.env))
            env.max_steps = full_args.env_cfg.max_episode_steps

            env.seed(args.seed + 10000 * i)
            envs.append(env)
        first_env = envs[0]

    # Generate evaluation envs
    eval_envs = []
    eval_episodes = getattr(full_args.env_cfg, "eval_episodes", 0)
    if full_args.env_cfg.no_eval_envs > 0:
        no_envs = full_args.env_cfg.no_eval_envs
        eval_envs, chunk_size = get_envs(full_args, env_wrapper, no_envs, n_actions=no_actions)

    # Define obss preprocessor
    max_image_value = full_args.env_cfg.max_image_value
    normalize_img = full_args.env_cfg.normalize
    permute = getattr(full_args.env_cfg, "permute", False)
    obss_preprocessor = getattr(full_args.env_cfg, "obss_preprocessor", None)
    obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, first_env.observation_space,
                                                             model_dir,
                                                             max_image_value=max_image_value,
                                                             normalize=normalize_img,
                                                             permute=permute,
                                                             type=obss_preprocessor)

    first_obs = first_env.reset()
    if "state" in first_obs:
        full_state_size = first_obs["state"].shape

        # Add full size shape
        add_to_cfg(full_args, MAIN_CFG_ARGS, "full_state_size", full_state_size)

    if "position" in first_obs:
        position_size = first_obs["position"].shape

        # Add full size shape
        add_to_cfg(full_args, MAIN_CFG_ARGS, "position_size", position_size)

    # Add the width and height of environment for position estimation
    model_args.width = first_env.unwrapped.width
    model_args.height = first_env.unwrapped.height

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
                          use_memory=model_args.mem)
        logger.info(f"Model [{model_args.name}] successfully created\n")

        # Print Model info
        logger.info("{}\n".format(model))

    if torch.cuda.is_available():
        model.cuda()
    logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

    # ==============================================================================================
    # Load Agent

    algo = get_agent(full_args.agent, envs, model, agent_data,
                     preprocess_obss=preprocess_obss, reshape_reward=None,
                     eval_envs=eval_envs, eval_episodes=eval_episodes)

    has_evaluator = hasattr(algo, "evaluate") and full_args.env_cfg.no_eval_envs > 0

    if return_models:
        return algo, model, envs, saver

    # ==============================================================================================
    # Train model

    prev_rewards = []
    crt_eprew = 0
    if "eprew" in other_data:
        crt_eprew = other_data["eprew"]
    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]
    update_start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        logs = algo.update_parameters()

        num_frames += logs["num_frames"]
        update += 1

        if update % args.eval_interval == 0 and has_evaluator:
            eval_logs = algo.evaluate(eval_key=main_r_key)
            logs.update(eval_logs)

        prev_start_time = update_start_time
        update_start_time = time.time()

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_start_time - prev_start_time)
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
            header += ["entropy", "value", "policy_loss", "value_loss"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"]]
            header += ["grad_norm"]
            data += [logs["grad_norm"]]

            # add log fields that are not in the standard log format (for example value_int)
            extra_fields = extra_log_fields(header, list(logs.keys()))
            header.extend(extra_fields)
            data += [logs[field] for field in extra_fields]

            # print to stdout the standard log fields + fields required in config
            keys_format, printable_data = print_keys(header, data, extra_logs)
            logger.info(keys_format.format(*printable_data))

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

            if main_r_key is None:
                crt_eprew = list(rreturn_per_episode.values())[0]
                prev_rewards.append(crt_eprew)
            else:
                crt_eprew = logs[main_r_key]
                prev_rewards.append(logs[main_r_key])

        # -- Save vocabulary and model

        if args.save_interval > 0 and update % args.save_interval == 0:
            preprocess_obss.vocab.save()

            saver.save_training_data(model, algo.get_save_data(), crt_eprew)

            logger.info("Model successfully saved")

            utils.save_status(status, model_dir)

        check_rew = np.mean(prev_rewards[-max_eprews_window:])
        if len(prev_rewards) > max_eprews_window and check_rew > max_eprews:
            print(f"Reached mean return {max_eprews} for a window of {max_eprews_window} steps")
            exit()


def main() -> None:
    import os

    """ Read configuration from disk (the old way)"""
    # Reading args
    full_args = read_config()  # type: Namespace
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
    from liftoff import parse_opts
    run(parse_opts())
