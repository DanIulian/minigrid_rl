#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
from liftoff.config import read_config
from argparse import Namespace
import os
import subprocess

import utils


def run(full_args: Namespace) -> None:

    assert os.path.isfile(full_args.train_path)

    args = full_args.main
    args.seed = full_args.run_id

    out_dir = full_args.out_dir
    os.environ["OPENAI_LOGDIR"] = out_dir
    os.environ["RCALL_NUM_GPU"] = "1"

    args_str = " ".join([f"--{k} {v}" for k, v in args.__dict__.items()])

    command = f"{full_args.train_path} {args_str}"

    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(command, shell=True, stdout=FNULL)
    process.wait()

    print(process.returncode)


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
