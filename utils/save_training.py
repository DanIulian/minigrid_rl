# AndreiN, 2019

import os
import torch
import numpy as np
import shutil
import itertools
import glob
import re


def get_training_data_path(model_dir, best=False, index=None):
    if best:
        return os.path.join(model_dir, "training_data_best.pt")

    if index:
        return os.path.join(model_dir, f"training_data_{index}.pt")

    return os.path.join(model_dir, "training_data.pt")


def get_last_training_path_idx(model_dir):
    if os.path.exists(model_dir):
        path = os.path.join(model_dir, "training_data_*.pt")

        max_index = 0
        for path in glob.glob(path):
            try:
                max_index = max(max_index,
                                int(re.findall("training_data_([1-9]\d*|0).pt", path)[0]))
            except:
                 pass

        return max_index
    return 0


class SaveData:
    def __init__(self, model_dir, save_best=True, save_all=False):
        self.model_dir = model_dir
        self.save_best = save_best
        self.save_all = save_all

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        self.best_reward = -np.inf

        start_idx = get_last_training_path_idx(model_dir)
        self.index = itertools.count(start=start_idx, step=1)

    def load_training_data(self, model_dir=None, best=False):
        """ If best is set to false, the last training model is loaded """
        model_dir = model_dir if model_dir is not None else self.model_dir

        training_data = None
        if best:
            path = get_training_data_path(model_dir, best=best)
            if os.path.isfile(path):
                training_data = torch.load(path)

        if training_data is None:
            path = get_training_data_path(model_dir, best=False)
            try:
                training_data = torch.load(path)
            except OSError:
                training_data = dict({"model": None, "agent": {}})

        if "eprew" in training_data:
            self.best_reward = training_data["eprew"]

        return training_data.pop("model"), training_data.pop("agent"), training_data

    def save_training_data(self, model, agent, eprew, other=None, model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir

        trainig_data = dict()
        trainig_data["model"] = model
        trainig_data["agent"] = agent
        trainig_data["eprew"] = eprew

        if other is not None:
            trainig_data.update(other)

        # Save standard
        path = get_training_data_path(model_dir)
        torch.save(trainig_data, path)

        if eprew > self.best_reward:
            self.best_reward = eprew
            best_path = get_training_data_path(model_dir, best=True)
            shutil.copyfile(path, best_path)

        if self.save_all:
            index_path = get_training_data_path(model_dir, index=next(self.index))
            shutil.copyfile(path, index_path)




