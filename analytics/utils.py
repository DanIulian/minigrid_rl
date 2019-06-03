import pandas as pd
import glob
import os
from pandas.io.json.normalize import nested_to_record
import yaml
import natsort
import numpy as np
from typing import Dict, Tuple, List


def get_experiment_files(experiment_path: str, files: dict= {}, flag=False) \
        -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:

    # Assumes each directory (/ experiment run) has a unique cfg
    cfg_files = glob.glob(f"{experiment_path}/**/cfg.yaml", recursive=True)
    cfg_files = natsort.natsorted(cfg_files)

    cfg_dfs = []

    data = dict()

    join_dfs = dict()
    # -- Load cfgs
    for run_index, cfg_file in enumerate(cfg_files):
        data[run_index] = dict()

        dir_name = os.path.dirname(cfg_file)
        data[run_index]["dir_name"] = dir_name

        run_name = dir_name.replace(experiment_path, "")
        run_name = run_name[1:] if run_name[0] == "/" else run_name
        data[run_index]["dir_name"] = run_name

        # -- Read cfg
        with open(os.path.join(cfg_file)) as handler:
            config_data = yaml.load(handler, Loader=yaml.SafeLoader)

        put_manual_id = False
        if "experiment_id" in config_data:
            experiment_id = config_data["experiment_id"]
        else:
            put_manual_id = True
            experiment_id = config_data["cfg_id"]

        run_id = getattr(config_data, "run_id", 0)

        data[run_index]["experiment_id"] = experiment_id
        data[run_index]["run_id"] = run_id

        if flag:
            cfg_df = pd.DataFrame(nested_to_record(config_data, sep="."), index=[0])

        else:
            nc = nested_to_record(config_data)
            for k, v in nc.items():
                if isinstance(v, list):
                    nc[k] = np.array(v).astype(np.object)
            cfg_df = pd.DataFrame.from_dict(nc, orient="index").transpose()

        cfg_df["run_name"] = run_name
        cfg_df["run_index"] = run_index
        cfg_dfs.append(cfg_df)

        data["cfg"] = cfg_df

        # -- Read logs
        for file_name, file_type in files.items():
            file_path = os.path.join(dir_name, file_name)

            if not os.path.isfile(file_path):
                file_path = None
                continue

            file_data = file_path
            if hasattr(pd, str(file_type)) and file_path is not None:
                # Some bad header for experiments Fix

                file_data = getattr(pd, file_type)(file_path)
                if put_manual_id:
                    file_data["experiment_id"] = experiment_id
                    file_data["run_id"] = run_id

                file_data["run_index"] = run_index

                if file_name not in join_dfs:
                    join_dfs[file_name] = []

                join_dfs[file_name].append(file_data)

            data[file_name] = file_data

    cfgs = pd.concat(cfg_dfs)
    merge_dfs = cfgs.copy()

    for join_df_name, join_df in join_dfs.items():
        other_df = pd.concat(join_df, sort=True)
        try:
            try_merge = pd.merge(other_df, merge_dfs, how="left", on="run_index", sort=True)
            merge_dfs = try_merge
        except:
            print(f"Cannot merge {join_df_name}")

    return data, cfgs, merge_dfs


def move_right(x, y):
    return x+1, y


def move_down(x, y):
    return x, y-1


def move_left(x, y):
    return x-1, y


def move_up(x, y):
    return x, y+1


def gen_spiral(no_points):
    moves = [move_right, move_down, move_left, move_up]
    from itertools import cycle
    _moves = cycle(moves)
    n = 1
    pos = 0, 0
    times_to_move = 1

    yield pos

    while True:
        for _ in range(2):
            move = next(_moves)
            for _ in range(times_to_move):
                if n >= no_points:
                    return
                pos = move(*pos)
                n += 1
                yield pos

        times_to_move += 1
