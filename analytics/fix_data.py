import glob
import os
import re
import shutil


# Fix cfgs - add cfg_id
def add_cfg_id_back(search_path):
    cfgs = glob.glob(f"{search_path}/**/cfg.yaml", recursive=True)
    for cfg_path in cfgs:
        exp_folder = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))
        match_cfg_id = int(re.match("^(\d+.)_", exp_folder).group(1))

        with open(cfg_path, "a") as myfile:
            myfile.write(f"\ncfg_id: {match_cfg_id}\n")


def change_cfg_id(search_path):
    cfgs = glob.glob(f"{search_path}/**/cfg.yaml", recursive=True)
    folders = dict()

    for cfg_path in cfgs:
        shutil.copy(cfg_path, cfg_path.replace("/cfg.yaml", "/_cfg_old.yaml"))
        # shutil.copy(cfg_path.replace("/cfg.yaml", "_cfg_old.yaml"), cfg_path)

        exp_folder = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))

        actual_folder = os.path.dirname(os.path.dirname(cfg_path))

        match = re.match("^(\d+.)_", exp_folder).group(1)
        match_cfg_id = int(re.match("^(\d+.)_", exp_folder).group(1))

        # new_id = (match_cfg_id//3) * 7 + match_cfg_id % 3 + 3
        new_id = match_cfg_id + 35

        folders[actual_folder] = actual_folder.replace(exp_folder,
                                                       exp_folder.replace(match, f"{new_id:04d}"))
        with open(cfg_path, "r") as myfile:
            cfg = myfile.readlines()

        for line_no, line in enumerate(cfg):
            if line.startswith("cfg_id:"):
                cfg[line_no] = f"cfg_id: {new_id}\n"

        with open(cfg_path, "w") as myfile:
            myfile.writelines(cfg)

    for k, v in folders.items():
        shutil.move(k, v)

    import pandas as pd
    logs = glob.glob(f"{search_path}/**/log.csv", recursive=True)
    for i, x in enumerate(logs):
        df = pd.read_csv(x)
        if not ("discovered" in df.columns):
            print(f"No discoverd in {x}")
        if "ObstructedMaze" in x:
            if not ("boxes_broken" in df.columns):
                print(f"No boxes_broken in {x}")
        if i % 50 == 0:
            print(f"Done {i}")

