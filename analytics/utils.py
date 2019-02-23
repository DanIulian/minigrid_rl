import pandas as pd
import glob
import os
from pandas.io.json.normalize import nested_to_record
import yaml
import types
import natsort


def get_experiment_files(experiment_path: str, files: dict= {}):

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

        experiment_id = config_data["experiment_id"]
        run_id = config_data["run_id"]

        data[run_index]["experiment_id"] = experiment_id
        data[run_index]["run_id"] = run_id

        cfg_df = pd.DataFrame(nested_to_record(config_data, sep="."), index=[0])
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
                # file_data = getattr(pd, file_type)(file_path, skiprows=1, names=['update', 'frames', 'FPS', 'duration', 'rreturn_mean', 'rreturn_std', 'rreturn_min', 'rreturn_max', 'num_frames_mean', 'num_frames_std', 'num_frames_min', 'num_frames_max', 'entropy', 'value', 'policy_loss', 'value_loss', 'grad_norm', 'value_ext', 'value_int', 'value_ext_loss', 'value_int_loss', 'return_mean', 'return_std', 'return_min', 'return_max'])
                file_data = getattr(pd, file_type)(file_path)
    
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
