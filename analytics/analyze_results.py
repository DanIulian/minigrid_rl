import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from pandas.io.json.normalize import nested_to_record
import yaml

experiment_path = "results/2019Feb12-171525_multiple_envs"

cfg_files = glob.glob(f"{experiment_path}/**/cfg.yaml", recursive=True)
log_files = glob.glob(f"{experiment_path}/**/log.csv", recursive=True)
cfg_files.sort()

# -- Load logs
log_dfs = []
cfg_dfs = []
for run_index, cfg_file in enumerate(cfg_files):
    dir_name = os.path.dirname(cfg_file)

    run_name = dir_name.replace(experiment_path, "")
    run_name = run_name[1:] if run_name[0] == "/" else run_name

    # -- Read cfg
    with open(os.path.join(cfg_file)) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)

    experiment_id = config_data["experiment_id"]
    run_id = config_data["run_id"]

    cfg_df = pd.DataFrame(nested_to_record(config_data, sep="."), index=[0])
    cfg_df["run_name"] = run_name
    cfg_df["run_index"] = run_index
    cfg_dfs.append(cfg_df)

    # -- Read logs
    log_file_path = os.path.join(dir_name, "log.csv")
    if os.path.isfile(log_file_path):
        log_df = pd.read_csv(log_file_path)
        log_df["run_index"] = run_index
        log_dfs.append(log_df)

log_dfs = pd.concat(log_dfs)
cfg_dfs = pd.concat(cfg_dfs)

df = pd.merge(log_dfs, cfg_dfs, how="outer", on="run_index")

# check experiment id is unique
if cfg_dfs.experiment_id.duplicated().any():
    print("[ALERT] Duplicate ids found")


log_dfs.boxplot(by="run_index", column="FPS", grid=0)

experiments_group = df.groupby("experiment_id")

size = int(pow(experiments_group.ngroups,1/2))+1
plt.figure()
# Iterate through continents

for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
    # create subplot axes in a 3x3 grid
    ax = plt.subplot(size, size, i + 1) # nrows, ncols, axes position
    # plot the continent on these axes
    exp_gdf.groupby("run_id").plot("frames", "rreturn_mean", ax=ax, legend=False)
    # set the title
    ax.set_title(exp_gdf.iloc[0].title)
    # set the aspect
    # adjustable datalim ensure that the plots have the same axes size
    # ax.set_aspect('equal', adjustable='datalim')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

