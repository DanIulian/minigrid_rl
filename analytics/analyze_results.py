import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from pandas.io.json.normalize import nested_to_record
import yaml

from analytics.utils import get_experiment_files


# -- Load experiment Data
experiment_path = "results/2019Mar01-1_multiple_envs/"
data, cfgs, df = get_experiment_files(experiment_path, files={"log.csv": "read_csv"})


df.boxplot(by="run_index", column="FPS", grid=0)

experiments_group = df.groupby("experiment_id", sort=False)

size = int(pow(experiments_group.ngroups, 1/2))+1
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

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

