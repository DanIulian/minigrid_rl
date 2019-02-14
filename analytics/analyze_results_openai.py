import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .utils import get_experiment_files


# -- Load experiment Data
experiment_path = "results/2019Feb14-155310_openai_rnd/"
data, cfgs, df = get_experiment_files(experiment_path, files={"progress.csv": "read_csv"})


(df.groupby("run_index").time_elapsed.max()/3600)


experiments_group = df.groupby("experiment_id")

size = int(pow(experiments_group.ngroups,1/2))+1
plt.figure()

# Iterate through continents

for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
    # create subplot axes in a 3x3 grid
    ax = plt.subplot(size, size, i + 1) # nrows, ncols, axes position
    exp_gdf = exp_gdf[~exp_gdf.tcount.isna()]
    if exp_gdf.tcount.any():
        # plot the continent on these axes
        exp_gdf.groupby("run_id").plot("tcount", "eprew", ax=ax, legend=False)
        # set the title
        ax.set_title(exp_gdf.iloc[0].title)
        # set the aspect
        # adjustable datalim ensure that the plots have the same axes size
        # ax.set_aspect('equal', adjustable='datalim')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
