import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (18.55, 9.86)
matplotlib.rcParams['axes.titlesize'] = 'medium'
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import os
from analytics.utils import get_experiment_files
import re

# -- Load experiment Data


experiment_path = "results/2019Apr10-202258_multiple_envs_icm/"
data, cfgs, df = get_experiment_files(experiment_path, files={"log.csv": "read_csv"})

def extract_name(s):
    s = re.sub("_env[^/]*/", "", s)
    s = re.sub("^\d*[^_]*_", "", s)
    return s


def plot_experiment(experiment_path, main_groupby="main.env", x_axis="frames", groupby_clm="run_name", kind="scatter",
                    show_legend=False, simple=True):
    # experiment_path = "results/latest/"
    data, cfgs, df = get_experiment_files(experiment_path, files={"log.csv": "read_csv"})

    # df.boxplot(by="run_index", column="FPS", grid=0)

    # experiments_group = df.groupby("experiment_id", sort=False)
    experiments_group = df.groupby(main_groupby, sort=False)

    save_path = os.path.join(experiment_path, "analysis")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    not_plot = ['frames', 'run_id', 'run_index', 'update', 'cfg_id', 'comment', 'commit', 'experiment',
                'extra_logs', 'out_dir', 'resume', 'title', '_experiment_parameters.env',  'run_name', 'cfg_dir']

    plot_data = set(df.columns) - set(not_plot)
    plot_data = [x for x in plot_data if "." not in x]

    experiment_name, exp_gdf = next(iter(experiments_group))
    no_subgroups = exp_gdf.groupby(groupby_clm).ngroups
    subgroup_names = list(exp_gdf.groupby(groupby_clm).groups.keys())
    if not simple:
        subgroup_titles = [extract_name(x) for x in subgroup_names]
    else:
        subgroup_titles = [str(x) for x in subgroup_names]

    colors = cm.rainbow(np.linspace(0, 1, no_subgroups))
    colors_map = {x: y for x, y in zip(subgroup_titles, colors)}
    legend_elements = {color_name: Line2D([0], [0], marker='o', color=colors_map[color_name], label='Scatter',
                                          markerfacecolor=colors_map[color_name], markersize=7)
                       for color_name in subgroup_titles}

    size = int(pow(experiments_group.ngroups, 1/2))+1

    for plot_p in plot_data:
        plt.figure()
        # Iterate through continents
        share_ax = None
        for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
            # create subplot axes in a 3x3 grid
            ax = plt.subplot(size, size, i + 1)  # n rows, n cols, axes position
            if share_ax is None:
                share_ax = ax
            # plot the continent on these axes
            if simple:
                exp_gdf.groupby(groupby_clm).plot(x_axis, plot_p, ax=ax, legend=False, kind=kind)
            else:
                for sub_group_name, sub_group_df in exp_gdf.groupby(groupby_clm):
                    color_name = extract_name(sub_group_name)
                    sub_group_df.plot(x_axis, plot_p, ax=ax, legend=False, kind=kind, c=colors_map[color_name], s=1.)

            # set the title
            if simple:
                ax.set_title(exp_gdf.iloc[0].title[13:-3])
            else:
                ax.set_title(experiment_name)

            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            # ax.set_aspect('equal', adjustable='data lim')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

        # handles, labels = ax.get_legend_handles_labels()
        if show_legend:
            plt.gcf().legend([legend_elements[x] for x in subgroup_titles], subgroup_titles, loc="lower right")

        plt.title(plot_p, fontsize=12, y=-0.50)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.35)
        plt.savefig(f"{save_path}/{plot_p}.png")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path')
    parser.add_argument('--complex', action='store_true', default=False)
    args = parser.parse_args()

    if not args.complex:
        plot_experiment(args.path, main_groupby="experiment_id", groupby_clm="run_id", kind="line", simple=True,
                        show_legend=False)
    else:
        plot_experiment(args.path, show_legend=True, kind="scatter", simple=False)
