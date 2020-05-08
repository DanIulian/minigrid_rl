'''
    by Dan Iulian Muntean 2020
'''
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import visdom
import seaborn as sns

from analytics.utils import get_experiment_files

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (20, 10)
matplotlib.rcParams['axes.titlesize'] = 'medium'

EXCLUDED_PLOTS = [
    'experiment_id', 'frames', 'run_id',
    'run_index', 'update', 'cfg_id',
    'comment', 'commit', 'experiment',
    'run_id_y', 'run_id_x', 'extra_logs',
    'out_dir', 'resume', 'title',
    '_experiment_parameters.env', 'run_name',
    'cfg_dir', 'extra_logs', "duration",
    'ep_completed', 'FPS', 'num_frames_max',
    'num_frames_min', 'num_frames_std'
]


def get_title(title: str):
    title = title.replace("MiniGrid-", "")
    title = re.sub("-v\d+", "", title)
    return title


def extract_name(s):
    return int(re.match("^(\d+.)_", s).group(1))


def error_fill_plot(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def calc_wavg(group, avg_name, weight_name):
    """ Calculate weighted average from pandas dataframe """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def calc_wvar(group, avg_name, weight_name, mean):
    """ Calculate weighted variance from pandas dataframe """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (((d - mean) ** 2) * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def calc_window_stats(df, index_cl, min_index, max_idx, feature_cl, weights_cl):
    index_cl = df[index_cl]
    df = df[min_index <= index_cl]
    window = df[index_cl < max_idx]

    wmean = calc_wavg(window, feature_cl, weights_cl)  # Mean over
    wstd = np.sqrt(calc_wvar(window, feature_cl, weights_cl, wmean))
    return wmean, wstd, window


def calc_window_stats_helper(args):
    return calc_window_stats(*args)


def get_experiments_data(experiments_path, algorithm_names, groupby_column):
    # Get experiment data
    dfs = []
    for algo_name, exp_path in zip(algorithm_names, experiments_path):
        _, _, df = get_experiment_files(exp_path, files={"log.csv": "read_csv"})
        df[groupby_column] = df[groupby_column].apply(lambda x: str(x))
        df['algo_name'] = algo_name
        dfs.append(df)

    return dfs


def filter_dfs(filter_column, filter_value, dfs):
    # Hack to filter some experiments
    if filter_column and filter_value:
        print("FILTERED")
        filter_value = type(dfs[0].iloc[0][filter_column])(filter_value)
        for df in dfs:
            df = df[df[filter_column] == filter_value]
    else:
        print("NO FILTER APPLIED")

    return dfs


def plot_experiments(experiments_path,
                     algorithm_names,
                     group_experiments=False,
                     title=None,
                     plot_values=None,
                     filter_by=None,
                     filter_value=None,
                     x_axis="frames",
                     plot_type="lineplot",
                     main_groupby="main.env",
                     groupby_column="run_id_x",
                     color_by="None",
                     rolling_window=None,
                     save_path="combined_results"):
    '''

    :param experiments_path:
    :param algorithm_names:
    :param group_experiments
    :param title:
    :param plot_value:
    :param filter_by:
    :param filter_value:
    :param x_axis:
    :param plot_type:
    :param main_groupby:
    :param groupby_column:
    :param color_by:
    :param rolling_window:
    :return:
    '''

    dfs = get_experiments_data(experiments_path, algorithm_names, groupby_column)

    # -- FILTER VALUES
    dfs = filter_dfs(filter_by, filter_value, dfs)

    # Get path to save plots
    print(f"Saving to ...{os.getcwd()}/{save_path}")

    full_save_path = os.getcwd() + "/" + save_path
    if not os.path.isdir(full_save_path):
        os.mkdir(full_save_path)
    save_path = full_save_path

    # Run column name
    run_idx = "run_id" if "run_id" in dfs[0].columns else "run_id_y"

    if plot_values is None:
        plot_data = set(dfs[0].columns) - set(EXCLUDED_PLOTS)
        plot_data = [x for x in plot_data if "." not in x]
        print("Specify a list of plot values from chosen from below")
        print(plot_data)
        exit(0)

    if type(plot_values) is not list:
        plot_values = list(plot_values)

    if group_experiments:
        # Concat everything
        data_frame = pd.concat(dfs, ignore_index=True)
        # For each subplot
        experiments_group = data_frame.groupby(main_groupby, sort=False)

    else:
        pass

    # ==============================================================================================
    # -- Generate color maps for unique color_by subgroups,

    no_colors = data_frame[color_by].nunique()
    subgroup_titles = list(data_frame[color_by].unique())

    colors = cm.rainbow(np.linspace(0, 1, no_colors))
    colors_map = {x: y for x, y in zip(subgroup_titles, colors)}
    legend_elements = {color_name: Line2D(
        [0], [0], marker='o', color=colors_map[color_name], label='Scatter',
        markerfacecolor=colors_map[color_name], markersize=7) for color_name in subgroup_titles}

    # ==============================================================================================
    # -- Split the experiments by the grouping element
    # -- If the x-axis is not the same for all experiments, complete it
    # -- by duplicating all values until max X axis value is get
    equal_training_steps = False

    def _complete_x_axis(df: pd.DataFrame, max_frames: int) -> pd.DataFrame:
        '''Duplicate last row until max_frames have been achieved
        '''
        step_size = df['frames'].iloc[0]
        last_frame = df['frames'].iloc[len(df['frames']) - 1]
        new_df = pd.DataFrame(df)
        new_df = new_df.reset_index().drop('index', axis=1)
        for fr in range(last_frame + step_size, max_frames + step_size, step_size):
            new_index = new_df.index[-1] + 1
            new_df = pd.concat([new_df, new_df.tail(1)], ignore_index=True)
            new_df.loc[new_index, 'frames'] = fr

        return new_df

    experiments = dict()
    for i, (experiment_name, exp_gdf) in enumerate(experiments_group):

        if equal_training_steps:
            experiments[experiment_name] = exp_gdf

        else:
            max_x_value = exp_gdf['frames'].max()
            df_list = []
            for _, current_df in exp_gdf.groupby(["algo_name", run_idx]):
                rez = _complete_x_axis(current_df, max_x_value)
                df_list.append(rez)
            experiments[experiment_name] = pd.concat(df_list, ignore_index=True)

    # ================================================================================================

    for plot_f in plot_values:
        print(f"Plotting data {plot_f} ...")

        # Iterate through continents
        for i, experiment_name in enumerate(experiments.keys()):

            exp_gdf = experiments[experiment_name]
            print(f"Experiment name: {experiment_name}")

            plt.figure()

            # ======================================================================================
            # plot the continent on these axes
            if plot_type == "lineplot":
                ax = sns.lineplot(x=x_axis,
                                  y=plot_f,
                                  hue='algo_name',
                                  data=exp_gdf,
                                  palette='Set1',
                                  ci=95)

            plt.ylabel(plot_f, fontsize=18)
            plt.xlabel(x_axis, fontsize=18)
            plt.setp(ax.get_legend().get_texts(), fontsize='18')  # for legend text

            # Set the title
            if title is None:
                plt.title(exp_gdf.iloc[0].title[9:-3], fontsize=24)
            else:
                plt.title(title, fontsize=24)

            plt.tight_layout()
            plt.savefig(f"{save_path}/{plot_f}_{experiment_name}.png")
            plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plots plots.')

    parser.add_argument(
        '--experiments_path', nargs='+', default=None,
        help="Specify a list of experiments whose results to combine"
    )
    parser.add_argument(
        '--algorithm_names', nargs='+', default=None,
        help="Name of algorithms used for each experiment"
    )
    parser.add_argument(
        '--plot_values', nargs='+', default=None,
        help="List of logged values for which we want plots\n"
             "If no value is specified, compute all available plots"
    )
    parser.add_argument(
        '--filter_by', default=None,
        help="Filter option. For example if you want to plot only for entrys where"
             "ep_completed is not 0"
    )
    parser.add_argument(
        '--filter_value', default=None
    )
    parser.add_argument(
        '--x_axis', default="frames"
    )
    parser.add_argument(
        '--plot_type', default="lineplot",
        help="Type of seaborn plot to be used"
    )
    parser.add_argument(
        '--main_groupby', nargs='+',
        default="main.env",
        help=""
    )
    parser.add_argument(
        '--groupby_column', default="env_cfg.max_episode_steps",
        help=""
    )
    parser.add_argument(
        '--color_by', default="env_cfg.max_episode_steps",
        help=""
    )
    parser.add_argument(
        '--rolling_window', type=int, default=1,
        help=""
    )
    parser.add_argument(
        '--group_experiments', action='store_true',
        default=False,
        help=""
    )

    args = parser.parse_args()
    plot_experiments(args.experiments_path,
                    args.algorithm_names,
                    group_experiments=args.group_experiments,
                    plot_values=args.plot_values,
                    filter_by=args.filter_by,
                    filter_value=args.filter_value,
                    x_axis=args.x_axis,
                    plot_type=args.plot_type,
                    main_groupby=args.main_groupby,
                    groupby_column=args.groupby_column,
                    color_by=args.color_by,
                    rolling_window=args.rolling_window)
