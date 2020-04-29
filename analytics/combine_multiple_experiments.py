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


def plot_experiments(experiments_path,
                     algorithm_names,
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

    # Validate config
    assert (plot_type == "barplot") == (rolling_window is not None), "Window only for error_bar"

    # Get experiment data
    dfs = []
    for algo_name, exp_path in zip(algorithm_names, experiments_path):
        _, _, df = get_experiment_files(exp_path, files={"log.csv": "read_csv"})
        df[groupby_column] = df[groupby_column].apply(lambda x: str(x))
        df['algo_name'] = algo_name
        dfs.append(df)

    # ==============================================================================================
    # -- FILTER VALUES

    # Hack to filter some experiments
    if filter_by and filter_value:
        print("FILTERED")
        filter_value = type(dfs[0].iloc[0][filter_by])(filter_value)
        for df in dfs:
            df = df[df[filter_by] == filter_value]
    else:
        print("NO FILTER APPLIED")

    # ==============================================================================================

    # Concat everything
    data_frame = dfs[0]
    for df in dfs[1:]:
        data_frame = data_frame.append(df)

    # For each subplot
    experiments_group = data_frame.groupby(main_groupby, sort=False)

    # ==============================================================================================

    # Get path to save plots
    print(f"Saving to ...{os.getcwd()}/{save_path}")

    full_save_path = os.getcwd() + "/" + save_path
    if not os.path.isdir(full_save_path):
        os.mkdir(full_save_path)
    save_path = full_save_path

    # Run column name
    run_idx = "run_id" if "run_id" in data_frame.columns else "run_id_y"

    # No plot data defined - plot all
    if plot_values is None:
        plot_values = set(data_frame.columns) - set(EXCLUDED_PLOTS)
        plot_values = [x for x in plot_values if "." not in x]

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

    def _complete_x_axis(df: pd.DataFrame, max_frames: int) -> pd.DataFrame:
        '''Duplicate last row until max_frames have been achieved
        '''
        step_size = df['frames'].iloc[0]
        last_frame = df['frames'].iloc[len(df['frames']) - 1]
        new_df = pd.DataFrame(df)
        for fr in range(last_frame + step_size, max_frames + step_size, step_size):
            new_index = new_df.index[-1] + 1
            new_df = pd.concat([new_df, new_df.tail(1)], ignore_index=True)
            new_df.loc[new_index, 'frames'] = fr

        return new_df

    experiments = dict()
    for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
        max_x_value = exp_gdf['frames'].max()
        df_list = []
        for (caca, current_df) in exp_gdf.groupby(["algo_name", run_idx]):
            df_list.append(_complete_x_axis(current_df, max_x_value))
        experiments[experiment_name] = pd.concat(df_list, ignore_index=True)

    # ================================================================================================

    size = int(pow(experiments_group.ngroups, 1 / 2)) + 1
    w_size = size
    h_size = size
    print(w_size, h_size)
    for plot_f in plot_data:
        print(f"Plotting data {plot_f} ...")

        plt.figure()
        share_ax = None
        share_ay = None
        export_data[plot_f] = dict()

        # Iterate through continents
        for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
            print(f"Experiment name: {experiment_name}")

            # create subplot axes in a 3x3 grid
            ax = plt.subplot(h_size, w_size, i + 1, sharex=share_ax, sharey=share_ay)  # n rows,
            # n cols,
            # axes position
            if share_ax is None:
                share_ax = ax
            if share_ax is None:
                share_ay = ax
            export_data[plot_f][experiment_name] = dict()

            # ======================================================================================

            if i % w_size == 0:
                ax.set_ylabel(plot_f)  # TODO default: plot_f
                # ax.set_ylabel("Fraction")
            if i >= w_size * (h_size - 1):
                ax.set_xlabel(x_axis)

            # Set the title
            if simple:
                ax.set_title(exp_gdf.iloc[0].title[13:-3])
            else:
                ax.set_title(get_title(experiment_name))

            # ======================================================================================

            # plot the continent on these axes
            if plot_type == "error_bar":
                for sub_group_name, sub_group_df in exp_gdf.groupby(groupby_clm):
                    print(f" Sub-group name: {sub_group_name}")

                    export_data[plot_f][experiment_name][sub_group_name] = dict()
                    crd_dict = export_data[plot_f][experiment_name][sub_group_name]
                    max_mean = -np.inf if save_max_w else np.inf

                    try:
                        assert sub_group_df[color_by].nunique() == 1, f"color by: {color_by} not " \
                                                                      f"unique in subgroup"

                        color_name = sub_group_df[color_by].unique()[0]

                        if weights_cl not in sub_group_df.columns:
                            print(f"[ERROR] No weight column, will fill with 1 ({weights_cl})")
                            sub_group_df[weights_cl] = 1
                        else:
                            if sub_group_df[weights_cl].isna().any():
                                print(f"[ERROR] Some ({weights_cl}) are NaN ")
                                sub_group_df[weights_cl] = sub_group_df[weights_cl].fillna(1.)

                        sub_group_dfs = sub_group_df[[x_axis, weights_cl, plot_f]]
                        unique_x = sub_group_dfs[x_axis].unique()
                        unique_x.sort()

                        means = np.zeros(len(unique_x))
                        stds = np.zeros(len(unique_x))

                        for idxs in range(len(unique_x) - rolling_window):
                            wmean, wstd, win = calc_window_stats(sub_group_dfs, x_axis,
                                                                 unique_x[idxs],
                                                                 unique_x[idxs + rolling_window],
                                                                 plot_f, weights_cl)

                            means[idxs] = wmean
                            stds[idxs] = wstd

                            if compare_mthd(wmean, max_mean):
                                crd_dict["df"] = win
                                max_mean = wmean
                    except:
                        print("Error!!!!!!!!!")

                    error_fill_plot(unique_x, means, stds, color=colors_map[color_name], ax=ax)

            elif plot_type == "color":
                for sub_group_name, sub_group_df in exp_gdf.groupby(groupby_clm):
                    assert sub_group_df[color_by].nunique() == 1, f"color by: {color_by} not " \
                                                                  f"unique in subgroup"
                    color_name = sub_group_df[color_by].unique()[0]
                    sub_group_df.plot(x_axis, plot_f, ax=ax, legend=False, kind=kind,
                                      c=colors_map[color_name].reshape(1, -1), s=1.)

            elif plot_type == "simple":
                exp_gdf.groupby(groupby_clm).plot(x_axis, plot_f, ax=ax, legend=False, kind=kind)
            # ======================================================================================

            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            # ax.set_aspect('equal', adjustable='datalim')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

        # ==========================================================================================

        if hack_7_plots:
            # TODO SO so ugly, dont't know why show axis does not work :(
            ax = plt.subplot(h_size, w_size, 8, sharex=share_ax, sharey=share_ay)  # n rows,
            ax.axis('off')

        # ==========================================================================================

        # ==========================================================================================

        if show_legend:
            plt.gcf().legend([legend_elements[x] for x in subgroup_titles], subgroup_titles,
                             loc="lower right")

        #  TODO default
        plt.gcf().suptitle(f"{plot_f.capitalize()} colored by max_steps", fontsize=12)  # , y=-0.50,
        # plt.gcf().suptitle(f"Number of interactions colored by type", fontsize=12)  #, y=-0.50,
        # plt.gcf().suptitle(f"Fraction of interactions colored by object type", fontsize=12)
        # x=-0.2)

        if hack_7_plots:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            plt.tight_layout()

        plt.subplots_adjust(wspace=0.15, hspace=0.35)

        plt.savefig(f"{save_path}/{plot_f}.png")
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
    parser.add_argument('--main_groupby', nargs='+', default="main.env")
    parser.add_argument('--groupby_column', default="env_cfg.max_episode_steps")
    parser.add_argument('--color_by', default="env_cfg.max_episode_steps")
    parser.add_argument('--rolling_window', type=int, default=None)

    args = parser.parse_args()
    plot_experiments(args.experiments_path,
                    args.algorithm_names,
                    plot_values=args.plot_values,
                    filter_by=args.filter_by,
                    filter_value=args.filter_value,
                    x_axis=args.x_axis,
                    plot_type=args.plot_type,
                    main_groupby=args.main_groupby,
                    groupby_column=args.groupby_column,
                    color_by=args.color_by,
                    rolling_window=args.rolling_window)
