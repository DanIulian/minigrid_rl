'''
    by Dan Iulian Muntean 2020
    
    Usage examples:
    
    1. When used to group multiple experiments together using lineplot
    
        python -m analytics.combine_multiple_experiments --experiments_path exp1 exp2 ...
               --algorithm_names  algo_for_exp1 algo_for_exp2 ...
               --group_experiments
               --plot_values plt_value1, plt_value2 ...
               --save_path path_to_save

    2. When used to group multiple values of same experiment together

        python -m analytics.combine_multiple_experiments --experiments_path exp1
               --save_path path_to _save
               --x_axis "frames"
               --melt_data doors_interactions keys_interactions boxes_interactions balls_interactions
               --algorithm_names algo_name_1
               --rolling_window 8

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
    filtered_dfs = []
    if filter_column and filter_value:
        print("FILTERED")
        filter_value = type(dfs[0].iloc[0][filter_column])(filter_value)
        for df in dfs:
            filtered_dfs.append(df[df[filter_column] != filter_value])
        return filtered_dfs
    else:
        print("NO FILTER APPLIED")
        return dfs


def complete_x_axis(df: pd.DataFrame, max_frames: int) -> pd.DataFrame:
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
        return (((d - mean)**2) * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def calc_window_stats(df, index_cl, min_index, max_idx, feature_cl, weights_cl):
    index_cl = df[index_cl]
    df = df[min_index <= index_cl]
    window = df[index_cl < max_idx]

    wmean = calc_wavg(window, feature_cl, weights_cl)  # Mean over
    wstd = np.sqrt(calc_wvar(window, feature_cl, weights_cl, wmean))
    return wmean, wstd, window


def check_weights_column(df: pd.DataFrame, weights_column: str):

    # Add weights column if not present
    if weights_column not in df.columns:
        print(f"No weight column, will fill with 1 ({weights_column})")
        df.loc[:, weights_column] = 1
    else:
        if df[weights_column].isna().any():
            print(f" Some ({weights_column}) are NaN ")
            df[weights_column] = df[weights_column].fillna(1.)


def generate_colors(df: pd.DataFrame, color_by: str):
    # -- Generate color maps for unique color_by subgroups,
    no_colors = df[color_by].nunique()
    subgroup_titles = list(df[color_by].unique())

    colors = cm.rainbow(np.linspace(0, 1, no_colors))
    colors_map = {x: y for x, y in zip(subgroup_titles, colors)}
    legend_elements = {color_name: Line2D(
        [0], [0], marker='o', color=colors_map[color_name], label='Scatter',
        markerfacecolor=colors_map[color_name], markersize=7) for color_name in subgroup_titles}

    return subgroup_titles, colors_map


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
                     melt_data=None,
                     rolling_window=8,
                     weights_column="ep_completed",
                     save_path="combined_results"):
    '''

    :param experiments_path:
    :param algorithm_names:
    :param group_experiments:
    :param title:
    :param plot_values:
    :param filter_by:
    :param filter_value:
    :param x_axis:
    :param plot_type:
    :param main_groupby:
    :param groupby_column:
    :param color_by:
    :param melt_data:
    :param rolling_window:
    :param weights_column:
    :param save_path:
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

    if group_experiments:

        if plot_values is None:
            plot_data = set(dfs[0].columns) - set(EXCLUDED_PLOTS)
            plot_data = [x for x in plot_data if "." not in x]
            import pdb; pdb.set_trace()
            print("Specify a list of plot values from chosen from below")
            print(plot_data)
            exit(0)

        if type(plot_values) is not list:
            plot_values = list(plot_values)

        # Concat everything
        data_frame = pd.concat(dfs, ignore_index=True)
        check_weights_column(data_frame, weights_column)

        # For each subplot
        experiments_group = data_frame.groupby(main_groupby, sort=False)

        plot_experiments_multiple(data_frame, experiments_group,
                                  plot_values, run_idx,
                                  x_axis, plot_type,
                                  save_path, title, color_by)

    else:
        check_weights_column(dfs[0], weights_column)
        plot_experiment_multiple_data(dfs[0],
                                      main_groupby,
                                      x_axis,
                                      melt_data,
                                      save_path,
                                      title,
                                      rolling_window=rolling_window,
                                      weights_column=weights_column,
                                      groupby_column="Melt_type",
                                      color_by="Melt_type")


def plot_experiments_multiple(data_frame: pd.DataFrame,
                              experiments_group: pd.core.groupby.generic.DataFrameGroupBy,
                              plot_values: list,
                              run_idx: str,
                              x_axis: str,
                              plot_type: str,
                              save_path: str,
                              title: str = None,
                              color_by: str = "None"):

    # -- Generate color maps for unique color_by subgroups,
    subgroup_titles, colors_map = generate_colors(data_frame, color_by)

    # ==============================================================================================
    # -- Split the experiments by the grouping element
    # -- If the x-axis is not the same for all experiments, complete it
    # -- by duplicating all values until max X axis value is get
    equal_training_steps = False

    experiments = dict()
    for i, (experiment_name, exp_gdf) in enumerate(experiments_group):

        if equal_training_steps:
            experiments[experiment_name] = exp_gdf

        else:
            max_x_value = exp_gdf['frames'].max()
            df_list = []
            for _, current_df in exp_gdf.groupby(["algo_name", run_idx]):
                rez = complete_x_axis(current_df, max_x_value)
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
                plt.setp(ax.get_legend().get_texts(), fontsize='18')  # for legend text

            elif plot_type == "error_plot":

                # For each sub-group that will appear in the same plot
                for sub_group_name, sub_group_df in exp_gdf.groupby('algo_name'):
                    print(f" Sub-group name: {sub_group_name}")
                    try:
                        assert sub_group_df[color_by].nunique() == 1, f"color by: {color_by} not " \
                                                                      f"unique in subgroup"

                        color_name = sub_group_df[color_by].unique()[0]
                        sub_group_dfs = sub_group_df[[x_axis, weights_column, plot_f]]
                        unique_x = sub_group_dfs[x_axis].unique()
                        unique_x.sort()

                        means = np.zeros(len(unique_x))
                        stds = np.zeros(len(unique_x))

                        # take mean and std value for rolling window
                        for idxs in range(len(unique_x) - rolling_window):
                            wmean, wstd, win = calc_window_stats(sub_group_dfs, x_axis,
                                                                 unique_x[idxs],
                                                                 unique_x[idxs + rolling_window],
                                                                 plot_f, weights_column)

                            means[idxs] = wmean
                            stds[idxs] = wstd
                    except:
                        print("Error!!!!!!!!!")

                    error_fill_plot(unique_x, means, stds, color=colors_map[color_name])
                plt.gcf().legend([legend_elements[x] for x in subgroup_titles], subgroup_titles,
                                 loc="upper left")


            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
            plt.ylabel(plot_f, fontsize=18)
            plt.yticks(fontsize=16)

            plt.xlabel(x_axis, fontsize=18)
            plt.xticks(fontsize=16)

            # Set the title
            if title is None:
                plt.title(exp_gdf.iloc[0].title[9:-3], fontsize=24)
            else:
                plt.title(title, fontsize=24)

            plt.tight_layout()
            plt.savefig(f"{save_path}/{plot_f}_{experiment_name}.png")
            plt.close()


def plot_experiment_multiple_data(df: pd.DataFrame,
                                  main_groupby: str,
                                  x_axis: str,
                                  melt_data: list,
                                  save_path: str,
                                  title: str = None,
                                  rolling_window: int = 8,
                                  weights_column: str = "ep_completed",
                                  groupby_column: str = "Melt_type",
                                  color_by: str = "Melt_type"):

    # -- SPECIAL MELT data
    has_clm = [x in df.columns for x in melt_data]
    assert all(has_clm), f"Incorrect melt list {melt_data}"
    id_vars = list(set(df.columns) - set(melt_data))
    df = pd.melt(df, id_vars=id_vars, var_name="Melt_type", value_name="Melt_value")

    # Generates colors
    subgroup_titles, colors_map = generate_colors(df, color_by)

    # Group by environment type
    experiments_group = df.groupby(main_groupby, sort=False)

    # ==============================================================================================
    plot_f = "Melt_value"
    print(f"Plotting data {plot_f} ...")

    # Iterate through continents
    for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
        print(f"Experiment name: {experiment_name}")
        plt.figure()

        # For each sub-group that will appear in the same plot
        for sub_group_name, sub_group_df in exp_gdf.groupby(groupby_column):
            print(f" Sub-group name: {sub_group_name}")
            try:
                assert sub_group_df[color_by].nunique() == 1, f"color by: {color_by} not " \
                                                              f"unique in subgroup"

                color_name = sub_group_df[color_by].unique()[0]
                sub_group_dfs = sub_group_df[[x_axis, weights_column, plot_f]]
                unique_x = sub_group_dfs[x_axis].unique()
                unique_x.sort()

                means = np.zeros(len(unique_x))
                stds = np.zeros(len(unique_x))

                # take mean and std value for rolling window
                for idxs in range(len(unique_x) - rolling_window):
                    wmean, wstd, win = calc_window_stats(sub_group_dfs, x_axis,
                                                         unique_x[idxs],
                                                         unique_x[idxs + rolling_window],
                                                         plot_f, weights_column)

                    means[idxs] = wmean
                    stds[idxs] = wstd
            except:
                    print("Error!!!!!!!!!")

            error_fill_plot(unique_x, means, stds, color=colors_map[color_name])

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
        plt.gcf().legend([legend_elements[x] for x in subgroup_titles], subgroup_titles,
                         loc="upper left")

        plt.ylabel("Number of interactions", fontsize=18)
        plt.yticks(fontsize=16)

        plt.xlabel(x_axis, fontsize=18)
        plt.xticks(fontsize=16)

        # Set the title
        if title is None:
            plt.title(exp_gdf.iloc[0].title[9:-3], fontsize=24)
        else:
            plt.title(title, fontsize=24)

        plt.tight_layout()
        plt.savefig(f"{save_path}/{experiment_name}.png")
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
    parser.add_argument(
        '--save_path', default="combined_results",
        help=""
    )
    parser.add_argument(
        '--melt_data_custom', nargs='+', default=None,
        help="List of values to be plotted together on the same figure"
    )

    parser.add_argument(
        '--weights_column', default="ep_completed",
        help="DataFrame column to be used as weights for weighted average\n"
             "when computing the rolling window for multiple plots on same graph"
    )
    parser.add_argument(
        "--melt_data", default=None,
        help="Choose melt data from predifined list of "
             "[object_type_interactions, objects_interactions, actions_intrinscic_rewards]"
    )

    melt_data_dict = {
        'object_type_interactions': [
            'keys_interactions',
            'boxes_interactions',
            'balls_interactions',
            'doors_interactions'
        ],
        'objects_interactions': [
            "doors_opened",
            "boxes_picked",
            "keys_picked",
            "balls_picked",
            "doors_closed",
            "boxes_broken",
            "keys_dropped",
            "balls_dropped",
            "boxes_dropped",
        ],
        'actions_intrinsic_rewards': [
            "move_forward_mean_int_r",
            "turn_mean_int_r",
            "obj_interactions_mean_int_r",
            "obj_toggle_mean_int_r"
        ]
    }

    args = parser.parse_args()
    if args.melt_data:
        melt_data = melt_data_dict[args.melt_data]
    else:
        melt_data = args.melt_data_custom

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
                     rolling_window=args.rolling_window,
                     save_path=args.save_path,
                     melt_data=melt_data,
                     weights_column=args.weights_column)
