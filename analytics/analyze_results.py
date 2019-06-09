"""
Examples:
python -m analytics.analyze_results 'exp_path  --plot_type error_bar --rolling_window 8 --plot_data
                    rreturn_mean --groupby_clm "agent.name" --color_by "agent.name"

MELT DATA
python -m analytics.analyze_results 'exp_path'   --plot_type error_bar --rolling_window 8

python -m analytics.analyze_results 'exp_path'   --plot_type error_bar --rolling_window 8
    --plot_data Melt_value --groupby_clm "Melt_type" --color_by "Melt_type" --max_frames 1e+06
    --filter_by "env_cfg.max_episode_steps" --filter_value 400  --melt_data doors_opened keys_picked
    boxes_picked balls_picked

python -m analytics.analyze_results 'exp_path' --plot_type error_bar --rolling_window 8
    --plot_data Melt_value --groupby_clm "Melt_type" --color_by "Melt_type" --max_frames 1e+06
    --filter_by "env_cfg.max_episode_steps" --filter_value 400  --melt_data doors_opened keys_picked
    boxes_picked balls_picked balls_dropped doors_closed boxes_dropped keys_dropped

"""

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd

from analytics.utils import get_experiment_files

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (18.55, 9.86)
matplotlib.rcParams['axes.titlesize'] = 'medium'

EXCLUDED_PLOTS = [
    'experiment_id', 'frames', 'run_id', 'run_index', 'update', 'cfg_id', 'comment', 'commit',
    'experiment', 'run_id_y', 'run_id_x' 'extra_logs', 'out_dir', 'resume',  'title',
    '_experiment_parameters.env', 'run_name', 'cfg_dir', 'extra_logs', "duration", "run_id_x",
]


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


def calc_window_stats_helper(args):
    return calc_window_stats(*args)


def get_out_dir(experiment_path, new_out_dir=False):
    if not new_out_dir:
        save_path = os.path.join(experiment_path, "analysis_0")
    else:
        import glob
        import re
        out = glob.glob(f"{experiment_path}/analysis_*")
        if len(out) > 0:
            last_idx = max([int(re.match(".*/analysis_(\d+).*", x).group(1)) for x in out])
        else:
            last_idx = 0
        save_path = os.path.join(experiment_path, f"analysis_{last_idx+1}")

    print(f"Saving to ...{save_path} (new dir:{new_out_dir})")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    return save_path


def plot_experiment(experiment_path,
                    x_axis="frames", plot_type="simple", kind="scatter", plot_data=None,
                    main_groupby="main.env", groupby_clm="env_cfg.max_episode_steps",
                    color_by="env_cfg.max_episode_steps",
                    show_legend=False, simple=True,
                    rolling_window=None, weights_cl="ep_completed",
                    new_out_dir=False,
                    max_frames=None,
                    melt_data=None,
                    filter_by=None, filter_value=None):
    """
    :param plot_type: [simple, error_bar, color]
    :param rolling_window: Used for error_bar kind
    """

    # ==============================================================================================

    # Validate config
    assert (plot_type == "error_bar") == (rolling_window is not None), "Window only for error_bar"

    # ==============================================================================================

    # Get experiment data
    data, cfgs, df = get_experiment_files(experiment_path, files={"log.csv": "read_csv"})

    # ==============================================================================================
    # -- FILTER VALUES
    if max_frames:
        print(f"Before max_frames filter: {len(df)}")
        df = df[df["frames"] <= max_frames]
        print(f"After max_frames filter: {len(df)}")

    # Hack to filter some experiments
    if filter_by and filter_value:
        print(f"Before filter: {len(df)}")
        print(f"Filter {filter_by} by {filter_value} from : [{df[filter_by].unique()}]")
        filter_value = type(df.iloc[0][filter_by])(filter_value)
        df = df[df[filter_by] == filter_value]
        print(f"After filter: {len(df)}")
    else:
        print("NO FILTER APPLIED")

    # ==============================================================================================
    # -- SPECIAL MELT data
    if melt_data:
        has_clm = [x in df.columns for x in melt_data]
        assert all(has_clm), f"Incorrect melt list {melt_data}"

        id_vars = list(set(df.columns) - set(melt_data))
        df = pd.melt(df, id_vars=id_vars, var_name="Melt_type", value_name="Melt_value")

    # ==============================================================================================

    # For each subplot
    experiments_group = df.groupby(main_groupby, sort=False)

    # Get path to save plots
    save_path = get_out_dir(experiment_path, new_out_dir=new_out_dir)

    # Run column name
    run_idx = "run_id" if "run_id" in df.columns else "run_id_y"

    # No plot data defined - plot all
    if plot_data is None:
        plot_data = set(df.columns) - set(EXCLUDED_PLOTS)
        plot_data = [x for x in plot_data if "." not in x]

    # ==============================================================================================
    # -- Generate color maps for unique color_by subgroups,

    no_colors = df[color_by].nunique()
    subgroup_titles = list(df[color_by].unique())

    colors = cm.rainbow(np.linspace(0, 1, no_colors))
    colors_map = {x: y for x, y in zip(subgroup_titles, colors)}
    legend_elements = {color_name: Line2D(
        [0], [0], marker='o', color=colors_map[color_name],  label='Scatter',
        markerfacecolor=colors_map[color_name], markersize=7) for color_name in subgroup_titles}

    # ==============================================================================================
    # Custom
    
    export_max_window = True
    export_data = dict()

    # ==============================================================================================

    size = int(pow(experiments_group.ngroups, 1/2))+1

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
            ax = plt.subplot(size, size, i + 1, sharex=share_ax, sharey=share_ay)  # n rows, n cols,
            # axes position
            if share_ax is None:
                share_ax = ax
            if share_ax is None:
                share_ay = ax
            export_data[plot_f][experiment_name] = dict()

            # ======================================================================================

            # plot the continent on these axes
            if plot_type == "error_bar":
                for sub_group_name, sub_group_df in exp_gdf.groupby(groupby_clm):
                    print(f" Sub-group name: {sub_group_name}")

                    export_data[plot_f][experiment_name][sub_group_name] = dict()
                    crd_dict = export_data[plot_f][experiment_name][sub_group_name]
                    max_mean = - np.inf

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

                            if wmean > max_mean:
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

            # Set the title
            if simple:
                ax.set_title(exp_gdf.iloc[0].title[13:-3])
            else:
                ax.set_title(experiment_name)

            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            # ax.set_aspect('equal', adjustable='datalim')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

        # ==========================================================================================

        if show_legend:
            plt.gcf().legend([legend_elements[x] for x in subgroup_titles], subgroup_titles,
                             loc="lower right")

        plt.title(plot_f, fontsize=12, y=-0.50)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.35)
        plt.savefig(f"{save_path}/{plot_f}.png")
        plt.close()

    if export_max_window:
        np.save(f"{save_path}/data", export_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plots plots.')
    parser.add_argument('experiment_path')
    parser.add_argument('--x_axis', default="frames")
    parser.add_argument('--plot_type', default="simple")
    parser.add_argument('--kind', default="scatter")
    parser.add_argument('--plot_data', nargs='+', default=None)
    parser.add_argument('--main_groupby', nargs='+', default="main.env")
    parser.add_argument('--groupby_clm', default="env_cfg.max_episode_steps")
    parser.add_argument('--color_by', default="env_cfg.max_episode_steps")
    parser.add_argument('--no_legend', action='store_true')
    parser.add_argument('--rolling_window', type=int, default=None)
    parser.add_argument('--weights_cl', default="ep_completed")
    parser.add_argument('--same_dir', action='store_true')

    parser.add_argument('--max_frames', type=float, default=None)
    parser.add_argument('--melt_data',  nargs='+', default=None)
    parser.add_argument('--filter_by',  default=None)
    parser.add_argument('--filter_value',  default=None)

    args = parser.parse_args()

    # experiment_path,
    plot_experiment(args.experiment_path, x_axis=args.x_axis, plot_type=args.plot_type,
                    kind=args.kind, plot_data=args.plot_data, main_groupby=args.main_groupby,
                    groupby_clm=args.groupby_clm, color_by=args.color_by,
                    show_legend=(not args.no_legend), rolling_window=args.rolling_window,
                    weights_cl=args.weights_cl, new_out_dir=(not args.same_dir),
                    max_frames=args.max_frames, melt_data=args.melt_data,
                    filter_by=args.filter_by, filter_value=args.filter_value)

