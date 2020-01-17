import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from itertools import cycle, islice

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16, 6)
matplotlib.rcParams['figure.figsize'] = (3.5*5, 3.5)
matplotlib.rcParams['axes.titlesize'] = 'medium'

# ==================================================================================================
# Configure multiple experiment paths

plot_data = "same_state_mean"
data_paths = \
{
"ICM": f"/media/andrei/Samsung_T51/data_rl/multiple_max_steps/2019Jun01-154212_v1_multiple_envs_icm/analysis_{plot_data}/data.npy",
"PPO": f"/media/andrei/Samsung_T51/data_rl/multiple_max_steps/2019Jun05-154454_v1_ppo-multiple-envss/analysis_{plot_data}/data.npy",
"RND": f"/media/andrei/Samsung_T51/data_rl/multiple_max_steps/2019Jun09-152149_v1_multiple_envs_rnd/analysis_{plot_data}/data.npy",
"PE": f"/media/andrei/Samsung_T51/data_rl/multiple_max_steps/2019Jun06-142757_v1_multiple_envs_pe/analysis_{plot_data}/data.npy",
}

data_paths = \
{
"tes": f"/home/andrei/Desktop/2019Jun11-222453_world_tests/analysis_6/data.npy",
}


# ==================================================================================================
# Configure columns - MULTIPLE Max steps experiments

PLOT_F = plot_data.capitalize()
EXPERIMENT = "Env"
SUBGROUP1 = "Max_steps"
SUBGROUP2 = "Algorithm"
FEATURE = plot_data
EP_COMPLETED = "ep_completed"

# ==================================================================================================
# Configure columns - Multiple algo eval

data_paths = \
    {
        "7mil_exp": "/media/andrei/Samsung_T51/data_rl/7mil_exp/analysis_1/data.npy",
     }

PLOT_F = "Return"
EXPERIMENT = "Env"
SUBGROUP1 = "Algorithm"
SUBGROUP2 = "None"
FEATURE = "rreturn_mean"
EP_COMPLETED = "ep_completed"

# ==================================================================================================
# Aggregate data from windows

# Filter only episodes completed
filter_completed = True


def aggregate_data(data_paths):
    merge_df = []

    for experiment_group, data_path in data_paths.items():
        data = np.load(data_path).item(0)

        for plot_f in data.keys():
            for experiment_name in data[plot_f].keys():
                for sub_group_name in data[plot_f][experiment_name].keys():
                    df = data[plot_f][experiment_name][sub_group_name]["df"]

                    df[PLOT_F] = plot_f
                    df[EXPERIMENT] = experiment_name
                    df[SUBGROUP1] = sub_group_name
                    df[SUBGROUP2] = experiment_group
                    merge_df.append(df)

    df = pd.concat(merge_df)

    if filter_completed:
        df = df[df[EP_COMPLETED] != 0]
    return df


df = aggregate_data(data_paths)

# df.boxplot("rreturn_mean", ["Env", "Max_steps"], rot=90); plt.show()
# df.boxplot("categories_interactions", ["Env", "Max_steps"], rot=90); plt.show()
# ==================================================================================================
import re


def get_title(title: str):
    title = title.replace("MiniGrid-", "")
    title = re.sub("-v\d+", "", title)
    return title

# ==================================================================================================
# -- Run multiple experiments box plot


subplot_by = EXPERIMENT
color_by = SUBGROUP2

groups = df[subplot_by].unique()
color_group = df[color_by].unique()
colors = ['b', 'y', 'm', 'c', 'g', 'b', 'r', 'k', ]

fig, axes = plt.subplots(1, len(groups), sharey=True)
cc = cycle(colors[:len(color_group)])

sub1 = df[SUBGROUP1].unique()
sub2 = df[SUBGROUP2].unique()
pos_t = []
tick = []

for i in range(len(sub1)):
    tick.append(sub1[i])
    for j in range(len(sub2)):
        pos_t.append(i * (len(sub2)+0.5) + j)
        tick.append("")
    tick.pop()

flierprops = dict(marker='.', markerfacecolor='black', markersize=1, linestyle='none',
                  markeredgecolor=None)
# axes = [axes]
for n, group in enumerate(groups):
    idf = df[df[subplot_by] == group]
    idf = idf.sort_values(SUBGROUP2)

    bp_dict = idf.boxplot(column=FEATURE, by=[SUBGROUP1, SUBGROUP2], rot=90, ax=axes[n],
                          return_type='both',
                          patch_artist=True,
                          positions=pos_t, flierprops=flierprops,showfliers=False)

    for row_key, (ax, row) in bp_dict.iteritems():
        ax.set_xlabel(f'{SUBGROUP1}')
        for i, box in enumerate(row['boxes']):
            box.set_facecolor(next(cc))

    axes[n].set_title(get_title(group))
    if n == 0:
        axes[n].set_ylabel("Percentage Discovered")
    axes[n].set_xticklabels(tick)
    # plt.setp(bp_dict['whiskers'], linewidth=3)

    # axes[n].set_xticklabels([e[1] for e in idf.columns])

plt.subplots_adjust(left=0.04, bottom=0.2, right=0.98, top=0.86, wspace=0.07, hspace=0.2)
markers = [plt.Line2D([0,0], [0,0], color=next(cc), marker='o', linestyle='') for item in
           range(len(sub2))]
plt.legend(markers, idf[SUBGROUP2].unique(), numpoints=1)

fig.set_size_inches(16.0, 4, forward=True)

fig.suptitle(f'{plot_data.capitalize()} boxplot grouped by (Env, Max_steps, Algorithm)')
# fig.suptitle("")
plt.show()


# ==================================================================================================
# -- Run multiple max score bar chart

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


mean_df = pd.DataFrame(df.groupby([EXPERIMENT, SUBGROUP1]).mean().to_records())
mean_df = mean_df.sort_values(FEATURE, ascending=False).reset_index()
# mean_df["frames_p"] = mean_df["frames"] / mean_df["frames"].max()
mean_df["env_name"] = mean_df[EXPERIMENT].apply(get_title)

fig, axes = plt.subplots(2, 1, sharex=True)
# plt.ticklabel_format(style='sci',axis='y')

mean_score = mean_df.groupby(EXPERIMENT).mean().sort_values(FEATURE, ascending=False)
idx = mean_score.index.values

pivot = mean_df.pivot("env_name", SUBGROUP1, FEATURE)
pivot_idx = list(pivot.index)
idx_s = [pivot_idx.index(get_title(x)) for x in idx]
print(mean_score.index.values)

ax1 = mean_df.pivot("env_name", SUBGROUP1, FEATURE).iloc[idx_s].plot(kind='bar', ax=axes[0], rot=90)
ax1.set_ylabel("Maximum reward")

ax2 = mean_df.pivot("env_name", SUBGROUP1, "frames").iloc[idx_s].plot(kind='bar', ax=axes[1],
                                                                      rot=90, style='sci')
ax2.set_ylabel("No. frames")

ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

ax1.yaxis.grid()
ax2.yaxis.grid()

ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

fig.suptitle("MiniGrid environments Baselines - Max reward", fontsize=16)
plt.show()


