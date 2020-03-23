import numpy as np
import os
import glob
import itertools
import pandas as pd
import re
from typing import List
from utils.gym_wrappers import JustMove

"""
import sys
from liftoff import parse_opts


sys.argv = ['evaluate_model.py', '/media/andrei/CE04D7C504D7AF292/rl/minigrid_rl/results/2020Mar22-195959_aaaE_cstR_ppo-multiple-envss/0000_env_cfg.env_args.goal_rand_offset_0/0/cfg.yaml', '--session-id', '1']
opts = parse_opts()
"""

AGENT_ID = 10

DIR_NAMES = ["right", "down", "left", "up"]
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


def run_freeMove(opts):
    import cv2
    from train_main import run as train_run
    import torch

    def img_view(img: np.ndarray, scale: int = 1, view: str = None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if view:
            cv2.imshow(view, img)
        return img

    algo, model, envs, saver = train_run(opts, return_models=True)
    device = algo.device
    orig_goal_rand_offset = getattr(opts.env_cfg.env_args, "goal_rand_offset", 0)
    orig_goal_pos = getattr(opts.env_cfg.env_args, "goal_pos", [5, 5])

    model_paths = glob.glob(f"{opts.out_dir}/checkpoints/training_data*")
    model_ids = [int(re.findall("training_data_(.*).pt", x)[0]) for x in model_paths]
    model_ids.sort()

    df_log = pd.read_csv(opts.out_dir + "/log.csv")
    env = envs[0][0]

    env_obs = env.observation(env.gen_obs())
    obs = env_obs["image"]
    env.unwrapped.goal_rand_offset = 0
    env.unwrapped.goal_start_pos = [3, 2]
    env.reset()

    env.unwrapped.agent_pos = [1, 1]
    env.unwrapped.agent_dir = 2

    img_render = env.render('rgb_array')
    agent_pos = env.unwrapped.agent_pos
    goal_pos = env.unwrapped.goal_start_pos
    print(obs[tuple(agent_pos)])

    # img = img_view(img_render, view="Test")
    # cv2.waitKey(0)
    #
    # ==============================================================================================
    # -- Run model for all states
    msize = env.unwrapped.width - 1

    # -- Get states
    all_states = []

    g_off = orig_goal_rand_offset + 1
    rangers = [
        range(-g_off, g_off + 1),
        range(-g_off, g_off + 1),
        range(1, msize), range(1, msize)
    ]
    states_conf = np.array(list(itertools.product(*rangers)))
    states_conf[:, 0] += orig_goal_pos[0]
    states_conf[:, 1] += orig_goal_pos[1]

    crt_goal_pos = (None, None)
    for goal_x, goal_y, pos_x, pos_y in states_conf:

        env.unwrapped.goal_rand_offset = None
        env.unwrapped.goal_start_pos = [goal_x, goal_y]
        env.reset()

        env.unwrapped.agent_pos = [pos_x, pos_y]
        env.unwrapped.agent_dir = np.random.randint(0, 4)

        env_obs = env.observation(env.gen_obs())
        all_states.append(env_obs)

    # -- Get values for states
    max_obs_batch = 1024
    models_data = dict({})
    models_other_info = dict({})
    for model_id in model_ids:
        model, agent_data, other_data = saver.load_training_data(best=False, index=model_id)
        other_data["epoch"] = epoch = model_id * opts.main.save_interval
        if sum(df_log["update"] == epoch) > 0 and "eval_r" in df_log.columns:
            other_data["eval_r"] = df_log[df_log["update"] == epoch].iloc[0]["eval_r"]

        dists, vpreds = [], []
        for i in range(0, len(all_states), max_obs_batch):
            prep_obs = algo.preprocess_obss(all_states[i: i+max_obs_batch], device=device)
            dist, vpred, _ = model(prep_obs, None)
            dists.append(dist)
            vpreds.append(vpred)

        dist.logits = torch.cat([x.logits for x in dists])
        vpred = torch.cat(vpreds)

        data = {
            'goal_x': states_conf[:, 0],
            'goal_y': states_conf[:, 1],
            'pos_x': states_conf[:, 2],
            'pos_y': states_conf[:, 3],
            'vpred': vpred.data.cpu().numpy(),
            'prob': list(dist.probs.data.cpu().numpy()),
            'entropy': list(dist.entropy().data.cpu().numpy()),
        }

        df = pd.DataFrame(data)
        df["model_id"] = model_id
        models_data[model_id] = df
        models_other_info[model_id] = other_data

    # df = pd.concat(models_data.values())

    # -- Plot

    plot_data = {"models_data": models_data, "other_info": models_other_info,
                 "model_ids": model_ids}
    np.save(f"e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+{orig_goal_rand_offset}", plot_data)


def plot_entropy_heatmap(opts):
    import cv2
    from train_main import run as train_run
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    matplotlib.use('TkAgg')

    def img_view(img: np.ndarray, scale: int = 1, view: str = None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if view:
            cv2.imshow(view, img)
        return img

    algo, model, envs, saver = train_run(opts, return_models=True)
    device = algo.device
    orig_goal_rand_offset = getattr(opts.env_cfg.env_args, "goal_rand_offset", 0)
    orig_goal_pos = getattr(opts.env_cfg.env_args, "goal_pos", [5, 5])

    model_paths = glob.glob(f"{opts.out_dir}/checkpoints/training_data*")
    model_ids = [int(re.findall("training_data_(.*).pt", x)[0]) for x in model_paths]
    model_ids.sort()

    env = envs[0][0]
    msize = env.unwrapped.width - 1

    env_obs = env.observation(env.gen_obs())
    obs = env_obs["image"]
    env.unwrapped.goal_rand_offset = 0
    env.unwrapped.goal_start_pos = [3, 2]
    env.reset()

    # ==============================================================================================
    # # -- Visualize
    # env.unwrapped.goal_rand_offset = None
    # env.unwrapped.goal_start_pos = goal_pos
    # env.reset()
    # env.unwrapped.agent_pos = [1, 1]
    # env.unwrapped.agent_dir = np.random.randint(0, 4)
    #
    # img_render = env.render('rgb_array')
    # background_img = img_view(img_render, view="Test")
    # cv2.waitKey(0)
    # # ---

    # ==============================================================================================
    # -- Get states

    ag_states = np.array(list(itertools.product(range(1, msize), range(1, msize))))
    goal_states = np.array(list(itertools.product(
        range(-orig_goal_rand_offset, orig_goal_rand_offset + 1),
        range(-orig_goal_rand_offset, orig_goal_rand_offset + 1)
    )))
    goal_states[:, 0] += orig_goal_pos[0]
    goal_states[:, 1] += orig_goal_pos[1]

    # ==============================================================================================

    def get_ag_states_stats(model_id, goal_pos):
        model, agent_data, other_data = saver.load_training_data(best=False, index=model_id)

        done, step = False, 0
        env.unwrapped.goal_rand_offset = None
        env.unwrapped.goal_start_pos = goal_pos
        env.reset()

        env_obss = []
        for x, y in ag_states:

            env.unwrapped.agent_pos = [x, y]
            env.unwrapped.agent_dir = np.random.randint(0, 4)

            env_obs = env.observation(env.gen_obs())
            env_obss.append(env_obs)

            # img_render = env.render('rgb_array')
            # background_img = img_view(img_render, view="Test")
            # cv2.waitKey(0)

        prep_obs = algo.preprocess_obss(env_obss, device=device)
        dist, vpred, memory = model(prep_obs, None)

        return dist, vpred

    goal_pos = [8, 8]
    dist, vpred = get_ag_states_stats(model_ids[-1], goal_pos)

    # map_scores = dist.entropy().data.cpu().view(14, 14).numpy().transpose()
    map_scores = vpred.data.cpu().view(14, 14).numpy().transpose()

    df_select = df[(df.goal_x == goal_pos[0]) & (df.goal_y == goal_pos[1])]
    map_scores = df_select["vpred"].values.reshape(14,14).transpose()

    fig, ax = plt.subplots()

    plt.imshow(map_scores, cmap='hot', interpolation=None)
    plt.colorbar()

    for i, j in env.unwrapped.test_goals - 1:
        text = ax.text(j, i, "e", ha="center", va="center", color="w")
    text = ax.text(goal_pos[0]-1, goal_pos[1]-1, "G", ha="center", va="center", color="w")
    plt.tight_layout()
    plt.show()


def get_directions(df_select):
    move = JustMove._move_actions

    probs = df_select.orig_prob.values
    probs = np.stack(list(probs))[:, :4] / 2.
    pos = df_select[["pos_x", "pos_y"]].values

    xs = []
    ys = []
    center = pos + 0.5

    for direction in range(4):
        mvc = np.stack([move[direction]] * len(probs)) * np.stack([probs[:, direction]]*2, axis=1)
        mvc = center + mvc

        for i in range(len(center)):
            xs.append(np.array([center[i, 0], mvc[i, 0]]))
            ys.append(np.array([center[i, 1], mvc[i, 1]]))
    return xs, ys


def plot_values(df_paths: List[str]):
    from bokeh.io import curdoc
    # from bokeh.io import output_notebook; output_notebook()
    from bokeh.plotting import figure, output_file
    from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
    from bokeh.transform import transform
    from bokeh.layouts import column, gridplot, row
    from bokeh.models import CustomJS, Slider
    from bokeh.models.widgets import Select, Div, RadioGroup
    from bokeh.models import ColumnDataSource
    from bokeh.models import Span

    experiments = dict()
    paths = dict()
    read_one = None

    def read_plot_data(path: str):
        plot_data = np.load(path, allow_pickle=True).item()
        plot_data["goals"] = np.unique(
            plot_data["models_data"][0][["goal_x", "goal_y"]].values, axis=0)
        return plot_data

    for df_path in df_paths:
        exp_name = os.path.splitext(os.path.basename(df_path))[0]
        paths[exp_name] = df_path
        if read_one is not None:
            experiments[exp_name] = None
        else:
            experiments[exp_name] = read_plot_data(df_path)
            read_one = exp_name

    exp_name = read_one
    plot_data = experiments[exp_name]
    goals = plot_data["goals"]
    goal_min_x, goal_max_x = min(goals[:, 0]), max(goals[:, 0])
    goal_min_y, goal_max_y = min(goals[:, 1]), max(goals[:, 1])

    df1 = plot_data["models_data"][experiments[exp_name]["model_ids"][0]]
    pos_max_x, pos_max_y = max(df1.pos_x), max(df1.pos_y)

    cname_stats = dict()
    for cname in ["vpred", "entropy"]:
        cname_stats[cname] = [np.inf, -np.inf]
        for select_model_id in plot_data["model_ids"]:
            df_new = plot_data["models_data"][select_model_id]
            cname_stats[cname][0] = min(cname_stats[cname][0], df_new[cname].min())
            cname_stats[cname][1] = max(cname_stats[cname][1], df_new[cname].max())

    def get_num_models(exp_name):
        print("SHOW LEN MODELS FOR ", exp_name)
        return len(experiments[exp_name]["model_ids"])

    def get_model(exp_name, select_model_id, goal, norm_type="per_training_step"):
        plot_data = experiments[exp_name]

        goals = plot_data["goals"]
        has_goal = np.all(goals == np.array(goal), axis=1).any()

        if not has_goal:
            print("ERRROR! NO GOAL !!!!")
            return None

        # Select model id
        df_new = plot_data["models_data"][select_model_id]

        # Select goal
        df_select = pd.DataFrame(df_new[(df_new.goal_x == goal[0]) & (df_new.goal_y == goal[1])])

        df_select["orig_prob"] = df_select["prob"]
        df_select["prob"] = df_select["prob"].apply(lambda x: str([np.round(y, 3) for y in x]))

        plots = []

        # ==========================================================================================
        # Plot Values

        def plot_content(cname: str):
            if norm_type == "per_training_step":
                plot_min, plot_max = df_select[cname].min(), df_select[cname].max()
            elif norm_type == "entire_training":
                plot_min, plot_max = cname_stats[cname][0], cname_stats[cname][1]

            df_prep = df_select
            # df_prep = df_select[df_select.dir == direction].copy()
            df_prep = df_prep.sort_values(["pos_x", "pos_y"])
            df_prep["x"] = df_prep.pos_x.values.astype(np.str)
            df_prep["y"] = df_prep.pos_y.values.astype(np.str)

            # Eliminate goal
            df_prep = df_prep[~((df_prep["x"] == str(goal[0])) & (df_prep["y"] == str(goal[1])))]

            # Specify goal position
            sel = df_prep[df_prep["x"] == str(goal[0])].index
            df_prep.loc[sel, "x"] = str(goal[0]) + "_goal"

            sel = df_prep[df_prep["y"] == str(goal[1])].index
            df_prep.loc[sel, "y"] = str(goal[1]) + "_goal"

            mapper = LinearColorMapper(palette='Viridis256', low=plot_min, high=plot_max)

            def get_order(l):
                int_values = [list(map(int, re.findall(r'\d+', x)))[0] for x in l]
                ooo = np.argsort(int_values)
                return l[ooo]

            x_range = get_order(np.unique(df_prep.x.values))
            y_range = get_order(np.unique(df_prep.y.values))[::-1]

            hm = figure(title=f"{cname} Function", tools=TOOLS,
                        toolbar_location='below',
                        x_axis_location="above",
                        x_range=x_range,
                        y_range=y_range,
                        sizing_mode="scale_width",
                        # id=f"exp_{exp_name}_{cname}_model_{select_model_id}_goal_{goal}",
                        tooltips=[('Value', f'@{cname}'), ('prob', '@prob')])

            x_tick = np.copy(x_range)
            x_tick[goal[0]-1] += "_goal"
            hm.xaxis.major_label_overrides = {x: y for x, y in zip(x_range, x_tick)}
            y_tick = np.copy(y_range)
            y_tick[goal[1]-1] += "_goal"
            hm.yaxis.major_label_overrides = {x: y for x, y in zip(y_range, y_tick)}

            hm.rect(x="x", y="y", source=ColumnDataSource(df_prep), width=1, height=1,
                    line_color=None, fill_color=transform(cname, mapper))

            color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                                 ticker=BasicTicker(desired_num_ticks=30),
                                 formatter=PrintfTickFormatter(format="%.3f"))

            hm.add_layout(color_bar, 'right')
            return hm

        # ==========================================================================================

        df_prep = df_select[~((df_select["pos_x"] == goal[0]) & (df_select["pos_y"] == goal[1]))]

        xs, ys = get_directions(df_prep)
        p1 = figure(x_range=(1, pos_max_x+1), y_range=(pos_max_y+1, 1))
        p1.multi_line(xs, ys, color="#0000FF", line_width=2, line_alpha=1.0)

        def dot(xx):
            return [np.array([x[0]-0.05, x[0]+0.05]) for x in xx]

        p1.multi_line(dot(xs), dot(ys), color="#FF0000", line_width=2, line_alpha=1.0)
        # p1.y_range.flipped = True
        p1.xaxis.ticker.desired_num_ticks = pos_max_x+1
        p1.yaxis.ticker.desired_num_ticks = pos_max_y+1
        plots.append(p1)

        # ==========================================================================================

        plots.append(plot_content("entropy"))
        plots.append(plot_content("vpred"))
        # ==========================================================================================

        grid = gridplot([plots])

        # models_plots.append(grid)
        # experiments[exp_name] = models_plots
        return grid

    gmean_x = (goal_min_x + goal_max_x) // 2
    gmean_y = (goal_min_y + goal_max_y) // 2

    global crt_exp
    crt_exp = list(experiments.keys())[0]
    cl_layout = row([get_model(crt_exp, 0, (gmean_x, gmean_y))])
    coeff = 10

    max_model = get_num_models(crt_exp) * coeff

    # -- Plot Reward
    plot_r = figure(title=f"Reward", plot_width=400, plot_height=400, tools=TOOLS,
                    # sizing_mode="scale_width",
                    tooltips=[('Reward', '@reward'), ('Episode', '@ep')])

    def get_ep_r(crt_exp):
        ep, r = [], []
        eval_r = []
        for k, v in experiments[crt_exp]["other_info"].items():
            ep.append(v["epoch"])
            r.append(v["eprew"])
            eval_r.append(v.get("eval_r", 0))
        d = dict({"ep": ep, "reward": r, "eval_r": eval_r})
        return d

    reward_column = ColumnDataSource(data=get_ep_r(crt_exp))
    plot_r.line(x="ep", y="reward", source=reward_column)
    plot_r.line(x="ep", y="eval_r", source=reward_column)

    loss_span = Span(
        location=0,
        dimension="height",
        line_dash="4 4",
        line_width=1,
        name="span",
    )
    plot_r.add_layout(loss_span)

    # -- Interactive MENU
    select_exp = Select(title="Experiment:", value=crt_exp, options=list(experiments.keys()))
    model_id_slider = Slider(start=coeff, end=max_model, value=coeff, step=coeff, title="Training step")
    goal_x_slider = Slider(start=goal_min_x, end=goal_max_x, value=gmean_x, step=1, title="Goal x")
    goal_y_slider = Slider(start=goal_min_y, end=goal_max_y, value=gmean_y, step=1, title="Goal y")

    norm_types = ["entire_training", "per_training_step"]
    select_norm = Select(title="Normalize data:", value=norm_types[0], options=norm_types)

    def update_data(attrname, old, new):
        global crt_exp

        # Get the current slider values
        if select_exp.value != crt_exp:
            crt_exp = select_exp.value
            if experiments[crt_exp] is None:
                experiments[crt_exp] = read_plot_data(paths[crt_exp])

            model_id_slider.end = get_num_models(crt_exp) * coeff
            model_id_slider.value = min(model_id_slider.end, model_id_slider.value)

            plot_data = experiments[crt_exp]
            goals = plot_data["goals"]
            goal_min_x, goal_max_x = min(goals[:, 0]), max(goals[:, 0])
            goal_min_y, goal_max_y = min(goals[:, 1]), max(goals[:, 1])
            goal_x_slider.start = goal_min_x
            goal_x_slider.end = goal_max_x
            goal_y_slider.start = goal_min_y
            goal_y_slider.end = goal_max_y

            reward_column.data = get_ep_r(crt_exp)

        new_grid = get_model(crt_exp, (model_id_slider.value / coeff) - 1 ,
                             (goal_x_slider.value, goal_y_slider.value),
                             norm_type=select_norm.value)

        cl_layout.children.clear()
        cl_layout.children.append(new_grid)

        loss_span.location = model_id_slider.value

    for w in [goal_x_slider, model_id_slider, select_exp,  goal_y_slider, select_norm]:
        w.on_change('value', update_data)

    some_info = Div(text="Hover for more info. Action probabilities prob[0-Left, 1-Up, "
                         "2-Right, 1-Down ...]")

    view_layout = column([select_exp, model_id_slider, goal_x_slider, goal_y_slider,
                          select_norm,
                          cl_layout,
                          some_info, plot_r])

    curdoc().add_root(view_layout)
    curdoc().title = "4room experiments"


plot_values(["e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+0.npy",
             "e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+1.npy",
             "e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+2.npy",
             "e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+3.npy",
             "e16_freeMove_Rcst_CD_ag_1.1_goal_8.8+4.npy"])


# if __name__ == "__main__":
#     pass
    # from liftoff import parse_opts
    #
    # opts = parse_opts()
    #
    # run(opts)
