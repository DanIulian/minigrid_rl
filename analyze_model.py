import numpy as np
import os
import glob
import itertools
import pandas as pd
import re
from typing import List


"""
sys.argv = ['evaluate_model.py', '/media/andrei/CE04D7C504D7AF292/rl/minigrid_rl/results/2020Mar02-151641_vf_ppo-multiple-envss/0003_env_cfg.env_args.goal_rand_offset_3/0/cfg.yaml', '--session-id', '1']

opts = parse_opts()
"""

AGENT_ID = 10
DIR_NAMES = ["right", "down", "left", "up"]
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


def run(opts):
    import cv2
    from train_main import run as train_run

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

    env_obs = env.observation(env.gen_obs())
    obs = env_obs["image"]

    env.unwrapped._goal_default_pos = [8, 5]
    env.reset()

    env.unwrapped.agent_pos = [0,0]
    env.unwrapped.agent_dir = 3
    img_render = env.render('rgb_array')
    agent_pos = env.unwrapped.agent_pos
    goal_pos = env.unwrapped._goal_default_pos
    print(obs[tuple(agent_pos)])

    img_view(img_render, view="Test")
    cv2.waitKey(0)

    # ==============================================================================================
    # -- Run model for all states

    # -- Get states
    all_states = []
    # all_states_render = []
    states_conf = np.array(list(itertools.product(range(-3, 4), range(-3, 4),
                                                  range(1, 9), range(1, 9), range(4))))
    states_conf[:, 0] += orig_goal_pos[0]
    states_conf[:, 1] += orig_goal_pos[1]

    crt_goal_pos = (None, None)
    for goal_x, goal_y, pos_x, pos_y, dir in states_conf:
        if (goal_x, goal_y) != crt_goal_pos:
            env.unwrapped.goal_rand_offset = 0
            env.unwrapped._goal_default_pos = [goal_x, goal_y]
            env.reset()

        env.unwrapped.agent_pos = [pos_x, pos_y]
        env.unwrapped.agent_dir = dir
        env_obs = env.observation(env.gen_obs())
        # img_render = env.render('rgb_array')
        all_states.append(env_obs)
        # all_states_render.append(img_render)

    # -- Get values for states
    models_data = dict({})
    models_other_info = dict({})
    for model_id in model_ids:
        model, agent_data, other_data = saver.load_training_data(best=False, index=model_id)

        prep_obs = algo.preprocess_obss(all_states, device=device)
        dist, vpred, memory = model(prep_obs, None)

        data = {
            'goal_x': states_conf[:, 0],
            'goal_y': states_conf[:, 1],
            'pos_x': states_conf[:, 2],
            'pos_y': states_conf[:, 3],
            'dir': states_conf[:, 4],
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
    np.save(f"CD_ag_1.1_goal_5.5+{orig_goal_rand_offset}", plot_data)

    # ==============================================================================================
    #  -- Run agent for an episode
    done, step = False, 0
    env.unwrapped.agent_pos = [3, 3]
    action, vpred = None, None
    while not done:
        print(f"Step: {step} - action: {action} - value: {vpred}")

        img_render = env.render('rgb_array')
        img_view(img_render, view="Test")
        cv2.waitKey(0)

        env_obs = env.observation(env.gen_obs())

        prep_obs = algo.preprocess_obss([env_obs], device=device)
        dist, vpred, memory = model(prep_obs, None)
        vpred = vpred[0].item()
        action = dist.sample()[0].item()

        env_obs, reward, done, info = env.step(action)


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

    def get_num_models(exp_name):
        print("SHOW LEN MODELS FOR ", exp_name)
        return len(experiments[exp_name]["model_ids"])

    def get_model(exp_name, select_model_id, goal):
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
        model_ids = plot_data["model_ids"]

        df_select["prob"] = df_select["prob"].apply(lambda x: str([np.round(y, 3) for y in x]))
        # print(df_select["prob"].iloc[0])
        # df_select["prob"] =
        plots = []
        for direction in range(4):
            plot_min, plot_max = df_select.vpred.min(), df_select.vpred.max()

            df_prep = df_select[df_select.dir == direction].copy()
            df_prep["x"] = df_prep.pos_x.values.astype(np.str)
            df_prep["y"] = df_prep.pos_y.values.astype(np.str)

            # Specify goal position
            sel = df_prep[df_prep["x"] == str(goal[0])].index
            df_prep.loc[sel, "x"] = str(goal[0]) + "_goal"

            sel = df_prep[df_prep["y"] == str(goal[1])].index
            df_prep.loc[sel, "y"] = str(goal[1]) + "_goal"

            mapper = LinearColorMapper(palette='Viridis256', low=plot_min, high=plot_max)

            x_range = np.unique(df_prep.x.values)
            y_range = np.unique(df_prep.y.values)[::-1]

            hm = figure(title=f"Direction {direction} ({DIR_NAMES[direction]})", tools=TOOLS,
                        toolbar_location='below',
                        x_axis_location="above",
                        x_range=x_range,
                        y_range=y_range,
                        sizing_mode="scale_width",
                        id=f"exp_{exp_name}_dir_{direction}_model_{select_model_id}_goal_{goal}",
                        tooltips=[('Value', '@vpred'), ('prob', '@prob')])

            x_tick = np.copy(x_range)
            x_tick[goal[0]-1] += "_goal"
            hm.xaxis.major_label_overrides = {x: y for x, y in zip(x_range, x_tick)}
            y_tick = np.copy(y_range)
            y_tick[goal[1]-1] += "_goal"
            hm.yaxis.major_label_overrides = {x: y for x, y in zip(y_range, y_tick)}

            hm.rect(x="x", y="y", source=ColumnDataSource(df_prep), width=1, height=1,
                    line_color=None, fill_color=transform('vpred', mapper))

            color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                                 ticker=BasicTicker(desired_num_ticks=30),
                                 formatter=PrintfTickFormatter(format="%.3f"))

            hm.add_layout(color_bar, 'right')
            plots.append(hm)

        # , plot_width=250, plot_height=250)
        grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])

        # models_plots.append(grid)
        # experiments[exp_name] = models_plots
        return grid

    gmean_x = (goal_min_x + goal_max_x) // 2
    gmean_y = (goal_min_y + goal_max_y) // 2

    global crt_exp
    crt_exp = list(experiments.keys())[0]
    cl_layout = row([get_model(crt_exp, 0, (gmean_x, gmean_y))])
    max_model = get_num_models(crt_exp)-1

    # -- Plot Reward
    plot_r = figure(title=f"Reward", plot_height=400, tools=TOOLS,
                    sizing_mode="scale_width",
                    tooltips=[('Reward', '@reward'), ('Episode', '@ep')])
    ep, r = [], []
    for k, v in experiments[crt_exp]["other_info"].items():
        ep.append(k)
        r.append(v["eprew"])
    reward_column = ColumnDataSource(data=dict({"ep": ep, "reward": r}))
    plot_r.line(x="ep", y="reward", source=reward_column)

    # -- Interactive MENU
    select_exp = Select(title="Experiment:", value=crt_exp, options=list(experiments.keys()))
    model_id_slider = Slider(start=0, end=max_model, value=0, step=1, title="Model ID")
    goal_x_slider = Slider(start=goal_min_x, end=goal_max_x, value=gmean_x, step=1, title="Goal x")
    goal_y_slider = Slider(start=goal_min_y, end=goal_max_y, value=gmean_y, step=1, title="Goal y")

    def update_data(attrname, old, new):
        global crt_exp

        # Get the current slider values
        if select_exp.value != crt_exp:
            crt_exp = select_exp.value
            if experiments[crt_exp] is None:
                experiments[crt_exp] = read_plot_data(paths[crt_exp])

            model_id_slider.end = get_num_models(crt_exp) - 1
            model_id_slider.value = min(model_id_slider.end, model_id_slider.value)

        new_grid = get_model(crt_exp, model_id_slider.value,
                             (goal_x_slider.value, goal_y_slider.value))

        cl_layout.children.clear()
        cl_layout.children.append(new_grid)

    for w in [goal_x_slider, model_id_slider, select_exp,  goal_y_slider]:
        w.on_change('value', update_data)

    some_info = Div(text="Hover for more info. Action probabilities prob[0-left, 1-right, "
                         "2-forward, ...]")

    view_layout = column([select_exp, model_id_slider, goal_x_slider, goal_y_slider, cl_layout,
                          some_info, plot_r])

    curdoc().add_root(view_layout)
    curdoc().title = "4room experiments"


plot_values(["CD_ag_1.1_goal_5.5+0.npy", "CD_ag_1.1_goal_5.5+1.npy", "CD_ag_1.1_goal_5.5+2.npy",
             "CD_ag_1.1_goal_5.5+3.npy"])
# plot_values(["CD_ag_1.1_goal_5.5+1.npy"])

# if __name__ == "__main__":
#     pass
    # from liftoff import parse_opts
    #
    # opts = parse_opts()
    #
    # run(opts)
