import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import sys
from liftoff import parse_opts
import os
import glob
import cv2
import itertools
import pandas as pd
import re
from typing import List

from train_main import run as train_run


"""
sys.argv = ['evaluate_model.py', 'results/2020Mar01-144543_default/0000_default/0/cfg.yaml', '--session-id', '1']

opts = parse_opts()
"""

AGENT_ID = 10
DIR_NAMES = ["right", "down", "left", "up"]


def img_view(img: np.ndarray, scale: int = 1, view: str = None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if view:
        cv2.imshow(view, img)
    return img


def run(opts):
    algo, model, envs, saver = train_run(opts, return_models=True)
    device = algo.device
    model_paths = glob.glob(f"{opts.out_dir}/checkpoints/training_data*")
    model_ids = [int(re.findall("training_data_(.*).pt", x)[0]) for x in model_paths]
    model_ids.sort()

    env = envs[0][0]

    env_obs = env.observation(env.gen_obs())
    obs = env_obs["image"]

    env.unwrapped.agent_pos = [3,1]
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
    all_states_render = []
    states_conf = np.array(list(itertools.product(range(8), range(8), range(4))))
    for pos_x, pos_y, dir in states_conf:
        env.unwrapped.agent_pos = [pos_x, pos_y]
        env.unwrapped.agent_dir = dir
        env_obs = env.observation(env.gen_obs())
        img_render = env.render('rgb_array')
        all_states.append(env_obs)
        all_states_render.append(img_render)

    # -- Get values for states
    models_data = dict({})
    for model_id in model_ids:
        model, agent_data, other_data = saver.load_training_data(best=False, index=model_id)

        prep_obs = algo.preprocess_obss(all_states, device=device)
        dist, vpred, memory = model(prep_obs, None)

        data = {
            'pos_x': states_conf[:, 0],
            'pos_y': states_conf[:, 1],
            'dir': states_conf[:, 2],
            'vpred': vpred.data.cpu().numpy(),
            'prob': list(dist.probs.data.cpu().numpy()),
            'entropy': list(dist.entropy().data.cpu().numpy()),
        }

        df = pd.DataFrame(data)
        df["model_id"] = model_id
        models_data[model_id] = df

    df = pd.concat(models_data.values())

    # -- Plot

    plot_data = {"df": df, "model_ids": model_ids}
    np.save("plot_test", plot_data)
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
    from bokeh.layouts import column, gridplot
    from bokeh.models import CustomJS, Slider
    from bokeh.models.widgets import Select, Div, RadioGroup

    experiments = dict()
    for df_path in df_paths:

        exp_name = os.path.splitext(os.path.basename(df_path))[0]
        plot_data = np.load(df_path, allow_pickle=True).item()
        df = plot_data["df"]
        model_ids = plot_data["model_ids"]

        models_plots = []
        for select_model_id in model_ids:
            plots = []
            for direction in range(4):
                df_prep = df[df.model_id == select_model_id]
                plot_min, plot_max = df_prep.vpred.min(), df_prep.vpred.max()

                df_prep = df_prep[df_prep.dir == direction].copy()
                df_prep["x"] = df_prep.pos_x.values.astype(np.str)
                df_prep["y"] = df_prep.pos_y.values.astype(np.str)

                colors = ["#75968f",  "#550b1d"]
                mapper = LinearColorMapper(palette='Viridis256', low=plot_min, high=plot_max)
                TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

                hm = figure(title=f"Direction {direction} ({DIR_NAMES[direction]})", tools=TOOLS,
                            toolbar_location='below',
                            x_axis_location="above",
                            x_range=np.unique(df_prep.x.values),
                            y_range=np.unique(df_prep.y.values)[::-1],
                            sizing_mode="scale_width",
                            id=f"dir_{direction}_model_{select_model_id}",
                            tooltips=[('Value', '@vpred')])

                hm.rect(x="x", y="y", source=df_prep, width=1, height=1,
                        line_color=None, fill_color=transform('vpred', mapper))

                color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                                     ticker=BasicTicker(desired_num_ticks=30),
                                     formatter=PrintfTickFormatter(format="%.3f"))

                hm.add_layout(color_bar, 'right')
                plots.append(hm)

            # , plot_width=250, plot_height=250)
            grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])

            models_plots.append(grid)
            experiments[exp_name] = models_plots

    global crt_exp
    crt_exp = list(experiments.keys())[0]
    cl_layout = column([experiments[crt_exp][0]])

    select_exp = Select(title="Experiment:", value=crt_exp, options=list(experiments.keys()))
    model_id_slider = Slider(start=0, end=len(experiments[crt_exp])-1, value=0, step=1,
                             title="Model ID")

    def update_data(attrname, old, new):
        global crt_exp

        # Get the current slider values
        if select_exp.value != crt_exp:
            crt_exp = select_exp.value
            model_id_slider.end = len(experiments[crt_exp])-1
            model_id_slider.value = min(model_id_slider.end, model_id_slider.value)
        cl_layout.children[0] = experiments[select_exp.value][model_id_slider.value]

    for w in [model_id_slider, select_exp]:
        w.on_change('value', update_data)

    div = Div(text="""Agent has direction.""", width=200, height=100)

    view_layout = column([select_exp, model_id_slider, cl_layout])

    curdoc().add_root(view_layout)
    curdoc().title = "4room experiments"


plot_values(["plot_test.npy"])


# if __name__ == "__main__":
#     pass
    # from liftoff import parse_opts
    #
    # opts = parse_opts()
    #
    # run(opts)
