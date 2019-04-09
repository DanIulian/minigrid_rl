import pandas as pd
import os
import random
import torch
import numpy as np
import cv2
from typing import List
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from analytics.state_decoder import StateDecoder


FONT_SCALE = 1
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (0, 255, 0)


def view_full_state(name: str, state: np.ndarray,
                    show: bool=True, scale: int =20, put_name: bool=False,
                    state_decoder: StateDecoder =None, full_state: bool=False) -> np.ndarray:
    """
        Visualize observations / states and show text on image
        Can be used with a StateDecoder - for human understandable render
    """

    if state_decoder is not None:
        if full_state:
            agent_locations = np.array(np.where(state == 15)).T
            agent_location = agent_locations[:, :2][0]
            state = state_decoder.get_state_render(state, agent_location)
        else:
            state = state_decoder.get_obs_render(state)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    else:
        state = np.clip(state * (255//15), 0, 255)
        state = cv2.resize(state, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if put_name:
        cv2.putText(state, name, (10, 10), FONT, FONT_SCALE, FONT_COLOR)

    if show:
        cv2.imshow(name, state)

    return state


def put_text(menu_text: List[str], img: np.ndarray, scale: float =1.) -> None:
    """ Show multi line text on image """
    y0, dy = int(10 * scale), int((25 * scale))

    for i, line in enumerate(menu_text):
        y = int(y0 + i * dy)
        cv2.putText(img, line, (50, y), FONT, FONT_SCALE, FONT_COLOR)


def draw_exp(step: int, states: np.ndarray, obs: np.ndarray, actions: np.ndarray,
             show_pre: int = 3, show_post: int = 3, scale: int = 40,
             text: str = None, state_decoder: StateDecoder = None) -> np.ndarray:

    # Render state
    state = view_full_state("state", states[step], show=False, scale=scale,
                            state_decoder=state_decoder, full_state=True)

    # Calculate max possible view obs (given batch)
    # TODO Not checking for same episode
    show_pre = step - max(step - show_pre, 0)
    show_post = min(step + show_post, len(obs) - 1) - step
    obs_size = show_pre + show_post + 1

    # Calculate state for observations
    h, w, _ = state.shape
    obs_h, obs_w, _ = obs[0].shape
    small_scale = int(scale / obs_size)

    # Batch observations
    all_obs = []
    pad = None
    fill_value = 200

    for i in range(step - show_pre, step + show_post + 1):
        all_obs.append(view_full_state(f"s:{i}-a:{actions[i]}", obs[i], show=False,
                                       scale=small_scale,
                                       put_name=True,
                                       state_decoder=state_decoder, full_state=False))
        if pad is None:
            new_h, _, _ = all_obs[0].shape
            pad = np.zeros((new_h, 2, 3), dtype=np.uint8)
            pad.fill(fill_value)

        all_obs.append(pad)
    obs = np.hstack(all_obs)

    # Stack with state
    obs_h, obs_w, _ = obs.shape
    if w > obs_w:
        pad = np.zeros((obs_h, w-obs_w, 3), dtype=np.uint8)
        pad.fill(fill_value)
        obs = np.hstack([obs, pad])
    elif w < obs_w:
        pad = np.zeros((h, obs_w-w, 3), dtype=np.uint8)
        pad.fill(fill_value)
        state = np.hstack([state, pad])

    img = np.vstack([state, obs])

    # Show text on full image
    if text is None:
        text = []
    elif isinstance(text, str):
        text = [text]

    put_text([f"{step}"] + text, img)
    return img


def play_game():
    import gym
    from utils.gym_wrappers import RecordFullState, ExploreActions

    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env.action_space.n = 6
    env.max_steps = 1000
    env = RecordFullState(env)
    env = ExploreActions(env)
    # OBS (OBJECT TYPE IDX, COLOR, INCLUDES)
    actions = np.arange(0, env.action_space.n)
    env.reset()

    for i in range(0, 1000):
        action = np.random.randint(0, env.action_space.n - 1)
        p = env.get_new_action_prob()
        action = np.random.choice(actions, p=p)

        obs, reward, done, info = env.step(action)

        if done:
            env.reset()

        full_view_img = env.render('rgb_array')
        cv2.imshow("full_view_img", full_view_img)

        view_full_state("obs", obs["image"])
        view_full_state("full state", obs["state"])
        cv2.waitKey(0)


def visualize_embeddings(file_path):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    data = np.load(file_path).item()
    columns = data["columns"]
    transitions = data["transitions"]

    df = pd.DataFrame(transitions, columns=columns)
    no_envs = len(df.iloc[0]["obs"])
    sq = int(np.sqrt(no_envs))
    steps = len(df)

    # for i in range(steps):
    z = torch.zeros_like(df.iloc[0]["eval_ag_memory"])
    df.iloc[-1]["eval_ag_memory"] = z

    envs_ag_states = []
    for env_id in range(no_envs):
        env_ag_state = []
        for i in range(steps):
            state = np.array(df.iloc[i]["obs"][env_id]["state"])
            env_ag_state.append(state)

        env_ag_state = np.stack(env_ag_state)
        envs_ag_states.append(env_ag_state)
    envs_ag_states = np.stack(envs_ag_states)

    coord = np.where(envs_ag_states[:, :, :, :, 0] == 15)

    loc = coord[:, 2] * 8 + coord[:, 3]
    loc_env = loc.reshape(4, 400, 1)
    # coord = np.stack(coord).T
    # ag_v = (envs_ag_states[envs_ag_states[:, :, :, :, 0] == 15])
    # ag = np.array([coord[0][0], coord[1][0], ag_v[1], ag_v[2]])

    x = torch.cat([x.unsqueeze(0) for x in df["eval_ag_memory"].values.tolist()], dim=0)

    loc_uniq = np.unique(loc_env)
    max_cl = len(loc_uniq)
    cmap = matplotlib.cm.get_cmap('Spectral')

    fig = plt.figure()

    for env_id in range(no_envs):
        x_embedded = TSNE(n_components=2).fit_transform(x[:, env_id].numpy())
        l = loc_env[env_id]
        # plt.subplot(sq, sq, env_id + 1)
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(x_embedded)):
            ax.scatter(x_embedded[i, 0], x_embedded[i, 1], i, c=clr)
        plt.show()

        for color_idx in range(max_cl):
            s = x_embedded[(l == loc_uniq[color_idx])[:,0]]
            if len(s) == 0:
                continue

            clr = cmap((color_idx+1)/max_cl)

            # ax = fig.add_subplot(111, projection='3d')

            plt.scatter(s[:, 0], s[:, 1], s[:, 2], c=clr)
    plt.show()


def play_experience(file_path: str, env_id: int = 0, build_website: bool = True,
                    show_pre: int = 3, show_post: int = 3):
    from analytics.make_site import make_website

    data = torch.load(file_path)
    data = data.__dict__

    columns = list(data.keys())

    for column in columns:
        if isinstance(data[column], torch.Tensor):
            data[column] = data[column].to("cpu")

    kobs = "obs_image"
    kstates = "states"
    kaction = "action"
    kgap = "gaps"
    kintrin_pre = "dst_intrinsic_r_pre"
    kintrin_post = "dst_intrinsic_r_post"

    steps = len(data[kobs])
    no_envs = len(data[kobs][0])
    frames, procs = data[kstates].shape[:2]
    max_df = steps

    data[kobs] = (data[kobs] * 15).numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    data[kstates] = (data[kstates] * 15).numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

    state_size = data[kstates][0][0].shape[:2]

    i = 0

    # ----------------------------------------------------------------------------------------------
    # Config state decoder
    state_decoder = StateDecoder(state_size)

    # Get agent locations
    agent_locations = np.array(np.where(data[kstates] == 15)).T
    agent_locations = agent_locations[:, 2:4]
    agent_locations = agent_locations.reshape(frames, procs, 2)

    agent_states = data[kstates].copy()
    agent_states[agent_states == 15] = 0

    # ----------------------------------------------------------------------------------------------
    # Generate folders to save data
    if build_website:
        out_folder = f"{file_path}_analysis"
        img_out = f"{file_path}_analysis/imgs"
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        if not os.path.isdir(img_out):
            os.mkdir(img_out)

        # ----------------------------------------------------------------------------------------------
        # Element types

        etypes = argparse.Namespace()
        etypes.str = "str"
        etypes.img = "img"
        etypes.img_path = "img_path"

        # ----------------------------------------------------------------------------------------------
        # Show reward histogram
        elements = []

        def save_fig(out):
            plt.show(block=False)
            plt.pause(1)
            plt.savefig(out)
            plt.close()

        bins = 40
        num_examples = 10

        # Intrinsic reward pre return norm

        intrisic_pre = data[kintrin_pre].numpy()
        hist_ir_pre = np.histogram(intrisic_pre.reshape(-1), bins=bins)

        hist_pre_path = f"{out_folder}/hist_ir_pre.png"
        plt.hist(intrisic_pre.reshape(-1), bins=bins)
        plt.title("Histogram intrinsic pre return norm")
        save_fig(hist_pre_path)
        elements += [(etypes.str, "Histogram intrinsic pre return norm"),
                     (etypes.str, f"{hist_ir_pre}"),
                     (etypes.img_path, "hist_ir_pre.png")]

        # Intrinsic reward post return norm
        intrisic_post = data[kintrin_post].numpy()
        hist_ir_post = np.histogram(intrisic_post.reshape(-1), bins=bins)
        hist_post_path = f"{out_folder}/hist_ir_post.png"
        plt.hist(intrisic_post.reshape(-1), bins=bins)
        plt.title("Histogram intrinsic post return norm")
        save_fig(hist_post_path)
        elements += [(etypes.str, "Histogram intrinsic post return norm"),
                     (etypes.str, f"{hist_ir_post}"),
                     (etypes.img_path, "hist_ir_post.png")]

        # Add examples experience for post
        hist = hist_ir_post
        r = intrisic_post

        for i in range(len(hist[0]))[::-1]:
            elements += [(etypes.str, f"EXAMPLES for bin {i}")]

            bin_start, bin_end = hist[1][i], hist[1][i+1]
            if i == len(hist[0]) - 1:
                bin_end += 0.001  # Last bin includes margin

            no_elem = hist[0][i]

            elements += [(etypes.str, f"Bin start: {bin_start}\nBin end: {bin_end}\n"
                                      f"No elem: {no_elem}")]

            include = np.where((r >= bin_start) & (r < bin_end))
            include = list(zip(include[0].tolist(), include[1].tolist()))
            random.shuffle(include)

            select = include[:num_examples]
            elements += [(etypes.str, f"Selected (f, p): {select}")]

            for s, p in select:
                img = draw_exp(s, data[kstates][:, p], data[kobs][:, p], data[kaction][:, p],
                               show_pre=show_pre, show_post=show_post, state_decoder=state_decoder)
                elements += [(etypes.img, img)]

        make_website(out_folder, elements)

    # ----------------------------------------------------------------------------------------------
    # Play experience

    while i < steps:
        print(f"_______________Step {i}___________________")
        obs_batch = data[kobs][i, env_id]
        state = data[kstates][i, env_id]
        next_obs = data[kobs][i+1, env_id]

        view_full_state("Full state", state, state_decoder=state_decoder, full_state=True)
        view_full_state("CRT_obs", obs_batch, state_decoder=state_decoder, full_state=False)
        view_full_state("Next obs", next_obs, state_decoder=state_decoder, full_state=False)

        if i < max_df:
            action = data[kaction][i, env_id].item()
            intrin_pre = data[kintrin_pre][i, env_id].item()
            intrin_post = data[kintrin_post][i, env_id].item()
            gap = data[kgap][i]
            print(f"[T_Act] {action:.2f} [T_Gap] {gap:.2f}")
            print(f"[T_IPr] {intrin_pre:.8f}")
            print(f"[T_IPo] {intrin_post:.8f}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("b"):
            i -= 2

        i += 1


def main(file_path, env_id=0):

    data = np.load(file_path).item()
    columns = data["columns"]
    transitions = data["transitions"]

    df = pd.DataFrame(transitions, columns=columns)

    state_size = np.array(df.iloc[0]["obs"][env_id]["state"]).shape[:2]

    state_decoder = StateDecoder(state_size)

    no_envs = len(df.iloc[0]["obs"])
    steps = len(df)
    max_df = len(df)
    i = 0

    while i < steps:
        print(f"_______________Step {i}___________________")
        state = np.array(df.iloc[i]["obs"][env_id]["state"])
        obs = np.array(df.iloc[i]["obs"][env_id]["image"])

        pred = df.iloc[i]["pred_full_state"][env_id]
        obs_batch = df.iloc[i]["obs_batch"][env_id]

        pred_state = (pred * 15).numpy().transpose(1, 2, 0).astype(np.uint8)
        obs_batch = (obs_batch * 15).numpy().transpose(1, 2, 0).astype(np.uint8)

        view_full_state("Full state", state, state_decoder=state_decoder, full_state=True)
        view_full_state("Pred state", pred_state, state_decoder=state_decoder, full_state=True)

        next_obs = np.array(df.iloc[i]["next_obs"][env_id]["image"])
        pred_diff = (df.iloc[i]["obs_predict"][env_id].numpy().transpose(1, 2, 0))
        pred_obs = pred_diff + obs.astype(np.float) / 15.
        pred_obs = pred_obs * 15
        pred_obs = pred_obs.astype(np.uint8)

        view_full_state("obs", obs, state_decoder=state_decoder, full_state=False)
        view_full_state("Next obs", next_obs, state_decoder=state_decoder, full_state=False)
        view_full_state("pred_diff", pred_diff)
        view_full_state("pred_obs", pred_obs, state_decoder=state_decoder, full_state=False)

        if i < max_df:
            action = df.iloc[i]['action'][env_id].item()
            if df.iloc[i]['pred_act'] is not None:
                pred_act = df.iloc[i]['pred_act'][env_id].numpy() * 5
                print(f"[T_A] {action:.2f} [A_p] {pred_act[action]:.2f}")
                print(f"[M_a] {pred_act.argmax():.2f} [M_p] {pred_act.max():.2f}")
                print("Pred:", pred_act.tolist())

        key = cv2.waitKey(0) & 0xFF
        if key == ord("b"):
            i -= 2

        i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path')
    parser.add_argument('--env-id', type=int, default=0)
    parser.add_argument('--exp', action='store_false', default=True)
    parser.add_argument('--website', action='store_false', default=True)
    args = parser.parse_args()

    if args.exp:
        play_experience(args.path, args.env_id, build_website=args.website)
    else:
        main(args.path, args.env_id)
