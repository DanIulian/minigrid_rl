import numpy as np
import pandas as pd
import torch
import cv2

columns = ["obs", "action", "reward", "done", "next_obs", "probs", "pred_full_state"]


def view_full_state(name, state):
    state = np.clip(state * (255//15), 0, 255)
    state = cv2.resize(state, (0, 0), fx=20, fy=20, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, state)


def play_game():
    import gym
    import gym_minigrid
    from gym_minigrid.wrappers import FullyObsWrapper
    from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
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
        # print(env.get_next_action_cnt())
        # print(p)
        action = np.random.choice(actions, p=p)
        # action = np.argmax(p)
        # print(action)

        obs, reward, done, info = env.step(action)
        # goal_visible = ('green', 'goal') in Grid.decode(obs['image'])
        # agent_sees_goal = env.agent_sees(*goal_pos)
        # assert agent_sees_goal == goal_visible

        if done:
            env.reset()

        full_view_img = env.render('rgb_array')
        cv2.imshow("full_view_img", full_view_img)

        view_full_state("obs", obs["image"])
        view_full_state("full state", obs["state"])
        cv2.waitKey(0)


def main():
    file_path = "/media/andrei/Samsung_T51/2019Mar15-150540_world_envs" \
                "/3_env_MiniGrid_UnlockPickup_v0/0/eval/eval_900.npy"

    data = np.load(file_path).item()
    columns = data["columns"]
    transitions = data["transitions"]

    df = pd.DataFrame(transitions, columns=columns)

    env_id = 0

    for i in range(200):
        state = np.array(df.iloc[i]["obs"][env_id]["state"])

        pred = df.iloc[i]["pred_full_state"][env_id]

        pred_state = (pred * 15).numpy().transpose(1, 2, 0).astype(np.uint8)

        view_full_state("Full state", state)
        view_full_state("Pred state", pred_state)

        next_obs = np.array(df.iloc[i]["obs"][env_id]["image"])
        pred_obs = df.iloc[i]["obs_predic"][env_id]
        pred_obs = (pred_obs * 15).numpy().transpose(1, 2, 0).astype(np.uint8)

        view_full_state("Next obs", next_obs)
        view_full_state("pred_obs", pred_obs)
        cv2.waitKey(0)
        print(df.iloc[i]['action'][env_id])

