''' Dan Iulian Muntean 2020
    Generic agent for policy evaluation
'''

from gym_minigrid.wrappers import *

import torch
import torch.nn

import utils.episodic_memory as ep_mem
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tqdm


class BonusRewardWrapper(object):
    """Environment wrapper that adds additional curiosity reward."""
    def __init__(self,
                 envs,
                 r_model,
                 device,
                 observation_preprocess_fn):

        self._r_model = r_model
        self._device = device
        self._observation_preprocess_fn = observation_preprocess_fn

        self._similarity_threshold = 0.9
        # Create an episodic memory for each env

        self._episodic_memory = ep_mem.EpisodicMemory(512, self._r_model.forward_similarity,
                                                        self._device, "fifo", 200)


    def _compute_curiosity_reward(self, observation, info, done):
        """Compute intrinsic curiosity reward.
        The reward is set to 0 when the episode is finished
        """

        frame = self._observation_preprocess_fn(observation, device=self._device)
        embedded_observation = self._r_model.forward(frame)

        similarity_to_memory = ep_mem.similarity_to_memory(embedded_observation,  self._episodic_memory)

        # Updates the episodic memory of every environment.
        # If we've reached the end of the episode, resets the memory
        # and always adds the first state of the new episode to the memory.
        if done:
            self._episodic_memories.reset()
            self._episodic_memories.add(embedded_observation.cpu(), info)


        # Only add the new state to the episodic memory if it is dissimilar
        # enough.
        if similarity_to_memory < self._similarity_threshold:
            self._episodic_memories.add(embedded_observation.cpu(), info)

        # Augment the reward with the exploration reward.
        bonus_rewards = [ 0 if done else 0.5 - similarity_to_memory ]
        bonus_rewards = np.array(bonus_rewards)

        return bonus_rewards

    def reset(self):

        observation = super(BonusRewardWrapper, self).reset()
        self._episodic_memory.reset()

        return observation

    def step(self, action):

        obs, reward, done, info = super(BonusRewardWrapper, self).step(action)

        # Exploration bonus.
        bonus_rewards = self._compute_curiosity_reward(obs, info, done)

        return obs, (np.array(reward), bonus_rewards), done, info

    def get_episodic_memeories(self):
        return self._episodic_memory


class RGBImgWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['rendered_image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'rendered_image': rgb_img
        }


class RGBImgPartialWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['rendered_image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'rendered_image': rgb_img_partial
        }


class EvalAgent(object):

    def __init__(self, envs,
                 model,
                 r_model,
                 obs_preprocess_fn,
                 save_dir,
                 nr_steps,
                 nr_runs,
                 argmax=False,
                 view_type="FullView"):

        self._model = model
        self._view_type = view_type
        self._save_dir = save_dir
        self._obs_preprocess_fn = obs_preprocess_fn
        self._argmax = argmax
        self._nr_runs = nr_runs
        self._nr_steps = nr_steps

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a new env
        head_env = envs[0][0] if isinstance(envs[0], list) else envs[0]
        env_name = head_env.spec.id
        self._eval_env = gym.make(env_name)
        self._eval_env.action_space.n = head_env.action_space.n
        self._eval_env.max_steps = head_env.max_steps
        self._eval_env.seed(np.random.randint(1, 10000))

        if self._view_type == "FullView":
            self._eval_env = RGBImgWrapper(self._eval_env)
        elif self._view_type == "AgentView":
            self._eval_env = RGBImgPartialWrapper(self._eval_env)
        else:
            raise ValueError("Incorrect view name: {}".format(self._view_type))

        #self._eval_env = BonusRewardWrapper(self._eval_env, r_model, self._device, self._obs_preprocess_fn)

        if self._model.recurrent:
            self._memories = torch.zeros(1, self._model.memory_size, device=self._device)

        self._eval_path = f"{self._save_dir}/eval_episodes"
        if not os.path.exists(self._eval_path):
            os.mkdir(self._eval_path)

        self._step_count = 0

        print("Argmax - {}".format(self._argmax))

    def _run_episode(self, nr_runs, nr_steps):

        print("Evaluating agent after {} for {} episodes".format(nr_steps, nr_runs))
        self._model.eval()

        #import pdb; pdb.set_trace()
        for i in tqdm.tqdm(range(nr_runs)):
            obs = self._eval_env.reset()
            human_obs = obs["rendered_image"]
            episode_obs = [human_obs]

            while True:
                #preprocess observation
                preprocessed_obs = self._obs_preprocess_fn([obs], device=self._device)

                with torch.no_grad():
                    if self._model.recurrent:
                        dist, _, self._memories = self._model(preprocessed_obs, self._memories)
                    else:
                        dis, _ = self._model(preprocessed_obs)

                if self._argmax:
                    action, _ = dist.probs.max(1, keepdim=True)
                else:
                    action = dist.sample()

                obs, reward, done, info = self._eval_env.step(action.cpu().numpy())

                human_obs = obs["rendered_image"]
                episode_obs.append(human_obs)
                if done:
                    break

            self._save_episode(episode_obs, i, nr_steps)

        self._model.train()

    def _save_episode(self, episode_obs, run_no, nr_steps):

        eval_dir = f"{self._eval_path}/{str(nr_steps)}"
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        curr_ep_dir = f"{eval_dir}/episode_{str(run_no)}"
        if not os.path.exists(curr_ep_dir):
            os.mkdir(curr_ep_dir)

        for i, img in enumerate(episode_obs):
            plt.imsave(f"{curr_ep_dir}/image_{i}.png", img, format='png')

    def on_new_observation(self,
                           unused_obs=None,
                           unused_rewards=None,
                           unused_dones=None,
                           unused_infos=None):

        self._step_count += 1

        if self._nr_steps == -1:
            return

        if self._step_count % self._nr_steps == 0:
            self._run_episode(self._nr_runs, self._step_count)

