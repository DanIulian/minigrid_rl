''' Dan Iulian Muntean 2020
    Generic agent for policy evaluation
'''
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import torch
import torch.nn

import utils.episodic_memory as ep_mem
import numpy as np



class BonusRewardWrapper(object):
    """Environment wrapper that adds additional curiosity reward."""
    def __init__(self,
                 envs,
                 r_model,
                 device,
                 observation_preprocess_fn):

        super(BonusRewardWrapper, self).__init__(envs)

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

        self.observation_space.spaces['image'] = spaces.Box(
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
            'image': obs,
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
        self.observation_space.spaces['image'] = spaces.Box(
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
            'image': obs,
            'rendered_image': rgb_img_partial
        }


class EvalAgent(object):

    def __init__(self, envs, model, r_model, obs_preprocess_fn, argmax=False, view_type="FullView",  save_dir=None):

        self._model = model
        self._view_type = view_type
        self._save_dir = save_dir
        self._obs_preprocess_fn = obs_preprocess_fn
        self._argmax = argmax

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a new env
        head_env = envs[0][0] if isinstance(envs[0], list) else envs[0]
        env_name = head_env.spec.id
        self._eval_env = gym.make(env_name)
        self._eval_env.action_space.n = head_env.action_space.n
        self._eval_env.max_steps = head_env.max_steps
        self._eval_env.seed(np.randint(1, 10000))

        if self._view_type == "FullView":
            self._eval_env = RGBImgWrapper(self._eval_env)
        elif self._view_type == "AgentView":
            self._eval_env = RGBImgPartialWrapper(self._eval_env)
        else:
            raise ValueError("Incorrect view name: {}".format(self._view_type))

        self._eval_env = BonusRewardWrapper(self._eval_env, r_model, self._device, self._obs_preprocess_fn)

        if self._model.recurrent:
            self._memories = torch.zeros(1, self._model.memory_size)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        if torch.cuda.is_available():
            actions = actions.cpu().numpy()

        return actions

    def run_episode(self):

        self._model.eval()

        with torch.no_grad():
            obs = self._eval_env.reset()

            while True:



        self._model.train()

        obs = self._eval_env.reset()




    def save_episode(self):
        pass




def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    import pdb; pdb.set_trace()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=False)