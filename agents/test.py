{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> Soft Actor Critic Demystified</h1>\n",
    "<h4> By Vaishak Kumar </h4>\n",
    "<br>\n",
    "<a href=\"https://arxiv.org/pdf/1801.01290.pdf\">Original Paper</a>\n",
    "<br> \n",
    "<a href=\"https://github.com/higgsfield/RL-Adventure-2\">Adapted from higgsfield's implementation</a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "import random\n",
    "\n",
    "import gym\n",
    "import numpy as np\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torch.nn.functional as F\n",
    "from torch.distributions import Normal\n",
    "\n",
    "from IPython.display import clear_output\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import animation\n",
    "from IPython.display import display\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "use_cuda = torch.cuda.is_available()\n",
    "device   = torch.device(\"cuda\" if use_cuda else \"cpu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Auxilliary Functions</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ReplayBuffer:\n",
    "    def __init__(self, capacity):\n",
    "        self.capacity = capacity\n",
    "        self.buffer = []\n",
    "        self.position = 0\n",
    "    \n",
    "    def push(self, state, action, reward, next_state, done):\n",
    "        if len(self.buffer) < self.capacity:\n",
    "            self.buffer.append(None)\n",
    "        self.buffer[self.position] = (state, action, reward, next_state, done)\n",
    "        self.position = (self.position + 1) % self.capacity\n",
    "    \n",
    "    def sample(self, batch_size):\n",
    "        batch = random.sample(self.buffer, batch_size)\n",
    "        state, action, reward, next_state, done = map(np.stack, zip(*batch))\n",
    "        return state, action, reward, next_state, done\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.buffer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "class NormalizedActions(gym.ActionWrapper):\n",
    "    def _action(self, action):\n",
    "        low  = self.action_space.low\n",
    "        high = self.action_space.high\n",
    "        \n",
    "        action = low + (action + 1.0) * 0.5 * (high - low)\n",
    "        action = np.clip(action, low, high)\n",
    "        \n",
    "        return action\n",
    "\n",
    "    def _reverse_action(self, action):\n",
    "        low  = self.action_space.low\n",
    "        high = self.action_space.high\n",
    "        \n",
    "        action = 2 * (action - low) / (high - low) - 1\n",
    "        action = np.clip(action, low, high)\n",
    "        \n",
    "        return actions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot(frame_idx, rewards):\n",
    "    clear_output(True)\n",
    "    plt.figure(figsize=(20,5))\n",
    "    plt.subplot(131)\n",
    "    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))\n",
    "    plt.plot(rewards)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Network Definitions</h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ValueNetwork(nn.Module):\n",
    "    def __init__(self, state_dim, hidden_dim, init_w=3e-3):\n",
    "        super(ValueNetwork, self).__init__()\n",
    "        \n",
    "        self.linear1 = nn.Linear(state_dim, hidden_dim)\n",
    "        self.linear2 = nn.Linear(hidden_dim, hidden_dim)\n",
    "        self.linear3 = nn.Linear(hidden_dim, 1)\n",
    "        \n",
    "        self.linear3.weight.data.uniform_(-init_w, init_w)\n",
    "        self.linear3.bias.data.uniform_(-init_w, init_w)\n",
    "        \n",
    "    def forward(self, state):\n",
    "        x = F.relu(self.linear1(state))\n",
    "        x = F.relu(self.linear2(x))\n",
    "        x = self.linear3(x)\n",
    "        return x\n",
    "        \n",
    "        \n",
    "class SoftQNetwork(nn.Module):\n",
    "    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):\n",
    "        super(SoftQNetwork, self).__init__()\n",
    "        \n",
    "        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)\n",
    "        self.linear2 = nn.Linear(hidden_size, hidden_size)\n",
    "        self.linear3 = nn.Linear(hidden_size, 1)\n",
    "        \n",
    "        self.linear3.weight.data.uniform_(-init_w, init_w)\n",
    "        self.linear3.bias.data.uniform_(-init_w, init_w)\n",
    "        \n",
    "    def forward(self, state, action):\n",
    "        x = torch.cat([state, action], 1)\n",
    "        x = F.relu(self.linear1(x))\n",
    "        x = F.relu(self.linear2(x))\n",
    "        x = self.linear3(x)\n",
    "        return x\n",
    "        \n",
    "        \n",
    "class PolicyNetwork(nn.Module):\n",
    "    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):\n",
    "        super(PolicyNetwork, self).__init__()\n",
    "        \n",
    "        self.log_std_min = log_std_min\n",
    "        self.log_std_max = log_std_max\n",
    "        \n",
    "        self.linear1 = nn.Linear(num_inputs, hidden_size)\n",
    "        self.linear2 = nn.Linear(hidden_size, hidden_size)\n",
    "        \n",
    "        self.mean_linear = nn.Linear(hidden_size, num_actions)\n",
    "        self.mean_linear.weight.data.uniform_(-init_w, init_w)\n",
    "        self.mean_linear.bias.data.uniform_(-init_w, init_w)\n",
    "        \n",
    "        self.log_std_linear = nn.Linear(hidden_size, num_actions)\n",
    "        self.log_std_linear.weight.data.uniform_(-init_w, init_w)\n",
    "        self.log_std_linear.bias.data.uniform_(-init_w, init_w)\n",
    "        \n",
    "    def forward(self, state):\n",
    "        x = F.relu(self.linear1(state))\n",
    "        x = F.relu(self.linear2(x))\n",
    "        \n",
    "        mean    = self.mean_linear(x)\n",
    "        log_std = self.log_std_linear(x)\n",
    "        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)\n",
    "        \n",
    "        return mean, log_std\n",
    "    \n",
    "    def evaluate(self, state, epsilon=1e-6):\n",
    "        mean, log_std = self.forward(state)\n",
    "        std = log_std.exp()\n",
    "        \n",
    "        normal = Normal(0, 1)\n",
    "        z      = normal.sample()\n",
    "        action = torch.tanh(mean+ std*z.to(device))\n",
    "        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)\n",
    "        return action, log_prob, z, mean, log_std\n",
    "        \n",
    "    \n",
    "    def get_action(self, state):\n",
    "        state = torch.FloatTensor(state).unsqueeze(0).to(device)\n",
    "        mean, log_std = self.forward(state)\n",
    "        std = log_std.exp()\n",
    "        \n",
    "        normal = Normal(0, 1)\n",
    "        z      = normal.sample().to(device)\n",
    "        action = torch.tanh(mean + std*z)\n",
    "        \n",
    "        action  = action.cpu()#.detach().cpu().numpy()\n",
    "        return action[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> Update Function </h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def update(batch_size,gamma=0.99,soft_tau=1e-2,):\n",
    "    \n",
    "    state, action, reward, next_state, done = replay_buffer.sample(batch_size)\n",
    "\n",
    "    state      = torch.FloatTensor(state).to(device)\n",
    "    next_state = torch.FloatTensor(next_state).to(device)\n",
    "    action     = torch.FloatTensor(action).to(device)\n",
    "    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)\n",
    "    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)\n",
    "\n",
    "    predicted_q_value1 = soft_q_net1(state, action)\n",
    "    predicted_q_value2 = soft_q_net2(state, action)\n",
    "    predicted_value    = value_net(state)\n",
    "    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)\n",
    "\n",
    "    \n",
    "    \n",
    "# Training Q Function\n",
    "    target_value = target_value_net(next_state)\n",
    "    target_q_value = reward + (1 - done) * gamma * target_value\n",
    "    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())\n",
    "    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())\n",
    "\n",
    "\n",
    "    soft_q_optimizer1.zero_grad()\n",
    "    q_value_loss1.backward()\n",
    "    soft_q_optimizer1.step()\n",
    "    soft_q_optimizer2.zero_grad()\n",
    "    q_value_loss2.backward()\n",
    "    soft_q_optimizer2.step()    \n",
    "# Training Value Function\n",
    "    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))\n",
    "    target_value_func = predicted_new_q_value - log_prob\n",
    "    value_loss = value_criterion(predicted_value, target_value_func.detach())\n",
    "\n",
    "    \n",
    "    value_optimizer.zero_grad()\n",
    "    value_loss.backward()\n",
    "    value_optimizer.step()\n",
    "# Training Policy Function\n",
    "    policy_loss = (log_prob - predicted_new_q_value).mean()\n",
    "\n",
    "    policy_optimizer.zero_grad()\n",
    "    policy_loss.backward()\n",
    "    policy_optimizer.step()\n",
    "    \n",
    "    \n",
    "    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):\n",
    "        target_param.data.copy_(\n",
    "            target_param.data * (1.0 - soft_tau) + param.data * soft_tau\n",
    "        )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> Initializations </h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/vaishak/anaconda3/lib/python3.6/site-packages/gym/envs/registration.py:14: PkgResourcesDeprecationWarning: Parameters to load are deprecated.  Call .resolve and .require separately.\n",
      "  result = entry_point.load(False)\n"
     ]
    }
   ],
   "source": [
    "env = NormalizedActions(gym.make(\"Pendulum-v0\"))\n",
    "\n",
    "action_dim = env.action_space.shape[0]\n",
    "state_dim  = env.observation_space.shape[0]\n",
    "hidden_dim = 256\n",
    "\n",
    "value_net        = ValueNetwork(state_dim, hidden_dim).to(device)\n",
    "target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)\n",
    "\n",
    "soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)\n",
    "soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)\n",
    "policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)\n",
    "\n",
    "for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):\n",
    "    target_param.data.copy_(param.data)\n",
    "    \n",
    "\n",
    "value_criterion  = nn.MSELoss()\n",
    "soft_q_criterion1 = nn.MSELoss()\n",
    "soft_q_criterion2 = nn.MSELoss()\n",
    "\n",
    "value_lr  = 3e-4\n",
    "soft_q_lr = 3e-4\n",
    "policy_lr = 3e-4\n",
    "\n",
    "value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)\n",
    "soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)\n",
    "soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)\n",
    "policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)\n",
    "\n",
    "\n",
    "replay_buffer_size = 1000000\n",
    "replay_buffer = ReplayBuffer(replay_buffer_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training Hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "max_frames  = 40000\n",
    "max_steps   = 500\n",
    "frame_idx   = 0\n",
    "rewards     = []\n",
    "batch_size  = 128"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training Loop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "while frame_idx < max_frames:\n",
    "    state = env.reset()\n",
    "    episode_reward = 0\n",
    "    \n",
    "    for step in range(max_steps):\n",
    "        if frame_idx >1000:\n",
    "            action = policy_net.get_action(state).detach()\n",
    "            next_state, reward, done, _ = env.step(action.numpy())\n",
    "        else:\n",
    "            action = env.action_space.sample()\n",
    "            next_state, reward, done, _ = env.step(action)\n",
    "        \n",
    "        \n",
    "        replay_buffer.push(state, action, reward, next_state, done)\n",
    "        \n",
    "        state = next_state\n",
    "        episode_reward += reward\n",
    "        frame_idx += 1\n",
    "        \n",
    "        if len(replay_buffer) > batch_size:\n",
    "            update(batch_size)\n",
    "        \n",
    "        if frame_idx % 1000 == 0:\n",
    "            plot(frame_idx, rewards)\n",
    "        \n",
    "        if done:\n",
    "            break\n",
    "        \n",
    "    rewards.append(episode_reward)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> Visualize Trained Algorithm </h2> - <a href=\"http://mckinziebrandon.me/TensorflowNotebooks/2016/12/21/openai.html\">source</a>  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_frames_as_gif(frames):\n",
    "    \"\"\"\n",
    "    Displays a list of frames as a gif, with controls\n",
    "    \"\"\"\n",
    "    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)\n",
    "    patch = plt.imshow(frames[0])\n",
    "    plt.axis('off')\n",
    "\n",
    "    def animate(i):\n",
    "        patch.set_data(frames[i])\n",
    "\n",
    "    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)\n",
    "    display(anim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/vaishak/anaconda3/lib/python3.6/site-packages/gym/envs/registration.py:14: PkgResourcesDeprecationWarning: Parameters to load are deprecated.  Call .resolve and .require separately.\n",
      "  result = entry_point.load(False)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.animation.FuncAnimation at 0x7f9d95f3b898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "env = gym.make(\"Pendulum-v0\")\n",
    "\n",
    "# Run a demo of the environment\n",
    "state = env.reset()\n",
    "cum_reward = 0\n",
    "frames = []\n",
    "for t in range(50000):\n",
    "    # Render into buffer. \n",
    "    frames.append(env.render(mode = 'rgb_array'))\n",
    "    action = policy_net.get_action(state)\n",
    "    state, reward, done, info = env.step(action.detach())\n",
    "    if done:\n",
    "        break\n",
    "env.close()\n",
    "display_frames_as_gif(frames)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}