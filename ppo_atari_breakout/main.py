import gymnasium as gym
import os
# if can't find roms
#os.environ["ALE_ROM_DIR"] = r"(path to your venv)\Lib\site-packages\ale_py\roms"

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter
import wandb
from dataclasses import dataclass
import torch.backends.cudnn as cudnn
from gymnasium.wrappers import AtariPreprocessing
import ale_py
from frame_stack import FrameStack  # Правильный путь
from FireResetEnv import FireResetEnv
from ppo import PPO
from network import NN

import sys
import io

cudnn.benchmark = True

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    NoopResetEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    ClipRewardEnv,
    EpisodicLifeEnv
)

#class ClipRewardEnv(gym.RewardWrapper):
#    def reward(self, reward):
#        return np.sign(reward)  # [-1, 0, 1]

#class CustomRewardEnv(gym.RewardWrapper):
#    def reward(self, reward):
#        if reward > 0:
#            return reward + 0.1  #Enhancing beneficial actions
#        elif reward < 0:
#            return reward * 0.5  # alleviating penalties for lose
#        return reward  # neutral actions

@dataclass
class Args:
    def __init__(self):
        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.seed = 1
        self.torch_deterministic = True # fixed order - in input1 = input2 -> output1 = output2
        self.cuda = True
        self.track = True # tracking metrics (in wandb)
        self.wandb_project_name = "Atari_PPO_PyTorch"
        self.wandb_entity = None # the name of the user/group
        self.capture_gif = True # capture gif after learning
        self.env_name = "ALE/Breakout-v5"
        self.frame_stack = 4
        self.total_timesteps = 1_000_000
        self.learning_rate = 2.5e-4 #0.0008876 # ordinary 2.5e-4 #0.0008104
        self.anneal_lr: bool = True
        self.gamma = 0.99
        self.clip = 0.1 # originally 0.2
        self.clip_epsilon = 0.2 # for critic_loss - useless
        self.local_num_envs = 8
        self.num_minibatches = 4
        self.n_updates_per_iteration = 4 #10
        self.max_timesteps_per_episode = 128 #512 if terminal_on_life_loss=False
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process
        self.max_grad_norm = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.world_size = 1
        #local_batch_size: int = 0
        #local_minibatch_size: int = 0
        self.num_envs: int = 0
        self.timesteps_per_batch: int = 1024 # --batch_size
        #minibatch_size: int = 0
        self.num_iterations = self.total_timesteps // self.timesteps_per_batch


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def save_frames_as_gif(frames, path='./', filename='atari_ppo_PT_post_training.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def make_env():
    env = gym.make(args.env_name,
                   render_mode="rgb_array",
                   frameskip=1, # frameskip set up in atari preprocessing
                   repeat_action_probability=0,
                   full_action_space=False
                   )
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = AtariPreprocessing(env,
                             frame_skip=4,
                             #noop_max=0,
                             screen_size=84,
                             terminal_on_life_loss=True,
                             grayscale_obs=True,
                             scale_obs=False
                             )
    #env = CustomRewardEnv(env)  # or ClipRewardEnv(env)
    env = FrameStack(env, num_stack=args.frame_stack)

    return env

def train(device=torch.device('cpu'), writer=None):
    print(f"Training", flush=True)  # Immediate printing
    print(device)
    env = make_env()
    model = PPO(policy_class=NN, env=env, device=device, writer=writer, **vars(args))
    frames, total_reward = model.learn(total_timesteps=args.total_timesteps)
    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

if __name__ == "__main__":
    args = Args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        entity=None,
        sync_tensorboard=True, # True
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(code_dir="."),  # add all files from dir
    )
    print(gym.envs.registry.keys())
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(device=device, writer=writer)