import gymnasium as gym
import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter
import wandb
from dataclasses import dataclass
import torch.backends.cudnn as cudnn
from ppo import PPO
from network import NN


cudnn.benchmark = True

@dataclass
class Args:
    def __init__(self):
        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.seed = 1
        self.torch_deterministic = True # fixed order - in input1 = input2 -> output1 = output2
        self.cuda = True
        self.track = False # tracking metrics (in wandb)
        self.wandb_project_name = "PPO_PyTorch_distributed"
        self.wandb_entity = None # the name of the user/group
        #capture_gif = False
        self.capture_video = False
        self.env_name = "BipedalWalker-v3"
        self.total_timesteps = 3_500_000
        self.learning_rate = 0.0008104
        self.anneal_lr: bool = True
        self.gamma = 0.988
        self.clip = 0.2566 #0.286
        self.clip_epsilon = 0.2 # for critic_loss - useless
        self.local_num_envs = 1 #2
        self.num_minibatches = 4
        self.n_updates_per_iteration = 14 # 4 --update_epochs
        self.max_timesteps_per_episode = 1600
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process
        self.max_grad_norm: float = 0.5
        self.ent_coef: float = 0.03
        self.gae_lambda: float = 0.95
        self.world_size = 1
        #local_batch_size: int = 0
        #local_minibatch_size: int = 0
        self.num_envs: int = 0
        self.timesteps_per_batch: int = 2048 # --batch_size
        #minibatch_size: int = 0
        self.num_iterations = self.total_timesteps // self.timesteps_per_batch


'''device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)'''
device = "cpu"

def save_frames_as_gif(frames, path='./', filename='bipedalWalker_ppo_PT_post_training.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def train(device=torch.device('cpu'), writer=None):
    print(f"Training", flush=True)  # Immediate printing
    print(device)
    env = gym.make(args.env_name, render_mode='rgb_array')
    model = PPO(policy_class=NN, env=env, device=device, writer=writer, **vars(args))
    frames, total_reward = model.learn(total_timesteps=args.total_timesteps)
    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

if __name__ == "__main__":
    args = Args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project="PPO_PyTorch",
        entity=None,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(code_dir="."),  # add all files from dir
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(device=device, writer=writer)
