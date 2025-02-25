import gymnasium as gym
import os
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from dataclasses import dataclass

from ppo import PPO
from network import NN
#from agent import AGENT

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
        self.total_timesteps = 100_000
        self.learning_rate = 0.0004755
        self.gamma = 0.99421
        self.clip = 0.2 #0.286
        self.clip_epsilon = 0.2 # for critic_loss
        self.local_num_envs = 1 #2
        self.num_minibatches = 4 # the number of minibatches for one updating
        self.update_epochs = 16 # 4 - oridinally -- n_updates_per_iteration
        self.max_timesteps_per_episode = 1600
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process #4096 // self.local_num_envs  # before updating - timesteps_per_batch / local_num_envs
        self.max_grad_norm: float = 0.5
        self.ent_coef: float = 0.01
        self.gae_lambda: float = 0.95
        #self.device_ids: List[int] = field(default_factory=lambda: [])
        #self.backend = "gloo"  # "nccl" # gloo
        #init in runtime
        self.world_size = 1
        #local_batch_size: int = 0
        #local_minibatch_size: int = 0
        self.num_envs: int = 0
        self.batch_size: int = 2048
        #minibatch_size: int = 0
        self.num_iterations: int = 0

cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
#torch.cuda.set_per_process_memory_fraction(0.85)  # Использовать не более 85% памяти

writer = SummaryWriter(
    log_dir=r"C:\Users\user\Python ML\PyTorch\BipedalWalker_ppo_PTorch\logs_ppo\ppo_pt_bipedal_walker_logs"
)




def save_frames_as_gif(frames, path='./', filename='bipedalWalker_ppo_PT_post_training.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def train(env_name, device=torch.device('cpu'), writer=None):
    """
    Trains the model.

    Parameters:
      env_name - the name of the environment to train on
      hyperparameters - a dict of hyperparameters to use, defined in main

    Return:
      None
    """
    print(f"Training", flush=True)  # Immediate printing
    print(device)

    # Создаём среду внутри функции
    env = gym.make(env_name, render_mode='rgb_array')

    model = PPO(policy_class=NN, env=env, device=device, writer=writer, **vars(args))

    frames, total_reward = model.learn()

    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

    # Закрываем среду и освобождаем память
    #env.close()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    #import multiprocessing
    args = Args()
    # Устанавливаем метод запуска процессов
    #multiprocessing.set_start_method('spawn', force=True)
    #agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0], **vars(args))

    env_name = "BipedalWalker-v3"
    train(env_name=env_name, device=device, writer=writer)
