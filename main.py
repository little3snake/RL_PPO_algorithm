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
        self.total_timesteps = 1_000_000
        self.learning_rate = 0.0006104
        self.anneal_lr: bool = True
        self.gamma = 0.988
        self.clip = 0.2566 #0.286
        self.clip_epsilon = 0.2 # for critic_loss
        self.local_num_envs = 1 #2
        self.num_minibatches = 4 # the number of minibatches for one updating
        self.n_updates_per_iteration = 7 # 4 - oridinally --update_epochs
        self.max_timesteps_per_episode = 1600
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process #4096 // self.local_num_envs  # before updating - timesteps_per_batch / local_num_envs
        self.max_grad_norm: float = 0.5
        self.entropy_coef: float = 0.01
        self.gae_lambda: float = 0.95
        #self.device_ids: List[int] = field(default_factory=lambda: [])
        #self.backend = "gloo"  # "nccl" # gloo
        #init in runtime
        self.world_size = 1
        #local_batch_size: int = 0
        #local_minibatch_size: int = 0
        self.num_envs: int = 0
        self.timesteps_per_batch: int = 3200 # - origionally --batch_size
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

    #model = PPO(policy_class=NN, env=env, device=device, writer=writer, **hyperparameters)
    model = PPO(policy_class=NN, env=env, device=device, writer=writer, **vars(args))
    frames, total_reward = model.learn(total_timesteps=args.total_timesteps)
    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

    # Закрываем среду и освобождаем память
    #env.close()
    #torch.cuda.empty_cache()


if __name__ == "__main__":
    args = Args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project="PPO_PyTorch",
        entity=None,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,  # for loggind and saving results,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(device=device, writer=writer)


'''import gymnasium as gym
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
import optuna

cudnn.benchmark = True

@dataclass
class Args:
    def __init__(self):
        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.seed = 1
        self.torch_deterministic = True  # fixed order - in input1 = input2 -> output1 = output2
        #self.cuda = True
        self.track = False  # tracking metrics (in wandb)
        self.wandb_project_name = "PPO_PyTorch_distributed"
        self.wandb_entity = None  # the name of the user/group
        self.capture_video = False
        self.env_name = "BipedalWalker-v3"
        self.total_timesteps = 100_000
        self.learning_rate = 0.0006104
        self.anneal_lr: bool = True
        self.gamma = 0.988
        self.clip = 0.2566  # 0.286
        self.clip_epsilon = 0.2  # for critic_loss
        self.local_num_envs = 1  # 2
        self.num_minibatches = 4  # the number of minibatches for one updating
        self.n_updates_per_iteration = 14  # 4 - originally --update_epochs
        self.max_timesteps_per_episode = 1600
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process  # 4096 // self.local_num_envs
        self.max_grad_norm: float = 0.5
        self.entropy_coef: float = 0.01
        self.gae_lambda: float = 0.95
        self.world_size = 1
        self.num_envs: int = 0
        self.timesteps_per_batch: int = 2048  # originally --batch_size
        self.num_iterations = self.total_timesteps // self.timesteps_per_batch


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
    #save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

    return total_reward


def objective(trial):
    # Определяем гиперпараметры, которые будем оптимизировать
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    args.timesteps_per_batch = trial.suggest_categorical("timesteps_per_batch", [3200, 4800, 6400, 8000])
    args.gamma = trial.suggest_float("gamma", 0.9, 0.999)
    args.clip = trial.suggest_float("clip", 0.1, 0.4)
    args.clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.4)
    args.entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1)
    args.gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    args.num_minibatches = trial.suggest_int("num_minibatches", 2, 8)
    args.n_updates_per_iteration = trial.suggest_int("n_updates_per_iteration", 4, 20)
    args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.5)

    # Запускаем обучение и возвращаем среднее вознаграждение
    total_reward = train(device=device, writer=None)
    return total_reward


if __name__ == "__main__":
    args = Args()

    # Создаем study для оптимизации
    study = optuna.create_study(direction="maximize")  # Мы хотим максимизировать вознаграждение
    study.optimize(objective, n_trials=50)

    # Выводим результаты
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Используем лучшие гиперпараметры для финального обучения
    for key, value in trial.params.items():
        setattr(args, key, value)

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project="PPO_PyTorch",
        entity=None,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(device=device, writer=writer)'''
