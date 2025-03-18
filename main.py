import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List

import torch
import random
import numpy as np
import tyro
from rich.pretty import pprint
import matplotlib.pyplot as plt
from matplotlib import animation
import torch.distributed as dist
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
from network import NN

@dataclass
class Args:
    def __init__(self):
        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.seed = 1
        self.torch_deterministic = True # fixed order - in input1 = input2 -> output1 = output2
        self.cuda = True
        self.track = True # tracking metrics (in wandb)
        self.wandb_project_name = "PPO_PyTorch_distributed"
        self.wandb_entity = None # the name of the user/group
        #capture_gif = False
        self.capture_video = False
        self.env_name = "BipedalWalker-v3"
        self.total_timesteps = 1_500_000
        self.learning_rate = 0.0004755#0.0004755#87104 # * world_size
        self.anneal_lr = False
        self.gamma = 0.99421#0.988
        self.clip = 0.286#0.2566
        self.local_num_envs = 2 #2
        self.num_minibatches = 4 # the number of minibatches for one updating
        self.update_epochs = 14#14 # 4 - oridinally -- n_updates_per_iteration
        self.max_timesteps_per_episode = 1600
        self.num_episodes_per_batch_in_one_process = 2
        self.num_steps = self.max_timesteps_per_episode * self.num_episodes_per_batch_in_one_process #4096 // self.local_num_envs  # before updating - timesteps_per_batch / local_num_envs
        self.max_grad_norm: float = 0.5
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95
        #self.device_ids: List[int] = field(default_factory=lambda: [])
        self.backend = "gloo"  # "nccl" # gloo
        #init in runtime
        self.world_size = 1
        #local_batch_size: int = 0
        #local_minibatch_size: int = 0
        self.num_envs: int = 0
        self.batch_size: int = 0
        #minibatch_size: int = 0
        self.num_iterations: int = 0



def init_distributed(args):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn("Running in non-distributed mode!")
    args.local_batch_size = int(args.local_num_envs * args.num_steps)  # for each process
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)  # for gradient descent
    args.num_envs = args.local_num_envs * args.world_size  # total number of envs
    args.batch_size = int(args.num_envs * args.num_steps)  # total batch size (total envs * num_steps_per_1_iter)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # total minibatch size
    args.num_iterations = args.total_timesteps // args.batch_size
    return local_rank

def make_env(env_name, id, capture_video, run_name):
    def thunk ():
        if id == 0:
            env = gym.make(env_name, render_mode="rgb_array")
        else:
            env = gym.make(env_name, render_mode="rgb_array")
        return env
    return thunk

def save_frames_as_gif(frames, path='./', filename='bipedalWalker_ppo_PT_post_training.gif'):
    if isinstance(frames, tuple):
        frames = frames[0]
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

if __name__ == "__main__":
    args = Args()
    local_rank = init_distributed(args)
    #device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"  # for loggind and saving results

    if local_rank == 0:
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,#True
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                #resume=True, # to continue writing
                save_code=True,
                settings=wandb.Settings(code_dir="."),  # add all files in dir
            )
            # Определяем, какие метрики будут использоваться как step для других метрик
            wandb.define_metric("global_step")
            wandb.define_metric("iteration")
            wandb.define_metric("global_step_world_size")

            # Указываем, какие метрики используют какие step
            wandb.define_metric("charts/reward", step_metric="global_step")
            wandb.define_metric("charts/allreducetime_persentage_in_batchtime", step_metric="global_step")
            wandb.define_metric("charts/allreducetime_persentage_in_itertime", step_metric="global_step")
            wandb.define_metric("charts/episode_len", step_metric="global_step")
            wandb.define_metric("charts/current_step", step_metric="global_step")
            wandb.define_metric("charts/actor_learning_rate", step_metric="global_step")
            wandb.define_metric("charts/reward_new", step_metric="iteration")
            wandb.define_metric("charts/reward_new_new", step_metric="global_step_world_size")

        writer = SummaryWriter(f"runs/{args.exp_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None
    device_count = torch.cuda.device_count()
    if device_count < args.world_size:
        #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        device = "cpu"
    else:
        #device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")
        device = "cpu"

    # Set unique seed per process
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)  # for matrix - or in ppo.py
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    print ("process ", local_rank, " with seed ", args.seed, " with device ", device, flush=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name) for i in range(args.local_num_envs)],
    )

    model = PPO(policy_class=NN, envs=envs, device=device, rank=local_rank, writer=writer, **vars(args))

    start_time = time.time()
    frames, total_reward = model.learn()
    end_time = time.time()
    print (f"total time of {args.total_timesteps} steps in {args.world_size} processes: {(end_time - start_time):.4f}")
    if local_rank == 0:
        #save_frames_as_gif(frames)
        print('total reward trained model =', total_reward)

    envs.close()
    if local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()

    if args.world_size > 1:
        dist.destroy_process_group()
