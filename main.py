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
    #def __init__(self):
    exp_name = os.path.basename(__file__)[: -len(".py")]
    seed = 1
    torch_deterministic = True # fixed order - in input1 = input2 -> output1 = output2
    cuda = True
    track = False # tracking metrics (in wandb)
    wandb_project_name = "PPO_PyTorch_distributed"
    wandb_entity = None # the name of the user/group
    #capture_gif = False
    capture_video = False
    env_name = "BipedalWalker-v3"
    total_timesteps = 100_000
    learning_rate = 0.0004755
    gamma = 0.99421
    clip = 0.286
    local_num_envs = 2
    num_steps = 4096// local_num_envs # before updating - timesteps_per_batch / local_num_envs
    num_minibatches = 4 # the number of minibatches for one updating
    update_epochs = 18 # 4 - oridinally -- n_updates_per_iteration
    max_timesteps_per_episode = 1600
    max_grad_norm: float = 0.5
    device_ids: List[int] = field(default_factory=lambda: [])
    backend = "gloo"  # "nccl" # gloo
    #init in runtime
    world_size = 1
    #local_batch_size: int = 0
    #local_minibatch_size: int = 0
    num_envs: int = 0
    batch_size: int = 0
    #minibatch_size: int = 0
    num_iterations: int = 0



'''def init_distributed(args):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn("Running in non-distributed mode!")
    #args.local_batch_size = int(args.local_num_envs * args.num_steps)  # for each process
    #args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)  # for gradient descent
    args.num_envs = args.local_num_envs * args.world_size  # total number of envs
    args.batch_size = int(args.num_envs * args.num_steps)  # total batch size (total envs * num_steps_per_1_iter)
    #args.minibatch_size = int(args.batch_size // args.num_minibatches)  # total minibatch size
    args.num_iterations = args.total_timesteps // args.batch_size
    return local_rank'''

def make_env(env_name, id, capture_video, run_name):
    def thunk ():
        #if capture_video and id == 0:
        if id == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        #env = gym.wrappers.GrayScaleObservation(env)
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
    args = tyro.cli(Args)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    # args.local_batch_size = int(args.local_num_envs * args.num_steps)  # for each process
    # args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)  # for gradient descent
    args.num_envs = args.local_num_envs * args.world_size  # total number of envs
    args.batch_size = int(args.num_envs * args.num_steps)  # total batch size (total envs * num_steps_per_1_iter)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)  # total minibatch size
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn("Running in non-distributed mode!")

    #args = Args()
    #local_rank = init_distributed(args)
    #device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"  # for loggind and saving results

    if local_rank == 0:
        if args.track:
            # weights and biases
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{args.exp_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        #pprint(args)
    else:
        writer = None
    #print("ARGS before PPO init:", vars(args))

    '''if len(args.device_ids) > 0:
        assert len(args.device_ids) == args.world_size, "you must specify the same number of device ids as `--nproc_per_node`"
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:'''
    #device_count = torch.cuda.device_count()
    #print ("device_count ", device_count)
    #if device_count < args.world_size:
    #    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #    #print ("if ",device)
    #else:
    #    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")
    #    #print("else ", device)

    # Set unique seed per process
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)  # for matrix - or in ppo.py
    torch.manual_seed(args.seed - local_rank) # originally - _seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_ids:
        assert len(args.device_ids) == args.world_size, "you must specify the same number of device ids as `--nproc_per_node`"
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device_count = torch.cuda.device_count()
        if device_count < args.world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")



    print ("process ", local_rank, " with seed ", args.seed, " with device ", device, flush=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name) for i in range(args.local_num_envs)],
    )
    #env = gym.make(args.env_id)
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    model = PPO(policy_class=NN, envs=envs, device=device, rank=local_rank, writer=writer, **vars(args))

    #model.learn(total_timesteps=args.total_timesteps, writer=writer)
    frames, total_reward = model.learn()
    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

    envs.close()
    if local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()

    if args.world_size > 1:
        dist.destroy_process_group()
