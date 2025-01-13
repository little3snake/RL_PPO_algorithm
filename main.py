import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
#from torch.utils.tensorboard import SummaryWriter
#import torch.backends.cudnn as cudnn
from ppo import PPO
from network import NN

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

#cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
#torch.cuda.set_per_process_memory_fraction(0.85)  # Использовать не более 85% памяти

#writer = SummaryWriter(
#    log_dir=r"C:\Users\user\Python ML\PyTorch\BipedalWalker_ppo_PTorch\logs_ppo\ppo_pt_bipedal_walker_logs"
#)
logger = TensorBoardLogger("logs", name="ppo_training")


def save_frames_as_gif(frames, path='./', filename='bipedalWalker_ppo_PT_post_training.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


'''def train(env_name, hyperparameters, device=torch.device('cpu'), writer=None):
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
    #env = gym.make(env_name, render_mode='rgb_array')

    model = PPO(policy_class=NN, env=env, device=device, writer=writer, **hyperparameters)

    frames, total_reward = model.learn(total_timesteps=500_000)

    save_frames_as_gif(frames)
    print('total reward trained model =', total_reward)

    # Закрываем среду и освобождаем память
    #env.close()
    torch.cuda.empty_cache()'''


if __name__ == "__main__":
    #Optim hyperparams (find via optuna)
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 1600,
        'gamma': 0.99421,
        'n_updates_per_iteration': 18,
        'learning_rate': 0.0004755,
        'clip': 0.286,
        'save_freq': 1000,
    }

    env_name = "BipedalWalker-v3"
    env = gym.make(env_name, render_mode='rgb_array')
    policy = PPO(policy_class=NN, env=env, writer=None, **hyperparameters)

    trainer = pl.Trainer(logger=logger, max_epochs=5, log_every_n_steps=50,
                         accelerator="cpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(policy)
    env.close()
    # Generating GIF
    frames, total_reward = policy.generate_gif_after_training()
    save_frames_as_gif(frames)
    print('Total reward with trained model:', total_reward)
