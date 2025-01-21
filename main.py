import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from ppo import PPOLightning

hyperparameters = {
    'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 1600,
    'gamma': 0.99421,
    'n_updates_per_iteration': 1,  # Уменьшили количество обновлений до 1 (было 18)
    'learning_rate': 0.0004755,
    'clip': 0.286,
    'save_freq': 10, #1000
}
torch.autograd.set_detect_anomaly(True)

env_name = "BipedalWalker-v3"
env = gym.make(env_name, render_mode='rgb_array')

model = PPOLightning(env, hyperparameters)

# TensorBoard логгер
logger = pl.loggers.TensorBoardLogger(save_dir="logs/", name="ppo_bipedal_walker")

trainer = pl.Trainer(
    max_epochs=500, #5
    logger=logger,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=1, # 50
    enable_checkpointing=False,  # Отключаем ModelCheckpoint
)
trainer.fit(model)

env.close()
