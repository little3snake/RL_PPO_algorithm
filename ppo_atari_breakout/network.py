import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# network is a natural cnn

def create_layer (layer, std = np.sqrt(2), bias_for_cr_l = 0.0):
  torch.nn.init.orthogonal_(layer.weight, std)  # std is standard deviation
  torch.nn.init.constant_(layer.bias, bias_for_cr_l)
  return layer

class NN(nn.Module):
  def __init__(self, envs):
    super(NN, self).__init__()

    self.net = nn.Sequential(
      create_layer(nn.Conv2d(4, 32, 8, stride=4)),
      nn.ReLU(),
      create_layer(nn.Conv2d(32, 64, 4, stride=2)),
      nn.ReLU(),
      create_layer(nn.Conv2d(64, 64, 3, stride=1)),
      nn.ReLU(),
      nn.Flatten(), # from 2... dim to 1 dim
      create_layer(nn.Linear(64*7*7, 512)),
      nn.ReLU(),
    )
    self.envs = envs
    self.actor = create_layer(nn.Linear(512, self.envs.action_space.n), std = 0.01)
    self.critic  = create_layer(nn.Linear(512, 1), std = 1)
