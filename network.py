"""
  This file contains a neural network module for us to
  define our actor and critic networks in PPO.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(NN, self).__init__()
    self.layer1 = nn.Linear(input_dim, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, output_dim)

  def forward(self, n_observations):
    x = F.tanh(self.layer1(n_observations))
    x = F.tanh(self.layer2(x))
    return self.layer3(x)
