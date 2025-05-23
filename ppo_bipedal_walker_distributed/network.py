import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)

class NN(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(NN, self).__init__()
    self.layer1 = nn.Linear(input_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, output_dim)
    self.apply(init_weights)

  def forward(self, n_observations):
    x = F.relu(self.layer1(n_observations))
    x = F.relu(self.layer2(x))
    return self.layer3(x)




