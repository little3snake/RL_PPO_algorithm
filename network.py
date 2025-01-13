"""
  This file contains a neural network module for us to
  define our actor and critic networks in PPO.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

class NN(nn.Module):
  def __init__(self, input_dim, output_dim, device):
    super(NN, self).__init__()
    self.layer1 = nn.Linear(input_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, output_dim)

    # Инициализация ковариационной матрицы для многомерного нормального распределения
    self.device = device
    print ("NN device", self.device)
    self.cov_var = torch.full(size=(output_dim,), fill_value=0.5).to(self.device)
    #self.cov_var = nn.Parameter(torch.full(size=(output_dim,), fill_value=0.5)).to(self.device)
    self.cov_mat = torch.diag(self.cov_var).to(self.device)

  def forward(self, n_observations):
    x = F.relu(self.layer1(n_observations))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

  def get_action(self, observation):
    """
    Return action and log_prob.
    """
    mean = self.forward(observation).to(self.device) # Получаем среднее значение распределения
    #print(mean)
    if torch.isnan(mean).any():
      raise ValueError("Mean contains NaN values.")
    #print(mean.device, self.cov_var.device, self.cov_mat.device)
    distribution = MultivariateNormal(mean, self.cov_mat)  # Создаем распределение
    #if torch.isnan(distribution).any():
    #  raise ValueError("Dist contains NaN values.")
    #print ("mean ", mean.device, "cov_mat ", self.cov_mat.device) #"dist device ", distribution.device)
    action = distribution.sample()  # Выбираем действие
    log_prob = distribution.log_prob(action.to(self.device))  # Логарифм вероятности действия
    return action.detach(), log_prob.detach()

  def evaluate(self, observations, actions):
    """
    Return log_prob.
    """
    mean = self.forward(observations)
    if torch.isnan(mean).any():
      print("Mean contains NaN:", mean)
    if torch.isnan(self.cov_mat).any():
      print("Covariance matrix contains NaN:", self.cov_mat)
    distribution = MultivariateNormal(mean, self.cov_mat)
    log_probs = distribution.log_prob(actions)
    return log_probs
