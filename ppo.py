import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning import LightningModule
#from multiprocessing import Pool, cpu_count
import time


class PPO(LightningModule):
    def __init__(self, policy_class, env_name, device, writer, **hyperparameters):
        super(PPO, self).__init__()
        # Make sure the environment is compatible with our code
        #assert isinstance(env.observation_space, gym.spaces.Box)
        #assert isinstance(env.action_space, gym.spaces.Box)

        self._device = device
        self.writer = writer

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)
        #self.save_hyperparameters(ignore=["device", "writer"])  # Save hyperparameters for Lightning

        # Environment setup
        self.env = gym.make(env_name)
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # Actor and Critic networks
        self.actor = policy_class(self.n_observations, self.n_actions).to(self._device)
        self.critic = policy_class(self.n_observations, 1).to(self._device)

        # Optimizers
        #self.learning_rate = hyperparameters['learning_rate']
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # PPO-specific parameters
        #self.gamma = hyperparameters['gamma']
        #self.clip = hyperparameters['clip']
        #self.timesteps_per_batch = hyperparameters['timesteps_per_batch']
        #self.max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
        #self.n_updates_per_iteration = hyperparameters['n_updates_per_iteration']

        # Covariance matrix for action sampling
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5).to(self._device)
        self.cov_mat = torch.diag(self.cov_var).to(self._device)

        # This logger will help us with printing out summaries of each iteration
        '''self.logger = {
            'delta_t': time.time_ns(),
            'timesteps_current': 0,
            'iterations_current': 0,
            'batch_lens': [],
            'batch_rewards': [],  # episodic returns
            'actor_losses': [],  # losses of actor network in current iteration
        }'''
        print('delta_t', time.time_ns())
        print('timesteps_current', 0)
        print('iterations_current', 0)


        #print('batch_lens_mean', avg_lens, on_step=False, on_epoch=True)
        #print('batch_rewards_mean', avg_rewards, on_step=False, on_epoch=True)
        #print('actor_losses_mean', avg_actor_losses, on_step=True, on_epoch=True)

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self, observation):
        mean = self.actor(observation)
        return MultivariateNormal(mean, self.cov_mat)

    def rollout(self):
        batch_observations, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_discounted_rewards, batch_lens = [], [], []
        current_timestep = 0

        while current_timestep < self.timesteps_per_batch:
            observation, _ = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float, device=self._device)
            episode_rewards = []

            timestep_start = current_timestep

            for _ in range(self.max_timesteps_per_episode):
                current_timestep += 1
                batch_observations.append(observation)

                with torch.no_grad():
                    action, log_prob = self.get_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                observation = torch.tensor(observation, dtype=torch.float, device=self._device)

                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if terminated or truncated:
                    break

            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)

        batch_observations = torch.stack(batch_observations).to(self._device)
        batch_actions = torch.stack(batch_actions).to(self._device)
        batch_log_probs = torch.stack(batch_log_probs).to(self._device)
        batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards).to(self._device)

        return batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens

    def compute_discounted_rewards(self, batch_rewards):
        batch_discounted_rewards = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_discounted_rewards.insert(0, discounted_reward)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float)

    def get_action(self, observation):
        distribution = self(observation)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def evaluate(self, batch_observations, batch_actions):
        V = self.critic(batch_observations).squeeze()
        mean = self.actor(batch_observations)
        distribution = MultivariateNormal(mean, self.cov_mat)
        log_probs = distribution.log_prob(batch_actions)
        return V, log_probs

    def training_step(self, batch, batch_idx):
        batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, _ = batch
        V, current_log_probs = self.evaluate(batch_observations, batch_actions)

        # Calculate advantage
        A_k = batch_discounted_rewards - V.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # Calculate actor loss
        ratios = torch.exp(current_log_probs - batch_log_probs)
        clip_loss = torch.min(ratios * A_k, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k)
        actor_loss = -clip_loss.mean()

        #check for mseLoss
        print(f"V shape: {V.shape}, batch_discounted_rewards shape: {batch_discounted_rewards.shape}")
        if V.shape != batch_discounted_rewards.shape:
            batch_discounted_rewards = batch_discounted_rewards.view_as(V)
        # Calculate critic loss
        critic_loss = nn.MSELoss()(V, batch_discounted_rewards)

        # Manual optimization
        self.actor_optim.zero_grad()  # Use self.actor_optim
        self.manual_backward(actor_loss)
        self.actor_optim.step()

        self.critic_optim.zero_grad()  # Use self.critic_optim
        self.manual_backward(critic_loss)
        self.critic_optim.step()

        self.log('actor_loss', actor_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('critic_loss', critic_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return actor_loss + critic_loss

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
            hyperparameters

        Return:
            None
        """
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1200
        self.n_updates_per_iteration = 15
        self.learning_rate = 0.000429
        self.gamma = 0.994
        self.clip = 0.28
        self.save_freq = 10

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def configure_optimizers(self):
        return [self.actor_optim, self.critic_optim]

    def on_epoch_end(self):
        self.batch_lens = []
        self.batch_rewards = []
        self.actor_losses = []
        avg_lens = np.mean(self.batch_lens) if self.batch_lens else 0
        avg_rewards = np.mean(self.batch_rewards) if self.batch_rewards else 0
        avg_actor_losses = np.mean(self.actor_losses) if self.actor_losses else 0
        print(
            f"Epoch {self.current_epoch}: batch_lens_mean={avg_lens}, batch_rewards_mean={avg_rewards}, actor_losses_mean={avg_actor_losses}")
