import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import time

from rollout_dataset import RolloutDataset

class PPO(pl.LightningModule):
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, writer=None, **hyperparameters):
        super().__init__()
        # Make sure the environment is compatible with our code
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        print ("PPO device", self.device)

        self.writer = writer
        #self.save_hyperparameters(**hyperparameters) # Initialize hyperparameters for training with PPO
        self.timesteps_per_batch = hyperparameters.get("timesteps_per_batch", 2048)
        self.max_timesteps_per_episode = hyperparameters.get("max_timesteps_per_episode", 1600)
        self.gamma = hyperparameters.get("gamma", 0.994)
        self.n_updates_per_iteration = hyperparameters.get("n_updates_per_iteration", 15)
        self.learning_rate = hyperparameters.get("learning_rate", 0.000429)
        self.clip = hyperparameters.get("clip", 0.28)
        self.save_freq = hyperparameters.get('save_freq',10)

        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.actor = policy_class(self.n_observations, self.n_actions, self.device)  # ALG STEP 1
        self.critic = policy_class(self.n_observations, 1, self.device)

        # Initialize optimizers later
        self.actor_optim = None
        self.critic_optim = None

        # Training conditions
        self.timesteps_current = 0
        self.iterations_current = 0

        # Logging
        self.logger_dict = {"actor_losses": []}

        # Initialize the covariance matrix for get_action(...), evaluate(...)
        #self.cov_mat = None
        #self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        #self.cov_mat = torch.diag(self.cov_var)

    #def on_train_start(self):
        # Move covariance matrix to device when training starts
        #self.cov_var = self.cov_var.to(self.device)
        #self.cov_mat = self.cov_mat.to(self.device)

    def compute_discounted_rewards(self, batch_rewards):
        batch_discounted_rewards = []
        for episode_rews in reversed(batch_rewards):
            discounted_reward = 0
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_discounted_rewards.insert(0, discounted_reward)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float)

    def evaluate(self, observations, actions):
        V = self.critic(observations.to(self.device)).squeeze()
        print ("obs", observations)
        print ("actor", self.actor)
        for param in self.actor.parameters():
            if torch.isnan(param).any():
                raise ValueError("Actor parameters contain NaN values.")

        print ("all", self.actor(observations))
        mean = self.actor(observations.to(self.device))
        print (mean)
        # Проверка на NaN в mean
        if torch.isnan(mean).any():
            raise ValueError("Mean contains NaN values.")
        #mean = self.actor(observations.to(self.device)) # actor on device
        #distribution = MultivariateNormal(mean, self.cov_mat) # cov_mat on device
        #log_probs = distribution.log_prob(actions.to(self.device)) # distribution on device
        log_probs = self.actor.evaluate(observations, actions)
        return V, log_probs

    '''def set_cov_mat(self, cov_mat):
        """
        Transfer cov matrix.
        """
        self.cov_mat = cov_mat'''

    def train_dataloader(self):
        """
        Custom RolloutDataset.
        """
        dataset = RolloutDataset(
            env=self.env,
            actor=self.actor,
            timesteps_per_batch=self.timesteps_per_batch,
            max_timesteps_per_episode=self.max_timesteps_per_episode,
            device=self.device,
            compute_discounted_rewards=self.compute_discounted_rewards
        )

        return DataLoader(dataset, batch_size=None, num_workers=0)
        # return DataLoader(dataset, batch_size=1, num_workers=0)
        # dataset = RolloutDataset(self.env, self.actor, self.timesteps_per_batch)
        # return DataLoader(dataset, batch_size=64)


    '''def save_hyperparameters(self, hyperparameters):
        for param, value in hyperparameters.items():
            setattr(self, param, value)
        #super().save_hyperparameters(**hyperparameters)
        super().save_hyperparameters()
        print (self.hyperparameters)
        #self.hparams = hyperparameters
        #self.timesteps_per_batch = 6400 # 4 episodes
        #self.max_timesteps_per_episode = 1600
        #self.n_updates_per_iteration = 15
        #self.learning_rate = 0.000429
        #self.gamma = 0.994
        #self.clip = 0.28
        #self.save_freq = 10
        
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))'''

    def training_step(self, batch, batch_idx):
        """
            Implements the PPO training step. Training_step defines the train loop.
        """
        #                                         Rollout phase
        # Calculate rollout time
        rollout_start = time.time()
        batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens = batch  # ALG STEP 3
        #batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens = self.rollout() # ALG STEP 3
        rollout_end = time.time()
        rollout_time = rollout_end - rollout_start

        # Calculate how many timesteps we collected this batch
        #timesteps_current += np.sum(batch_lens)
        self.timesteps_current += torch.sum(torch.tensor(batch_lens)).item()
        self.iterations_current += 1

        counting_start = time.time()
        V, _ = self.evaluate(batch_observations, batch_actions)
        # Compute advantage
        A_k = (batch_discounted_rewards - V.detach()).to(self.device)
        #std = A_k.std()
        #if std < 1e-6:
        #    std = 1.0
        A_k = (A_k - A_k.mean()) / (A_k.std())
        #A_k = (A_k - A_k.mean()) / (std + 1e-10)
        counting_end = time.time()
        counting_time = counting_end - counting_start

        # Update the policy and value networks
        weights_update_start = time.time()
        actor_loss, critic_loss = self.optimize_policy(batch_observations, batch_actions, batch_log_probs, A_k, batch_discounted_rewards)
        weights_update_end = time.time()
        weights_update_time = weights_update_end - weights_update_start

        #log results
        self.log("rollout_time", rollout_time, logger=True)
        self.log("counting_time", counting_time, logger=True)
        self.log("weights_update_time", weights_update_time, logger=True)

        self.log("actor_loss", actor_loss, prog_bar=True, logger=True)
        self.log("critic_loss", critic_loss, prog_bar=True, logger=True)
        self.log("timesteps_current", self.timesteps_current, logger=True)
        self.log("iterations_current", self.iterations_current, logger=True)
        # Log training summary
        #self._log_summary()            --- printing

        # Save model periodically
        #if iterations_current % self.save_freq == 0:
        if self.current_step % self.save_freq == 0:
            self.save_model()
        #return loss
        return actor_loss + critic_loss

    def optimize_policy(self, observations, actions, old_log_probs, advantages, discounted_rewards):
        """
        Performs multiple gradient updates on the policy and value networks.
        """
        # Update network for some n epochs
        for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, current_log_probs = self.evaluate(observations, actions)
            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            ratios = torch.exp(current_log_probs - old_log_probs)

            # Calculate clip loss
            clip_loss = torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages)
            # Calculate actor and critic losses
            actor_loss = -clip_loss.mean()
            critic_loss = nn.MSELoss()(V, discounted_rewards)

            # checking configure_optimizers availability
            if self.actor_optim is None or self.critic_optim is None:
                raise ValueError("Optimizers not configured. Call `configure_optimizers` first.")
            # Calculate gradients and perform backward propagation for (actor and critic) networks
            self.actor_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        return actor_loss.detach().cpu(), critic_loss.detach().cpu()

    def configure_optimizers(self):
        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)
        return [self.actor_optim, self.critic_optim]

    def save_model(self):
        """
        Saves the actor and critic models to disk.
        """
        torch.save(self.actor.state_dict(), './ppo_actor.pth') # f'ppo_actor_{self.global_step}.pth'
        torch.save(self.critic.state_dict(), './ppo_critic.pth') # f'ppo_critic_{self.global_step}.pth'

    def on_train_end(self):
        """
        Create GIf and print total reward in the end
        """
        frames, total_reward = self.generate_gif_after_training()
        # torch.cuda.empty_cache() ---- where is it
        self.log("total_reward", total_reward, logger=True)
        print(f"Total Reward: {total_reward}")

    def generate_gif_after_training(self):
        """
        Generating gif after training.
        """
        obs, _ = self.env.reset()
        frames = []
        total_reward = 0
        done = False

        while not done:
            frame = self.env.render()
            frames.append(frame)
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            action, _ = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            total_reward += reward
            done = terminated or truncated

        return frames, total_reward

    '''def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        timesteps_current = self.logger['timesteps_current']
        iterations_current = self.logger['iterations_current']
        avg_episode_lens = np.mean(self.logger['batch_lens'])
        avg_episode_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in self.logger['batch_rewards']])
        avg_actor_loss = np.mean([losses.float().cpu().mean().item() for losses in self.logger['actor_losses']])
        rollout_time = self.logger.get('rollout_time', 0)
        counting_time = self.logger.get('counting_time', 0)
        weights_update_time = self.logger.get('weights_update_time', 0)

        avg_episode_lens = str(round(avg_episode_lens, 2))
        avg_episode_rewards = str(round(avg_episode_rewards, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{iterations_current} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_episode_lens}", flush=True)
        print(f"Average Episodic Return: {avg_episode_rewards}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps current: {timesteps_current}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Rollout time: {rollout_time:.2f} secs", flush=True)
        print(f"Counting time: {counting_time:.2f} secs", flush=True)
        print(f"Weights update time: {weights_update_time:.2f} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'] = []
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []'''
