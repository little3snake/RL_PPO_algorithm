import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.cuda.amp import autocast, GradScaler
#from multiprocessing import Pool, cpu_count
import time



class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, device, writer, **hyperparameters):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
            policy_class - the policy class to use for our actor/critic networks.
            env - the environment to train on.
            device - cuda, mps or cpu
            hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
            None
        """
        # Make sure the environment is compatible with our code
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.device = device
        self.writer = writer
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.actor = policy_class(self.n_observations, self.n_actions).to(self.device)  # ALG STEP 1
        self.critic = policy_class(self.n_observations, 1).to(self.device)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Initialize the covariance matrix for get_action(...), evaluate(...)
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            'timesteps_current': 0,
            'iterations_current': 0,
            'batch_lens': [],
            'batch_rewards': [],    # episodic returns
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
            total_timesteps - the total number of timesteps to train for
            writer - a TensorBoard writer object (SummaryWriter for logging metrics)

        Return:
            frames - list of frames for gif
            total_reward - total reward of the trained model
        """
        timesteps_current = 0
        iterations_current = 0

        while timesteps_current < total_timesteps:
            # Calculate rollout time
            rollout_start = time.time()
            # Collecting batch data
            batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens = \
                self.rollout()  # ALG STEP 3
            rollout_end = time.time()
            self.logger['rollout_time'] = rollout_end - rollout_start

            # Calculate how many timesteps we collected this batch
            timesteps_current += np.sum(batch_lens)
            iterations_current += 1
            self.logger['timesteps_current'] = timesteps_current
            self.logger['iterations_current'] = iterations_current

            # Calculate counting A_k and V time
            counting_start = time.time()
            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_observations, batch_actions)
            A_k = (batch_discounted_rewards - V.detach()).to(self.device)
            # Normalizing advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            counting_end = time.time()
            self.logger['counting_time'] = counting_end - counting_start

            # Calculate weight_update time
            weights_update_start = time.time()
            # Update network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, current_log_probs = self.evaluate(batch_observations, batch_actions)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(current_log_probs - batch_log_probs)

                 # Calculate parts of clip loss
                #clip_loss_1part = ratios * A_k
                #clip_loss_2part = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                clip_loss = torch.min(ratios * A_k, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k)

                # Calculate actor and critic losses
                actor_loss = -clip_loss.mean()
                #actor_loss = (-torch.min(clip_loss_1part, clip_loss_2part)).mean()
                critic_loss = nn.MSELoss()(V, batch_discounted_rewards)

                self.actor_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach().cpu())

            weights_update_end = time.time()
            self.logger['weights_update_time'] = weights_update_end - weights_update_start
            # Log training summary
            self._log_summary()

            # Save model
            if iterations_current % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')


        # Generate gif after learning
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

        torch.cuda.empty_cache()
        print ("before return in learn")
        return frames, total_reward

    def rollout(self):
        """
        This is where we collect the fresh batch of data
        from simulation (on-policy algorithm).

        Parameters:
            None

        Return:
            batch_observations - Shape: (number of timesteps, dimension of observation)
            batch_actions - Shape: (number of timesteps, dimension of action)
            batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
            batch_discounted_rewards - the discounted rewards of each timestep in this batch. Shape: (number of timesteps)
            batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        batch_observations, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_discounted_rewards, batch_lens = [], [], []
        current_timestep = 0

        while current_timestep < self.timesteps_per_batch:
            episode_rewards = []
            observation, _ = self.env.reset()  # Reset the environment
            observation = torch.tensor(observation, dtype=torch.float, device=self.device)
            done = False
            timestep_start = current_timestep

            for _ in range(self.max_timesteps_per_episode):
                current_timestep += 1
                batch_observations.append(observation)
                with torch.no_grad():
                    action, log_prob = self.get_action(observation)
                #action = action.clone().detach().to(dtype=torch.float, device=self.device)
                #log_prob = log_prob.clone().detach().to(dtype=torch.float, device=self.device)

                #action, log_prob = self.get_action(torch.tensor(observation, dtype=torch.float).to(self.device))
                observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                observation = torch.tensor(observation, dtype=torch.float, device=self.device)
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if terminated or truncated:
                    break

            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)

        #batch_observations = torch.tensor(batch_observations, dtype=torch.float).to(self.device)
        batch_observations = torch.stack(batch_observations).to(self.device)
        #batch_actions = torch.stack(batch_actions).to(self.device)
        batch_actions = torch.stack(batch_actions).to(self.device)
        #batch_log_probs = torch.stack(batch_log_probs).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)
        batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards).to(self.device)

        self.logger['batch_rewards'] = batch_rewards
        self.logger['batch_lens'] = batch_lens

        return batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens

    def compute_discounted_rewards(self, batch_rewards):
        """
        Compute the discounted rewards of each timestep in a batch given the rewards.

        Parameters:
            batch_rewards - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
            batch_discounted_rewards - the rewards to go, Shape: (number of timesteps in batch)
        """
        batch_discounted_rewards = []

        for episode_rews in reversed(batch_rewards):
            discounted_reward = 0
            #episode_discounted_reward = []
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_discounted_rewards.insert(0, discounted_reward)
            #batch_discounted_rewards.extend(episode_discounted_reward)
        #batch_discounted_rewards = torch.tensor(batch_discounted_rewards, dtype=torch.float, device=device)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float)

    def get_action(self, observation):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            observation - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """
        #if not isinstance(observation, torch.Tensor):
        #    observation = torch.tensor(observation, dtype=torch.float, device=device)
        #observation = observation.unsqueeze(0)

        #with torch.no_grad():
        mean = self.actor(observation)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach(), log_prob.detach()

    def evaluate(self, batch_observations, batch_actions):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_observations - the observations from the most recently collected batch as a tensor.
                  Shape: (number of timesteps in batch, dimension of observation)
            batch_actions - the actions from the most recently collected batch as a tensor.
                  Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        #batch_observations = batch_observations.to(device)
        #batch_actions = batch_actions.to(device)

        V = self.critic(batch_observations).squeeze()

        mean = self.actor(batch_observations) # actor on device
        distribution = MultivariateNormal(mean, self.cov_mat) # cov_mat on device
        log_probs = distribution.log_prob(batch_actions) # distribution on device

        return V, log_probs

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

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
            None

        Return:
            None
        """
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
        self.logger['actor_losses'] = []
