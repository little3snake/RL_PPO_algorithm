import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Normal
from torch.cuda.amp import autocast, GradScaler
#from multiprocessing import Pool, cpu_count
import time

from memory import Memory



class PPO:
    def __init__(self, policy_class, env, device, writer, **kwargs):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.device = device
        self.writer = writer
        # Initialize hyperparameters for training with PPO
        self.total_timesteps = kwargs.get('total_timesteps', 100_000)
        self.batch_size = kwargs.get('batch_size', 2048)
        self.max_timesteps_per_episode = kwargs.get('max_timesteps_per_episode', 1600)
        self.n_updates_per_iteration = kwargs.get('n_updates_per_iteration', 20)
        self.learning_rate = kwargs.get('learning_rate', 0.000429)
        self.gamma = kwargs.get('gamma', 0.9814)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip = kwargs.get('clip', 0.3598)  # 0.28
        self.save_freq = kwargs.get('update_epochs', 10)
        self.num_minibatches = kwargs.get('num_minibatches', 4)
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.entropy_coef = kwargs.get('ent_coef', 0.01)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)

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

        self.memory = Memory(obs_size=self.env.observation_space.shape[0], action_size=self.env.action_space.shape[0], **kwargs)
        self.current_timestep = 0

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            'timesteps_current': 0,
            'iterations_current': 0,
            'batch_lens': [],
            'batch_rewards': [],    # episodic returns
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def learn(self):
        #timesteps_current = 0
        iterations_current = 0

        while self.current_timestep < self.total_timesteps:
            #rollout_start = time.time()
            #batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens = \
            #    self.rollout()  # ALG STEP 3
            batch_len = self.rollout()  # ALG STEP 3
            #rollout_end = time.time()
            #self.logger['rollout_time'] = rollout_end - rollout_start

            self.current_timestep += batch_len
            iterations_current += 1
            #self.logger['timesteps_current'] = timesteps_current
            self.logger['iterations_current'] = iterations_current

            print("Logprobs in memory:", self.memory.logprobs)

            #counting_start = time.time()
            # calculate old value state
            #V, _ = self.evaluate(batch_observations, batch_actions)
            self.calculate_V()
            # calculate advantage
            #A_k = (batch_discounted_rewards - V.detach()).to(self.device)
            #A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            self.calculate_advantages()
            #counting_end = time.time()
            #self.logger['counting_time'] = counting_end - counting_start

            # Calculate weight_update time
            #weights_update_start = time.time()
            # Update network for some n epochs
            batch_indices = np.arange(self.batch_size)
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                np.random.shuffle(batch_indices)
                for i in range (0, self.batch_size, self.minibatch_size):
                    start = i
                    finish = i + self.minibatch_size
                    minibatch_inds = batch_indices[start:finish]

                    V_evaluate, current_log_probs, entropy = self.evaluate(self.memory.observations[minibatch_inds].detach(), self.memory.actions[minibatch_inds].detach())

                    #print (current_log_probs.shape)
                    #print (self.memory.logprobs[minibatch_inds].detach().shape)
                    ratios = torch.exp(current_log_probs - self.memory.logprobs[minibatch_inds].detach())

                    advantages = self.memory.advantages[minibatch_inds].detach()
                    minibatch_A_k = (advantages - advantages.mean()) / (advantages.std() + 1e-10) # normalization
                    clip_loss = torch.min(ratios * minibatch_A_k, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * minibatch_A_k)
                    actor_loss = -clip_loss.mean()
                    actor_loss -= entropy.mean()
                    #self.logger['actor_losses'].append(actor_loss.detach().cpu())

                    critic_loss = nn.MSELoss()(V_evaluate, self.memory.discounted_rewards[minibatch_inds].detach())
                    #V_clipped = self.memory.V[minibatch_inds] + torch.clamp(V_evaluate - self.memory.V[minibatch_inds], -self.clip_epsilon, self.clip_epsilon)
                    #critic_loss2 = nn.MSELoss()(V_clipped, self.memory.discounted_rewards[minibatch_inds])
                    #critic_loss = 0.5 * (torch.max(critic_loss1, critic_loss2)).mean()
                    #self.logger['critic_losses'].append(critic_loss.detach().cpu())
                    #total_loss = actor_loss + critic_loss

                    #estimated_value = self.critic_nn(states).squeeze(-1) -- V
                    #critic_loss1 = torch.square(estimated_value - gt) -- MSELoss
                    # and we need to calculate clipped loss where estimated value is replaced with old estimated value + clipped difference
                    #estimated_value_clipped = old_value_state + torch.clamp(
                    #    self.critic_nn(states).squeeze(-1) - old_value_state, - Config.CLIPPING_EPSILON,
                    #    Config.CLIPPING_EPSILON)
                    #critic_loss2 = torch.square(estimated_value_clipped - gt) --MSELoss with clip
                    # Compare two losses and take bigger and calculate mean to get final critic loss
                    #critic_loss = 0.5 * (torch.maximum(critic_loss1, critic_loss2)).mean()

                    self.actor_optim.zero_grad()  # better than simple .zero_grad() --set_to_none=True
                    self.critic_optim.zero_grad()  # better than simple .zero_grad() --set_to_none=True
                    actor_loss.backward()
                    critic_loss.backward()
                    #total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    self.critic_optim.step()
                    # Log actor loss
                    self.logger['actor_losses'].append(actor_loss.detach().cpu())

                # Log actor loss
                #self.logger['actor_losses'].append(actor_loss.detach().cpu())
            #weights_update_end = time.time()
            #self.logger['weights_update_time'] = weights_update_end - weights_update_start
            self._log_summary()

            if iterations_current % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

            self.memory.clear()

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
       # it's in memory
        '''batch_observations, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_discounted_rewards, batch_lens = [], [], []
        current_timestep = 0'''
        batch_lens, batch_rewards = [], []
        observation, _ = self.env.reset() # obs and info
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        batch_len = 0

        episode_rewards = []
        for current_timestep in range (self.batch_size):
            batch_len += 1

            #episode_rewards = [] -in memory
            #observation, _ = self.env.reset()  - in the bottom
            # observation = torch.tensor(observation, dtype=torch.float, device=self.device) - in the bottom
            #done = False - in memory
            #timestep_start = current_timestep - don't use
            #for _ in range(self.max_timesteps_per_episode): - useless with dones
            with torch.no_grad():
                action, log_prob = self.get_action(observation)
            #batch_observations.append(observation) - in memory
            next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated # True if any situation

            episode_rewards.append(reward)

            self.add_to_memory(observation, action, log_prob, next_observation, reward, done, current_timestep)
            observation = next_observation
            observation = torch.tensor(observation, dtype=torch.float, device=self.device)
            #observation = torch.tensor(observation, dtype=torch.float, device=self.device) - in memory
            #episode_rewards.append(reward) - in memory
            #batch_actions.append(action) - in memory
            #batch_log_probs.append(log_prob) - in memory
            #if terminated or truncated: - in done
            #    break
            if done:
                batch_lens.append(len(episode_rewards))
                batch_rewards.append(episode_rewards)
                episode_rewards = []
                observation, _ = self.env.reset()
                observation = torch.tensor(observation, dtype=torch.float, device=self.device)

        #batch_lens += current_timestep # logger
        #batch_rewards.append(episode_rewards) # logger

        '''batch_observations = torch.stack(batch_observations).to(self.device)
        batch_actions = torch.stack(batch_actions).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)'''
        # all in memory
        #batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards).to(self.device)

        self.logger['batch_rewards'] = batch_rewards
        self.logger['batch_lens'] = batch_lens

        #return batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens
        return batch_len

    def add_to_memory (self, obs, action, logprob, new_obs, reward, done, current_step):
        self.memory.add(obs, action, logprob, new_obs, reward, done, current_step)

    '''def compute_discounted_rewards(self, batch_rewards):
        batch_discounted_rewards = [] # == gt

        for episode_rews in reversed(batch_rewards):
            discounted_reward = 0
            #episode_discounted_reward = []
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_discounted_rewards.insert(0, discounted_reward)
            #batch_discounted_rewards.extend(episode_discounted_reward)
        #batch_discounted_rewards = torch.tensor(batch_discounted_rewards, dtype=torch.float, device=device)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float)'''

    def calculate_V (self):
        self.memory.set_V(self.critic(self.memory.observations).squeeze())
        # full self.memory.V

    def calculate_advantages (self):
        V_next = self.critic(self.memory.next_observations).squeeze().detach()
        #next_values = self.agent_control.get_critic_value(self.memory.new_states).squeeze(-1).detach()
        self.memory.set_gae_advantages(V_next)
        # full self.memory.advantages

    def get_action(self, observation):
        #    observation = torch.tensor(observation, dtype=torch.float, device=device)
        #observation = observation.unsqueeze(0)
        #with torch.no_grad():
        #print ("get action ", observation)
        mean = self.actor(observation)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach(), log_prob.detach()

    def evaluate(self, batch_observations, batch_actions):
        #batch_observations = batch_observations.to(device)
        #batch_actions = batch_actions.to(device)
        #V = self.critic(batch_observations).squeeze()
        # calculate V  - tensor in memory
        #self.memory.set_V(self.critic(self.memory.batch_observations).squeeze())
        # V - is a tensor from minibatch
        V = self.critic(batch_observations).squeeze()
        mean = self.actor(batch_observations)

        #actions_logstd = torch.full_like(mean, fill_value=-0.5)
        #actions_std = torch.exp(actions_logstd)
        #distribution = Normal(mean, actions_std)
        distribution = MultivariateNormal(mean, self.cov_mat) # cov_mat on device

        log_probs = distribution.log_prob(batch_actions) # distribution on device
        entropy = distribution.entropy()

        return V, log_probs, entropy

    '''def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
            hyperparameters

        Return:
            None
        """
        self.batch_size = 2048
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 20
        self.learning_rate = 0.000429
        self.gamma = 0.9814
        self.gae_lambda = 0.95
        self.clip =0.3598#0.28
        self.save_freq = 10
        self.num_minibatches = 4
        self.minibatch_size = self.batch_size // self.num_minibatches

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))'''

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
