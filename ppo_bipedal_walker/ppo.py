import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.cuda.amp import autocast, GradScaler
import time

class PPO:
    def __init__(self, policy_class, env, device, writer, **kwargs):
        self.device = device
        self.writer = writer

        #self.total_timesteps = kwargs.get('total_timesteps', 100_000)
        self.timesteps_per_batch = kwargs.get('timesteps_per_batch', 2048)
        self.max_timesteps_per_episode = kwargs.get('max_timesteps_per_episode', 1600)
        self.n_updates_per_iteration = kwargs.get('n_updates_per_iteration', 20)
        self.learning_rate = kwargs.get('learning_rate', 0.000429)
        self.anneal_lr = kwargs.get('anneal_lr', True)
        self.gamma = kwargs.get('gamma', 0.9814)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip = kwargs.get('clip', 0.3598)  # 0.28
        self.save_freq = kwargs.get('update_epochs', 10)
        self.num_minibatches = kwargs.get('num_minibatches', 4)
        self.minibatch_size = self.timesteps_per_batch // self.num_minibatches # batch_size / num_minibatches
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.entropy_coef = kwargs.get('ent_coef', 0.01)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.num_iterations = kwargs.get('num_iterations', 400)
        #runtime
        self.current_batch_size = 0

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
            'batch_rewards': [],
            'actor_losses': [],
        }

    def learn(self, total_timesteps):
        timesteps_current = 0
        iterations_current = 0
        start_time = time.time() # for wandb

        while timesteps_current < total_timesteps:

            if self.anneal_lr:
                frac = 1.0 - (iterations_current - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.actor_optim.param_groups[0]["lr"] = lrnow
                self.critic_optim.param_groups[0]["lr"] = lrnow

            rollout_start = time.time()
            batch_observations, batch_actions, batch_log_probs, batch_next_observations, batch_rewards, batch_dones, batch_discounted_rewards, rollout_timestep_curr = self.rollout(timesteps_current)
            rollout_end = time.time()
            self.logger['rollout_time'] = rollout_end - rollout_start

            self.current_batch_size = batch_dones.size(0)

            timesteps_current = rollout_timestep_curr
            iterations_current += 1
            self.logger['timesteps_current'] = timesteps_current
            self.logger['iterations_current'] = iterations_current
            self.writer.add_scalar("charts/current_step", int(time.time() - start_time), timesteps_current)
            self.writer.add_scalar("charts/current_iteration", int(time.time() - start_time), iterations_current)

            counting_start = time.time()
            V = self.critic(batch_observations).squeeze()
            batch_advantages = (batch_discounted_rewards - V.detach()).to(self.device)
            A_k = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            counting_end = time.time()
            self.logger['counting_time'] = counting_end - counting_start

            weights_update_start = time.time()
            #batch_inds = np.arange(self.current_batch_size)
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                V, current_log_probs, entropy = self.evaluate(batch_observations, batch_actions)
                ratios = torch.exp(current_log_probs - batch_log_probs)
                clip_loss = torch.min(ratios * A_k, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k)
                actor_loss = -clip_loss.mean()
                actor_loss -= entropy.mean() * self.entropy_coef
                critic_loss = nn.MSELoss()(V, batch_discounted_rewards[0:self.current_batch_size])
                self.actor_optim.zero_grad(set_to_none=True)  # better than simple .zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                self.critic_optim.zero_grad(set_to_none=True)  # better than simple .zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                self.logger['actor_losses'].append(actor_loss.detach().cpu())
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), timesteps_current)
                self.writer.add_scalar("losses/critic_loss", critic_loss.item(), timesteps_current)
                '''np.random.shuffle(batch_inds)
                for i in range(0, self.timesteps_per_batch, self.minibatch_size):
                    start = i
                    finish = i + self.minibatch_size
                    minibatch_inds = batch_inds[start:finish]
                    V, current_log_probs, entropy = self.evaluate(batch_observations[minibatch_inds], batch_actions[minibatch_inds])
                    ratios = torch.exp(current_log_probs - batch_log_probs[minibatch_inds])
                    clip_loss = torch.min(ratios * A_k[minibatch_inds], torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k[minibatch_inds])
                    actor_loss = -clip_loss.mean()
                    actor_loss -= entropy.mean() * self.entropy_coef
                    critic_loss = nn.MSELoss()(V, batch_discounted_rewards[minibatch_inds])

                    self.actor_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    self.critic_optim.zero_grad(set_to_none=True) # better than simple .zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    self.logger['actor_losses'].append(actor_loss.detach().cpu())
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), timesteps_current)
                    self.writer.add_scalar("losses/critic_loss", critic_loss.item(), timesteps_current)
                #self.writer.add_scalar("charts/reward", reward, timesteps_current)'''

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
            action, _ = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

        torch.cuda.empty_cache()
        print ("before return in learn")
        return frames, total_reward

    def rollout(self, timestep_current):
        batch_observations, batch_actions, batch_log_probs, batch_next_observations = [], [], [],[]
        batch_rewards, batch_rews, batch_dones, batch_lens = [], [], [], []
        current_timestep = 0
        observation, _ = self.env.reset()  # obs and info
        episode_rewards, last_episode_rewards = [], [] # last only for writer
        batch_len = 0

        while current_timestep < self.timesteps_per_batch:
            episode_rewards = [] # for rews
            observation, _ = self.env.reset()
            done = False
            timestep_start = current_timestep
            for _ in range(self.max_timesteps_per_episode):
                current_timestep += 1
                with torch.no_grad():
                    action, log_prob = self.get_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                done = 1 if done else 0
                batch_observations.append(torch.tensor(observation).to(self.device))
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_next_observations.append(torch.tensor(next_observation).to(self.device))
                batch_rewards.append(torch.tensor(reward).to(self.device))
                batch_dones.append(torch.tensor(done).to(self.device))
                episode_rewards.append(reward)  # for rews
                observation = next_observation
                if done == 1:
                    break

            batch_lens.append(len(episode_rewards))
            batch_rews.append(episode_rewards)

        batch_observations = torch.stack(batch_observations).to(self.device)
        batch_actions = torch.stack(batch_actions).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)
        batch_next_observations = torch.stack(batch_next_observations).to(self.device)
        batch_rewards = torch.stack(batch_rewards).to(self.device)
        batch_dones = torch.stack(batch_dones).to(self.device)
        batch_discounted_rewards = self.compute_discounted_rewards(batch_rews).to(self.device)

        self.logger['batch_rewards'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        timestep_current += np.sum(batch_lens)
        avg_episode_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in self.logger['batch_rewards']])
        avg_episode_lens = np.mean(self.logger['batch_lens'])
        self.writer.add_scalar("charts/avg_episode_rewards", avg_episode_rewards, timestep_current)
        self.writer.add_scalar("charts/avg_episode_lens", avg_episode_lens, timestep_current)

        return batch_observations, batch_actions, batch_log_probs, batch_next_observations, batch_rewards, batch_dones, batch_discounted_rewards, timestep_current

    '''def calculate_advantages(self, batch_observations, batch_next_observations, batch_rewards, batch_dones):
        batch_discounted_rewards = torch.zeros(self.current_batch_size + 1).to(self.device)
        batch_advantages = torch.zeros(self.current_batch_size).to(self.device)

        V_next = self.critic(batch_next_observations).squeeze(-1).detach()
        V = self.critic(batch_observations).squeeze(-1).detach()

        batch_discounted_rewards[self.current_batch_size] = V_next[-1]
        last_advantage = 0
        last_value = V_next[-1]
        for t in reversed(range(self.current_batch_size)):
            mask = 1.0 - batch_dones[t]
            last_value = last_value * mask
            last_advantage = last_advantage + mask
            delta = batch_rewards[t] + self.gamma * last_value - V[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            batch_advantages[t] = last_advantage
            last_value = V_next[t]
            batch_discounted_rewards[t] = batch_rewards[t] + self.gamma * batch_discounted_rewards[t + 1] * mask
        return batch_discounted_rewards, batch_advantages'''

    def compute_discounted_rewards(self, batch_rewards):
        batch_discounted_rewards = []

        for episode_rews in reversed(batch_rewards):
            discounted_reward = 0
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_discounted_rewards.insert(0, discounted_reward)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float)

    def get_action(self, observation):
        #with torch.no_grad():
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        mean = self.actor(observation)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.detach(), log_prob.detach()

    def evaluate(self, batch_observations, batch_actions):
        V = self.critic(batch_observations).squeeze()
        mean = self.actor(batch_observations)
        distribution = MultivariateNormal(mean, self.cov_mat)
        log_probs = distribution.log_prob(batch_actions)
        entropy = distribution.entropy()
        return V, log_probs, entropy

    def _log_summary(self):
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
