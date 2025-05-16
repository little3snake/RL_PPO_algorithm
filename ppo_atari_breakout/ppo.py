import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.optim import Adam # RMSprop for experiments
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import time

from network import NN

class PPO:
    def __init__(self, policy_class, env, device, writer, **kwargs):
        self.device = device
        self.writer = writer

        self.capture_gif = kwargs.get('capture_gif', False)
        self.anneal_lr = kwargs.get('anneal_lr', True)

        self.timesteps_per_batch = kwargs.get('timesteps_per_batch', 2048)
        self.max_timesteps_per_episode = kwargs.get('max_timesteps_per_episode', 512)
        self.n_updates_per_iteration = kwargs.get('n_updates_per_iteration', 10)
        self.learning_rate = kwargs.get('learning_rate', 2.5e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip = kwargs.get('clip', 0.1)
        #self.save_freq = kwargs.get('update_epochs', 10)
        self.num_minibatches = kwargs.get('num_minibatches', 4)
        self.minibatch_size = self.timesteps_per_batch // self.num_minibatches # batch_size / num_minibatches
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.entropy_coef = kwargs.get('ent_coef', 0.01)
        self.num_iterations = kwargs.get('num_iterations', 400)
        self.current_batch_size = 0

        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.net = NN(self.env).to(self.device)
        self.network = self.net.net.to(self.device)
        self.actor = self.net.actor.to(self.device)
        self.critic = self.net.critic.to(self.device)


        # Initialize optimizers for actor and critic
        self.optimizer = Adam(self.net.parameters(), lr=self.learning_rate) # eps = 1e-5
        #self.optimizer = RMSprop(self.net.parameters(), lr=self.learning_rate)  # eps = 1e-5
        #self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        #self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

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
        observation, _ = self.env.reset()
        observation = torch.tensor(observation).to(self.device)
        done = torch.tensor(False).to(self.device)

        for i in range(1, self.num_iterations + 1):

            if self.anneal_lr:
                frac = 1.0 - (iterations_current - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
                #self.actor_optim.param_groups[0]["lr"] = lrnow
                #self.critic_optim.param_groups[0]["lr"] = lrnow

            rollout_start = time.time()

            batch_observations, batch_actions, batch_log_probs, batch_next_observations, batch_rewards, batch_dones, batch_discounted_rewards, last_done = self.rollout(observation, done, timesteps_current)

            rollout_end = time.time()
            self.logger['rollout_time'] = rollout_end - rollout_start
            self.current_batch_size = batch_dones.size(0)
            timesteps_current += self.timesteps_per_batch
            iterations_current += 1
            self.logger['timesteps_current'] = timesteps_current
            self.logger['iterations_current'] = iterations_current
            self.writer.add_scalar("charts/current_step", int(time.time() - start_time), timesteps_current)
            self.writer.add_scalar("charts/current_iteration", int(time.time() - start_time), iterations_current)
            counting_start = time.time()

            batch_discounted_rewards, A_k = self.calculate_advantages(
                batch_observations, batch_next_observations, batch_rewards, batch_dones, last_done
            )

            counting_end = time.time()
            self.logger['counting_time'] = counting_end - counting_start
            weights_update_start = time.time()
            batch_inds = np.arange(self.timesteps_per_batch)

            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7

                np.random.shuffle(batch_inds)
                for i in range(0, self.timesteps_per_batch, self.minibatch_size):

                    start = i
                    finish = i + self.minibatch_size
                    minibatch_inds = batch_inds[start:finish]

                    obs_mb = batch_observations[minibatch_inds]
                    actions_mb = batch_actions[minibatch_inds]
                    old_log_probs_mb = batch_log_probs[minibatch_inds]
                    advantages_mb = A_k[minibatch_inds]
                    advantages_mb = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-8)
                    returns_mb = batch_discounted_rewards[minibatch_inds]

                    V, current_log_probs, entropy = self.evaluate(obs_mb, actions_mb)
                    logratio = current_log_probs - old_log_probs_mb
                    ratios = torch.exp(logratio)
                    clip_loss = torch.max(-advantages_mb * ratios , -advantages_mb * torch.clamp(ratios, 1 - self.clip, 1 + self.clip)) #A_k[minibatch_inds])
                    actor_loss = clip_loss.mean()
                    actor_loss -= entropy.mean() * self.entropy_coef
                    critic_loss = nn.MSELoss()(V, returns_mb)

                    # only for wandb  - understanding where program falls
                    approx_kl = (old_log_probs_mb - current_log_probs).mean()
                    self.writer.add_scalar("check/approx_kl", approx_kl.item(), timesteps_current)

                    total_loss = actor_loss + critic_loss * 0.5
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    self.logger['actor_losses'].append(actor_loss.detach().cpu())
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), timesteps_current)
                    self.writer.add_scalar("losses/critic_loss", critic_loss.item(), timesteps_current)
                    self.writer.add_scalar("losses/entropy", entropy.mean().item(), timesteps_current)
                    self.writer.add_scalar("check/advantages mean/std", advantages_mb.mean().item() / advantages_mb.std().item(), timesteps_current)
                    self.writer.add_scalar("check/ratios mean/std",
                                           ratios.mean().item() / ratios.std().item(), timesteps_current)

            weights_update_end = time.time()
            self.logger['weights_update_time'] = weights_update_end - weights_update_start
            # Log training summary
            self._log_summary()

        # find total reward after learning (and maybe frames for gif if self.capture_gif = True)
        obs, _ = self.env.reset()
        frames = []
        total_reward = 0
        done = False
        while not done:
            if self.capture_gif:
                frame = self.env.render()
                frames.append(frame)

            action, _ = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

        torch.cuda.empty_cache()
        print ("before return in learn")
        return frames, total_reward # ordinary return
        #return total_reward # return for optuna

    def rollout(self, observation, done, timestep_current):

        batch_observations = torch.zeros((self.timesteps_per_batch,) + self.env.observation_space.shape).to(self.device)
        batch_next_observations = torch.zeros((self.timesteps_per_batch,) + self.env.observation_space.shape).to(self.device)
        batch_actions = torch.zeros((self.timesteps_per_batch,)).to(self.device)
        batch_log_probs = torch.zeros((self.timesteps_per_batch,)).to(self.device)
        batch_rews, batch_lens = [], []
        batch_rewards = torch.zeros((self.timesteps_per_batch,)).to(self.device)
        batch_dones = torch.zeros((self.timesteps_per_batch,)).to(self.device)

        current_timestep = 0
        episode_rewards, last_episode_rewards = [], []  # only for writer
        batch_len = 0
        action_counts = {i: 0 for i in range(self.n_actions)} # only for wandb

        for step in range (0, self.timesteps_per_batch):
            batch_observations[step] = observation
            batch_dones[step] = done
            with torch.no_grad():
                action, log_prob = self.get_action(observation)

            action_counts[action.item()] += 1 # only for wandb

            next_observation, reward, terminated, truncated, _ = self.env.step(
                action.cpu().numpy())
            batch_rewards[step] = torch.tensor(reward).to(self.device)
            batch_actions[step] = action.unsqueeze(0)
            batch_log_probs[step] = log_prob
            done = terminated or truncated
            done = torch.tensor(float(done)).to(self.device)
            next_observation = torch.tensor(next_observation).to(self.device)
            batch_next_observations[step] = next_observation
            observation = next_observation

            episode_rewards.append(reward)  # for rews
            if done == True:
                batch_rews.append(episode_rewards)
                batch_lens.append(len(episode_rewards))
                self.writer.add_scalar("charts/episode_rewards", np.sum(episode_rewards), timestep_current)
                self.writer.add_scalar("charts/episode_lens", len(episode_rewards), timestep_current)

                episode_rewards = []
                observation, _ = self.env.reset()
                observation = torch.tensor(observation).to(self.device)
                done = torch.tensor(0).to(self.device)



        # logging
        total_actions = sum(action_counts.values())
        action_distribution = {k: v / total_actions for k, v in action_counts.items()}
        self.logger["action_distribution"] = action_distribution

        if self.writer:
            for action, prob in action_distribution.items():
                self.writer.add_scalar(f"action_distribution/action_{action}", prob, timestep_current)

        batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards, batch_dones).to(self.device)
        last_done = done

        self.logger['batch_rewards'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        timestep_current += np.sum(batch_lens)
        avg_episode_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in self.logger['batch_rewards']])
        avg_episode_lens = np.mean(self.logger['batch_lens'])
        self.writer.add_scalar("charts/avg_episode_rewards", avg_episode_rewards, timestep_current)
        self.writer.add_scalar("charts/avg_episode_lens", avg_episode_lens, timestep_current)

        return batch_observations, batch_actions, batch_log_probs, batch_next_observations, batch_rewards, batch_dones, batch_discounted_rewards, done

    def calculate_advantages(self, batch_observations, batch_next_observations, batch_rewards, batch_dones, last_done):
        with torch.no_grad():
            hidden = self.network(batch_observations / 255.0)
            V = self.critic(hidden).squeeze()
            hidden = self.network(batch_next_observations / 255.0)
            V_next = self.critic(hidden).squeeze()
            advantages = torch.zeros_like(batch_rewards).to(self.device)

            last_advantage = 0
            for t in reversed(range(len(batch_rewards))): # for TD
                mask = 1.0 - last_done if t == self.timesteps_per_batch - 1 else 1.0 - batch_dones[t + 1]
                last_value = V_next[t]
                delta = batch_rewards[t] + self.gamma * last_value * mask - V[t]
                last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
                advantages[t] = last_advantage
            batch_discounted_rewards = advantages + V
        return batch_discounted_rewards.to(self.device), advantages

    def compute_discounted_rewards(self, batch_rewards, batch_dones):

        batch_discounted_rewards = torch.zeros_like(batch_rewards).to(self.device)
        discounted_reward_t = torch.zeros(1).to(self.device)
        for t in reversed(range(self.timesteps_per_batch)):
            # If the episode is done, reset the discounted reward
            discounted_reward_t = batch_rewards[t] + self.gamma * discounted_reward_t * (1 - batch_dones[t])
            batch_discounted_rewards[t] = discounted_reward_t

        return batch_discounted_rewards

    def get_action(self, observation):
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)  # (1, C, H, W)
        hidden = self.network(observation / 255.0)
        logits = self.actor(hidden)
        distribution = Categorical(logits=logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.squeeze(), log_prob

    def evaluate(self, batch_observations, batch_actions):
        hidden = self.network(batch_observations / 255.0)
        V = self.critic(hidden).squeeze()
        logits = self.actor(hidden)
        distribution = Categorical(logits=logits)
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