from cmath import acosh
import gymnasium as gym
import time
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch import dtype
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

class PPO:
    def __init__(self, policy_class, envs, device, rank=0, writer=None, **kwargs):
        self.device = device
        self.world_size = kwargs.get("world_size", 1)
        print("world size", self.world_size)
        self.rank = rank
        self.writer = writer
        self.seed = kwargs.get("seed", 1)
        self.track = kwargs.get("track", False)
        self.create_gif_bool = kwargs.get("create_gif", False)
        self.save_policies = kwargs.get("save_policies", False)

        self.num_steps = kwargs.get("num_steps", 1) # rollout steps for 1 process per batch
        self.gamma = kwargs.get("gamma", 0.99421)
        self.local_num_envs = kwargs.get("local_num_envs", 1)
        self.num_iterations = kwargs.get("num_iterations", 1)
        self.num_envs = kwargs.get("num_envs", 1) # in rollout
        self.update_epochs = kwargs.get("update_epochs", 1)
        self.clip = kwargs.get("clip", 0.2)
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.save_freq = kwargs.get("save_freq", 1)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.local_batch_size = kwargs.get("local_batch_size", 1)
        self.local_minibatch_size = kwargs.get("local_minibatch_size", 1)

        self.envs = envs
        self.n_observations = envs.single_observation_space.shape[0]
        self.n_actions = envs.single_action_space.shape[0]
        self.actor = policy_class(self.n_observations, self.n_actions).to(self.device)  # ALG STEP 1
        self.critic = policy_class(self.n_observations, 1).to(self.device)
        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate) # eps=1e-5
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Initialize the covariance matrix for get_action(...), evaluate(...)
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.zeros_init()

        if self.world_size > 1:
            for param in self.actor.parameters():
                dist.broadcast(param.data, src=0)
            for param in self.critic.parameters():
                dist.broadcast(param.data, src=0)
        # For root
        if self.rank == 0:
            '''self.logger = {
                'delta_t': time.time_ns(),
                'timesteps_current': 0,
                'iterations_current': 0,
                'batch_lens': [],
                'batch_rewards': [],    # episodic returns
                'actor_losses': [],     # losses of actor network in current iteration
            }'''

    def zeros_init(self):
        self.observations = torch.zeros((self.num_steps, self.local_num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.local_num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.local_num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.local_num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.local_num_envs)).to(self.device)
        self.discounted_rewards = torch.zeros_like(self.rewards).to(self.device)
        self.global_step = 0
        self.global_step_for_writer = 0

    def learn(self):
        self.global_step = 0
        start_time = time.time()
        obs, _ = self.envs.reset(seed=self.seed)
        obs = torch.Tensor(obs).to(self.device)
        done = torch.zeros(self.local_num_envs).to(self.device)
        itertime = 0

        for i in range(1, self.num_iterations + 1):
            itertimestart = time.time()
            self.rollout(obs, done, i) # ALG STEP 3
            V = self.critic(self.observations).squeeze()
            A_k = (self.discounted_rewards - V.detach()).to(self.device)
            if A_k.numel() > 1:
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            else:
                A_k = torch.zeros_like(A_k)  # Если данных недостаточно, обнуляем A_k

            allreducetime, batchtime = self.update_model(A_k) # ALG STEP 6 & 7
            itertimeend = time.time()
            itertime = itertimeend - itertimestart
            #if self.rank == 0:
                # self._log_summary()

            if self.save_policies and  self.rank == 0 and i % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

            if self.rank == 0 and self.track:
                reward, episode_len = self.try_for_reward_in_wandb()
                wandb.log({
                    "global_step": self.global_step,
                    "iteration": i,
                    "charts/allreducetime_persentage_in_batchtime": allreducetime/batchtime,
                    "charts/allreducetime_persentage_in_itertime": allreducetime/itertime,
                    "global_step_world_size": self.global_step * self.world_size,
                    "charts/reward": reward,
                    "charts/episode_len": episode_len,
                    "charts/current_step": int(time.time() - start_time),
                    "charts/actor_learning_rate": self.actor_optim.param_groups[0]["lr"],
                    #"charts/reward_new": reward,
                    "charts/reward_new_new": reward,
                })
                print ("g_s ", self.global_step, " i ", i, " g_s*w_s ", self.global_step * self.world_size)

        # creating a gif
        if self.create_gif_bool and self.rank == 0:
            frames, total_reward = self.create_gif()
            #frames = []
            #total_reward = 0
            print ("before return in learn")
            return list(frames), total_reward
        return [], 0


    def rollout(self, obs, done, iteration):
        for step in range (0, self.num_steps):
            #self.global_step += self.num_envs
            self.global_step += self.local_num_envs
            self.global_step_for_writer += self.num_steps * self.world_size
            self.observations[step] = obs #- initial code
            self.dones[step] = done
            with torch.no_grad():
                action, log_prob = self.get_action(obs)
            obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)  # to 1dim tensor
            self.actions[step] = action
            self.logprobs[step] = log_prob
            done = np.logical_or(terminated, truncated)
            obs = torch.tensor(obs).to(self.device)
            done = torch.tensor(done).to(self.device)

        # for different seeds
        #self.compare_rollouts(self.observations)

        print(
            f"local_rank: {self.rank}, action.sum(): {action.sum()}, iteration: {iteration}, "
            f"agent.actor.weight.sum(): {sum(p.sum() for p in self.actor.parameters() if p.requires_grad)}"
            #f"reward: {total_reward}"
        )

        self.compute_discounted_rewards()

    def compute_discounted_rewards(self):
        discounted_reward_t = torch.zeros(self.local_num_envs).to(self.device)
        for t in reversed(range(self.num_steps)):
            # If the episode is done, reset the discounted reward
            discounted_reward_t = self.rewards[t] + self.gamma * discounted_reward_t * (1 - self.dones[t])
            self.discounted_rewards[t] = discounted_reward_t

    #for actor
    def get_action(self, observation):
        mean = self.actor(observation)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.detach(), log_prob.detach()

    def evaluate(self, observations, actions):
        V = self.critic(observations).squeeze()
        mean = self.actor(observations)
        distribution = MultivariateNormal(mean, self.cov_mat)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return V, log_probs, entropy

    def reshape (self, A_k):
        batch_A_k = A_k.reshape(-1)
        batch_observations = self.observations.reshape((-1,) + self.envs.single_observation_space.shape)
        batch_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        batch_disc_rewards = self.discounted_rewards.reshape(-1)
        batch_logprobs = self.logprobs.reshape(-1)
        return batch_A_k, batch_observations, batch_actions, batch_disc_rewards, batch_logprobs


    def update_model(self, A_k):
        try:
            batch_A_k, batch_observations, batch_actions, batch_disc_rewards, \
                batch_logprobs = self.reshape(A_k)
            batch_inds = np.arange(self.local_batch_size)
            batchtime = 0
            allreducetime = 0

            for _ in range(self.update_epochs):
                np.random.shuffle(batch_inds)
                for i in range(0, self.local_batch_size, self.local_minibatch_size):
                    batchtimestart = time.time()
                    start = i
                    finish = i + self.local_minibatch_size
                    minibatch_inds = batch_inds[start:finish]

                    batch_V, current_log_probs, entropy = self.evaluate(batch_observations[minibatch_inds], batch_actions[minibatch_inds])
                    ratios = torch.exp(current_log_probs - batch_logprobs[minibatch_inds])
                    clip_loss = torch.min(ratios * batch_A_k[minibatch_inds], torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_A_k[minibatch_inds])
                    actor_loss = -clip_loss.mean()
                    actor_loss -= entropy.mean() * self.entropy_coef
                    critic_loss = nn.MSELoss()(batch_V, batch_disc_rewards[minibatch_inds])
                    self.actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()

                    if self.world_size > 1:
                        timestart = time.time()
                        all_actor_grads_list = []
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                all_actor_grads_list.append(param.grad.view(-1))
                        all_actor_grads = torch.cat(all_actor_grads_list)
                        if all_actor_grads.device != torch.device("cpu"): # because of dist and gloo
                            all_actor_grads = all_actor_grads.cpu()
                        dist.all_reduce(all_actor_grads, op=dist.ReduceOp.SUM)

                        offset = 0
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                param.grad.data.copy_(
                                    all_actor_grads[offset: offset + param.numel()].view_as(
                                        param.grad.data) / self.world_size
                                )
                                offset += param.numel()

                        all_critic_grads_list = []
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                all_critic_grads_list.append(param.grad.view(-1))
                        all_critic_grads = torch.cat(all_critic_grads_list)
                        dist.all_reduce(all_critic_grads, op=dist.ReduceOp.SUM)

                        offset = 0
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                param.grad.data.copy_(
                                    all_critic_grads[offset: offset + param.numel()].view_as(
                                        param.grad.data) / self.world_size
                                )
                                offset += param.numel()
                        timeend = time.time()
                        allreducetime += timeend - timestart

                    # cut gradients
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                    self.actor_optim.step()
                    self.critic_optim.step()

                    batchtimeend = time.time()
                    batchtime +=batchtimeend - batchtimestart
                #return actor_loss, critic_loss
            #time_persentage = allreducetime / batchtime
            return allreducetime, batchtime

        except Exception as e:
            print(f"Error in rank {self.rank}: {e}")
            raise

    def compare_rollouts(self, data):
        """For compare with different seeds"""
        data_tensor = torch.tensor(data, dtype=torch.float32).to("cpu")
        gathered_data = [torch.zeros_like(data_tensor) for _ in range(self.world_size)]

        dist.all_gather(gathered_data, data_tensor)

        if self.rank == 0:
            for i, d in enumerate(gathered_data):
                #print(f"Процесс {i}: Среднее = {d.mean().item()}, Std = {d.std().item()}, Min = {d.min().item()}, Max = {d.max().item()}")
                print(
                    f"Процесс {i}:, {d.mean().item()}, {d.std().item()}, {d.min().item()}, {d.max().item()}")

    def try_for_reward_in_wandb(self):
        episode_len = 0
        obs, _ = self.envs.reset()
        total_reward = 0
        done = False
        while not done:
            episode_len += 1
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            action, _ = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
            total_reward += reward
            done = np.logical_or(terminated[0], truncated[0])
        return np.mean(total_reward), episode_len

    def create_gif (self):
        obs, _ = self.envs.reset()
        frames = []
        total_reward = 0
        done = False
        while not done:
            frame = self.envs.render()
            frames.append(frame[0])
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            action, _ = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
            total_reward += reward
            done = np.logical_or(terminated[0], truncated[0])
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
