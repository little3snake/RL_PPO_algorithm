import torch
import numpy as np
import gymnasium as gym
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from network import Actor, Critic

torch.autograd.set_detect_anomaly(True)

# Фиктивный Dataset для DataLoader
class DummyDataset(Dataset):
    def __len__(self):
        return 1  # Просто возвращаем 1, чтобы Lightning не ругался

    def __getitem__(self, idx):
        return torch.tensor([0.0])  # Возвращаем фиктивный тензор

class PPOLightning(pl.LightningModule):
    def __init__(self, env, hyperparameters):
        #super(PPOLightning, self).__init__()
        super().__init__()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.env = env
        self.hyperparameters = hyperparameters

        # Отключаем автоматическую оптимизацию
        self.automatic_optimization = False

        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.critic = Critic(env.observation_space.shape[0], 1).to(self.device)

        self.actor_optim = None
        self.critic_optim = None

        self.cov_var = torch.full(size=(env.action_space.shape[0],), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.logger_dict = {
            'batch_rewards': [],
            'batch_lens': [],
            'actor_losses': [],
        }

    def forward(self, x):
        return self.actor(x)

    def train_dataloader(self):
        # Возвращаем фиктивный DataLoader
        return DataLoader(DummyDataset(), batch_size=1)

    def training_step(self, batch, batch_idx):
        opt_actor, opt_critic = self.optimizers()
        batch_obs, batch_acts, batch_log_probs, batch_discounted_rewards, batch_lens = self.rollout()

        V, _ = self.evaluate(batch_obs, batch_acts)
        #V = self.critic(batch_obs).squeeze()

        A_k = (batch_discounted_rewards - V.detach()).to(self.device)
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        actor_losses, critic_losses = [], []
        for _ in range(self.hyperparameters['n_updates_per_iteration']):
            # Обновление актора
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            clip_loss = torch.min(ratios * A_k, torch.clamp(ratios, 1 - self.hyperparameters['clip'], 1 + self.hyperparameters['clip']) * A_k)
            actor_loss = -clip_loss.mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            #self.manual_backward(actor_loss)  # retain_graph = True
            self.actor_optim.step()

            critic_loss = nn.MSELoss()(V, batch_discounted_rewards)

            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            #self.manual_backward(critic_loss)  # Граф больше не нужен, поэтому retain_graph=False
            self.critic_optim.step()

            '''
            #Изначальный вариант
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            clip_loss = torch.min(ratios * A_k, torch.clamp(ratios, 1 - self.hyperparameters['clip'],
                                                            1 + self.hyperparameters['clip']) * A_k)
            actor_loss = -clip_loss.mean()
            actor_loss.backward()

            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            critic_loss = nn.MSELoss()(V, batch_discounted_rewards)
            critic_loss.backward()'''

            '''
            # Отдельные обновления актора и критика и расчеты для них (не получается - все равно один граф)
            # Обновление актора
            mean = self.actor(batch_obs)
            cov_mat = self.cov_mat.to(self.device)
            dist = MultivariateNormal(mean, cov_mat)
            curr_log_probs = dist.log_prob(batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            clip_loss = torch.min(
                ratios * A_k,
                torch.clamp(ratios, 1 - self.hyperparameters['clip'], 1 + self.hyperparameters['clip']) * A_k
            )
            actor_loss = -clip_loss.mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()  # Граф освобождается
            self.actor_optim.step()

            # Обновление критика
            #V = self.critic(batch_obs).squeeze()
            critic_loss = nn.MSELoss()(V, batch_discounted_rewards)

            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()  # Граф освобождается
            self.critic_optim.step()'''

            '''
            # Ещё одна версия раздельного
            # --- Обновление актора ---
            mean = self.actor(batch_obs)
            cov_mat = self.cov_mat.to(self.device)
            dist = MultivariateNormal(mean, cov_mat)
            curr_log_probs = dist.log_prob(batch_acts)
            # Вычисляем (A_k) для актора
            V_actor = self.critic(batch_obs).squeeze().detach()  # Отключаем граф для актора
            A_k_actor = (batch_discounted_rewards - V_actor).to(self.device)
            A_k_actor = (A_k_actor - A_k_actor.mean()) / (A_k_actor.std() + 1e-10)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            clip_loss = torch.min(
                ratios * A_k_actor,
                torch.clamp(ratios, 1 - self.hyperparameters['clip'], 1 + self.hyperparameters['clip']) * A_k_actor
            )
            actor_loss = -clip_loss.mean()
            # Обнуляем градиенты актора и выполняем backward
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()  # Граф освобождается
            self.actor_optim.step()

            # --- Обновление критика ---
            # Вычисляем все для критика
            V_critic = self.critic(batch_obs).squeeze()
            critic_loss = nn.MSELoss()(V_critic, batch_discounted_rewards)
            # Обнуляем градиенты критика и выполняем backward
            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()  # Граф освобождается
            self.critic_optim.step()'''

            '''
            # Версия с retain graph
            # --- Обновление актора ---
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            A_k = (batch_discounted_rewards - V.detach()).to(self.device)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            clip_loss = torch.min(
                ratios * A_k,
                torch.clamp(ratios, 1 - self.hyperparameters['clip'], 1 + self.hyperparameters['clip']) * A_k
            )
            actor_loss = -clip_loss.mean()
            self.actor_optim.zero_grad() 
            actor_loss.backward(retain_graph = True)  
            self.actor_optim.step()  

            # --- Обновление критика ---
            V = self.critic(batch_obs).squeeze()
            critic_loss = nn.MSELoss()(V, batch_discounted_rewards)
            self.critic_optim.zero_grad()  
            critic_loss.backward()  
            self.critic_optim.step() '''

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        # Логирование
        self.logger_dict['actor_losses'].append(np.mean(actor_losses))
        self.logger_dict['batch_rewards'].append(batch_discounted_rewards.mean().item())
        self.logger_dict['batch_lens'].append(np.mean(batch_lens))

        self.log("actor_loss", np.mean(actor_losses), prog_bar=True)
        self.log("critic_loss", np.mean(critic_losses), prog_bar=True)
        self.log("avg_reward", batch_discounted_rewards.mean().item(), prog_bar=True)

        return {"actor_loss": np.mean(actor_losses), "critic_loss": np.mean(critic_losses)}

    def configure_optimizers(self):
        self.actor_optim = Adam(self.actor.parameters(), lr=self.hyperparameters['learning_rate'])
        self.critic_optim = Adam(self.critic.parameters(), lr=self.hyperparameters['learning_rate'])
        return [self.actor_optim, self.critic_optim]

    def rollout(self):
        batch_obs, batch_acts, batch_log_probs = [], [], []
        batch_rews, batch_lens = [], []
        #ep_rews = []
        current_timestep = 0

        while len(batch_obs) < self.hyperparameters['timesteps_per_batch']:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False

            for _ in range(self.hyperparameters['max_timesteps_per_episode']):
                current_timestep += 1
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action, log_prob = self.get_action(obs_tensor)
                obs, rew, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                batch_obs.append(obs_tensor)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(rew)
                if done:
                    break

            batch_lens.append(len(ep_rews))
            batch_rews.append(ep_rews)

        batch_obs = torch.stack(batch_obs).to(self.device)
        batch_acts = torch.stack(batch_acts).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)
        batch_rtgs = self.compute_discounted_rewards(batch_rews).to(self.device)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_discounted_rewards(self, batch_rews):
        #batch_rtgs = []
        batch_discounted_rewards = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.hyperparameters['gamma']
                #batch_rtgs.insert(0, discounted_reward)
                batch_discounted_rewards.insert(0, discounted_reward)
        #return torch.tensor(batch_rtgs, dtype=torch.float32)
        return torch.tensor(batch_discounted_rewards, dtype=torch.float32) # .to(self.device)

    def get_action(self, obs):
        mean = self.actor(obs)
        #dist = Normal(mean, torch.sqrt(self.cov_var).to(self.device))  # Используем Normal вместо MultivariateNormal
        cov_mat = self.cov_mat.to(self.device)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        #log_prob = dist.log_prob(action).sum(dim=-1)  # Суммируем логарифмические вероятности
        log_prob = dist.log_prob(action) #action.to(self.device)
        return action, log_prob
        #return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        #dist = Normal(mean, torch.sqrt(self.cov_var).to(self.device))
        cov_mat = self.cov_mat.to(self.device)
        dist = MultivariateNormal(mean, cov_mat)
        #log_probs = dist.log_prob(batch_acts).sum(dim=-1)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
