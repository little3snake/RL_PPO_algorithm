import torch
from torch.utils.data import IterableDataset, DataLoader

class RolloutDataset(IterableDataset):
    def __init__(self, env, actor, timesteps_per_batch, max_timesteps_per_episode, device, compute_discounted_rewards):
        """
        env: окружение для взаимодействия.
        actor: политика агента.
        timesteps_per_batch: количество шагов в каждом батче.
        max_timesteps_per_episode: максимальное количество шагов в одном эпизоде.
        device: устройство (CPU или GPU).
        compute_discounted_rewards: функция для вычисления дисконтированных наград.
        """
        super().__init__()
        self.env = env
        self.actor = actor
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.device = device
        #self.device = "cpu"
        print ("Rollout device", self.device)
        self.compute_discounted_rewards = compute_discounted_rewards

    def __iter__(self):
        """
        Генерация данных: собираем rollout, возвращаем состояния, действия, награды и т.д.
        """
        batch_observations, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_lens = [], []
        current_timestep = 0

        while current_timestep < self.timesteps_per_batch:
            episode_rewards = []
            observation, _ = self.env.reset()  # Сброс среды
            observation = torch.tensor(observation, dtype=torch.float, device=self.device)
            done = False

            for _ in range(self.max_timesteps_per_episode):
                current_timestep += 1
                batch_observations.append(observation)
                with torch.no_grad():
                    action, log_prob = self.actor.get_action(observation)

                observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                observation = torch.tensor(observation, dtype=torch.float, device=self.device)
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if terminated or truncated:
                    break

            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)

        # Преобразуем списки в тензоры
        batch_observations = torch.stack(batch_observations).to(self.device)
        batch_actions = torch.stack(batch_actions).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)
        batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards).to(self.device)

        # Возвращаем данные по одному элементу за раз
        for i in range(len(batch_observations)):
            yield batch_observations[i], batch_actions[i], batch_log_probs[i], batch_discounted_rewards[i], batch_lens[i // self.max_timesteps_per_episode]