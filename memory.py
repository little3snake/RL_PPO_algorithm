import torch

class Memory:

    def __init__(self, device, obs_size, action_size, **kwargs):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.obs_size = obs_size
        self.action_size = action_size
        print("Memory kwargs:", kwargs)

        self.current_batch_size = 0 # init later in finalize
        #print ("batch_size", self.batch_size)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        # in one batch
        #self.observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)
        self.observations = []
        #self.actions = torch.zeros(self.batch_size, self.action_size).to(self.device)
        self.actions = []
        #self.logprobs = torch.zeros(self.batch_size).to(self.device)
        self.logprobs = []
        #self.next_observations = torch.zeros(self.batch_size, self.obs_size).to(self.device) # for disc_rew
        self.next_observations = []
        #self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.rewards = []
        #self.dones = torch.zeros(self.batch_size).to(self.device)
        self.dones = []
        #self.advantages = torch.zeros(self.batch_size).to(self.device)
        self.advantages = []
        #self.discounted_rewards = torch.zeros(self.batch_size + 1).to(self.device)
        self.discounted_rewards = []
        #self.V = torch.zeros(self.batch_size).to(self.device) # values form critic -- old_value_state
        self.V = []

    def add(self, obs, action, logprob, new_obs, reward, done, current_step):
        '''self.observations[current_step] = torch.Tensor(obs).to(self.device)
        self.actions[current_step] = torch.Tensor(action).to(self.device)
        self.logprobs[current_step] = logprob.detach()
        self.next_observations[current_step] = torch.Tensor(new_obs).to(self.device)
        self.rewards[current_step] = torch.Tensor((reward, )).squeeze(-1).to(self.device)
        self.dones[current_step] = torch.Tensor((int(done is True), )).squeeze(-1).to(self.device)'''
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        logprob_tensor = logprob.detach()  # Отключаем граф вычислений для logprob
        new_obs_tensor = torch.tensor(new_obs, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device).squeeze(-1)
        done_tensor = torch.tensor([int(done)], dtype=torch.float32, device=self.device).squeeze(-1)
        # Добавляем данные в списки
        self.observations.append(obs_tensor)
        self.actions.append(action_tensor)
        self.logprobs.append(logprob_tensor)
        self.next_observations.append(new_obs_tensor)
        self.rewards.append(reward_tensor)
        self.dones.append(done_tensor)

    def finalize(self):
        self.observations = torch.stack(self.observations)
        self.actions = torch.stack(self.actions)
        self.logprobs = torch.stack(self.logprobs)
        self.next_observations = torch.stack(self.next_observations)
        self.rewards = torch.stack(self.rewards)
        self.dones = torch.stack(self.dones)

        self.current_batch_size = len(self.dones)
        self.advantages = torch.zeros(self.current_batch_size, device=self.device)
        self.discounted_rewards = torch.zeros(self.current_batch_size + 1, device=self.device)
        self.V = torch.zeros(self.current_batch_size, device=self.device)

    def set_V (self, V):
        self.V = V

    def get_logprobs(self):
        return self.logprobs
    def get_advantages(self):
        return self.advantages
    def get_discounted_rewards(self):
        return self.discounted_rewards
    def get_current_batch_size(self):
        return self.current_batch_size

    def set_gae_advantages (self, V_s, V_next):
        # advantages in memory - discounted rewards too
        #V_next.detach()
        #self.V.detach()

        self.discounted_rewards[self.current_batch_size] = V_next[-1]
        last_advantage = 0
        last_value = V_next[-1]
        for t in reversed(range(self.current_batch_size)):
            mask = 1.0 - self.dones[t]
            last_value = last_value * mask
            last_advantage = last_advantage + mask
            #delta = self.rewards[t] + self.gamma * last_value - self.V[t]
            delta = self.rewards[t] + self.gamma * last_value - V_s[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            self.advantages[t] = last_advantage
            last_value = V_next[t]
            self.discounted_rewards[t] = self.rewards[t] + self.gamma * self.discounted_rewards[t+1] * mask
        #return self.advantages
        # Normalization - update -- in minibatches
        #self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-10)

    def clear (self):
        '''self.observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)
        self.actions = torch.zeros(self.batch_size, self.action_size).to(self.device)
        self.logprobs = torch.zeros(self.batch_size).to(self.device)
        self.next_observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)  # for disc_rew
        self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.dones = torch.zeros(self.batch_size).to(self.device)
        self.advantages = torch.zeros(self.batch_size).to(self.device)
        self.discounted_rewards = torch.zeros(self.batch_size + 1).to(self.device)
        self.V = torch.zeros(self.batch_size).to(self.device)  # values form critic'''

        self.observations = []
        self.actions = []
        self.logprobs = []
        self.next_observations = []
        self.rewards = []
        self.dones = []

        self.advantages = None
        self.discounted_rewards = None
        self.V = None

    '''def set_old_value_state(self, old_v_s):
        self.old_value_state = old_v_s

    def calculate_advantage(self, next_value, values):
        gt = next_value
        for i in reversed(range(self.batch_size)):
            gt = self.rewards[i] + Config.GAMMA * gt * (1 - self.dones[i])
            self.gt[i] = gt
            self.advantages[i] = gt - values[i]

    def calculate_gae_advantage(self, values, next_values):
        self.gt[self.batch_size] = next_values[-1]
        for i in reversed(range(self.batch_size)):
            delta = self.rewards[i] + Config.GAMMA * next_values[i] * (1 - self.dones[i]) - values[i]
            self.advantages[i] = delta + Config.LAMBDA * Config.GAMMA * self.advantages[i+1] * (1 - self.dones[i])
            # For critic
            self.gt[i] = self.rewards[i] + Config.GAMMA * self.gt[i+1] * (1 - self.dones[i])'''
