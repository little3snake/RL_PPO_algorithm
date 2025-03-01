import torch

class Memory:

    def __init__(self, device, obs_size, action_size, **kwargs):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.obs_size = obs_size
        self.action_size = action_size
        print("Memory kwargs:", kwargs)

        self.batch_size = kwargs.get('timesteps_per_batch', 64)
        #print ("batch_size", self.batch_size)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        # in one batch
        self.observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)
        self.actions = torch.zeros(self.batch_size, self.action_size).to(self.device)
        #self.logprobs = torch.zeros(self.batch_size, self.action_size).to(self.device)
        self.logprobs = torch.zeros(self.batch_size).to(self.device)
        self.next_observations = torch.zeros(self.batch_size, self.obs_size).to(self.device) # for disc_rew
        self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.dones = torch.zeros(self.batch_size).to(self.device)
        self.advantages = torch.zeros(self.batch_size).to(self.device)
        self.discounted_rewards = torch.zeros(self.batch_size + 1).to(self.device)
        self.V = torch.zeros(self.batch_size).to(self.device) # values form critic -- old_value_state

    def add(self, obs, action, logprob, new_obs, reward, done, current_step):
        self.observations[current_step] = torch.Tensor(obs).to(self.device)
        self.actions[current_step] = torch.Tensor(action).to(self.device)
        self.logprobs[current_step] = logprob.detach()

        #                            HOW
        self.next_observations[current_step] = torch.Tensor(new_obs).to(self.device)
        self.rewards[current_step] = torch.Tensor((reward, )).squeeze(-1).to(self.device)
        self.dones[current_step] = torch.Tensor((int(done is True), )).squeeze(-1).to(self.device)

    def set_V (self, V):
        self.V = V

    def get_logprobs(self):
        return self.logprobs
    def get_advantages(self):
        return self.advantages
    def get_discounted_rewards(self):
        return self.discounted_rewards

    def set_gae_advantages (self, V_s, V_next):
        # advantages in memory - discounted rewards too
        #V_next.detach()
        #self.V.detach()

        self.discounted_rewards[self.batch_size] = V_next[-1]
        last_advantage = 0
        last_value = V_next[-1]
        for t in reversed(range(self.batch_size)):
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
        self.observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)
        self.actions = torch.zeros(self.batch_size, self.action_size).to(self.device)
        self.logprobs = torch.zeros(self.batch_size).to(self.device)
        self.next_observations = torch.zeros(self.batch_size, self.obs_size).to(self.device)  # for disc_rew
        self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.dones = torch.zeros(self.batch_size).to(self.device)
        self.advantages = torch.zeros(self.batch_size).to(self.device)
        self.discounted_rewards = torch.zeros(self.batch_size + 1).to(self.device)
        self.V = torch.zeros(self.batch_size).to(self.device)  # values form critic

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
