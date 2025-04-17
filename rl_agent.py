"""
DDPG Agent implementation with actor and critic networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the Actor network.
class Actor(nn.Module):
    def __init__(self, obs_dim, lr_range, action_dim=5):
        """
        Actor network mapping observation to final action parameters.
        
        The network outputs:
          - Learning rate: scaled to be in the range [lr_range[0], lr_range[1]].
          - Mixing ratios: softmax over the three values (indices 1-3).
          - Sample usage fraction: a number between 0 and 1 (via sigmoid) that is later scaled.
        
        Args:
            obs_dim (int): Dimension of the observation vector.
            lr_range (list or tuple): [min_lr, max_lr] defining the learning rate range.
            action_dim (int): Dimension of the action vector (default 5).
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)
        self.lr_range = lr_range  # e.g., [0.001, 0.1]
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.out(x)  # shape: (batch, 5)
        # Map index 0 to a learning rate in [min_lr, max_lr]
        lr_offset = self.lr_range[0]
        lr_max = self.lr_range[1]
        # Sigmoid outputs a value in (0, 1) that we scale to the desired range.
        lr = lr_offset + (lr_max - lr_offset) * torch.sigmoid(raw[:, 0:1])
        # Convert indices 1-3 to valid mixing ratios via softmax.
        mix = F.softmax(raw[:, 1:4], dim=-1)
        # For sample usage (index 4), use a sigmoid to have values between 0 and 1.
        sample_usage = torch.sigmoid(raw[:, 4:5])
        action = torch.cat([lr, mix, sample_usage], dim=-1)
        return action

# Define the Critic network.
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim=5):
        """
        Critic network estimating the Q-value for an observation-action pair.
        
        Args:
            obs_dim (int): Dimension of the observation vector.
            action_dim (int): Dimension of the action vector (now 5).
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value
    
# Ornstein-Uhlenbeck noise for exploration.
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state

class DDPGAgent:
    def __init__(self, obs_dim, action_dim, config):
        """
        DDPG Agent with actor and critic networks.
        
        Args:
            obs_dim (int): Dimension of observation space.
            action_dim (int): Dimension of action space (should be 15).
            config (dict): RL related hyperparameters from the configuration file.
        """
        self.device = torch.device(config["device"])

        self.lr_range = config["curriculum"]["learning_rate_range"]
        
        self.actor = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)

        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["rl"]["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["rl"]["critic_lr"])
        
        # Copy weights from the networks to the targets.
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        self.gamma = config["rl"]["gamma"]
        self.tau = config["rl"]["tau"]
        self.exploration_noise = config["rl"]["exploration_noise"]
        self.ou_noise = OUNoise(action_dim)
        
    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state, noise_enable=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if noise_enable:
            noise = self.ou_noise.noise()
            # Only perturb mixing ratios (1–3) and sample usage (4)
            action[1:4] += noise[1:4] * self.exploration_noise
            action[4:5] += noise[4:5] * self.exploration_noise

        # 1) Clip learning rate to valid range
        min_lr, max_lr = self.lr_range
        action[0] = float(np.clip(action[0], min_lr, max_lr))

        # 2) Renormalize mixing ratios to be non-negative & sum→1
        mix = action[1:4]
        mix = np.maximum(mix, 0.0)
        s = mix.sum()
        if s > 0:
            mix /= s
        else:
            mix = np.ones_like(mix) / len(mix)
        action[1:4] = mix

        # 3) Clip sample usage fraction to [0,1]
        action[4] = float(np.clip(action[4], 0.0, 1.0))

        return action

    
    def _softmax_over_head(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def update(self, replay_buffer, batch_size):
        """
        Update actor and critic networks based on samples from the replay buffer.
        
        Args:
            replay_buffer (ReplayBuffer): The replay buffer instance.
            batch_size (int): The size of the batch to sample.
        """
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        
        # Critic loss.
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            y = reward + self.gamma * (1 - done) * target_q
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss.
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft-update targets.
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

        
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
