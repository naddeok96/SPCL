"""
DDPG Agent implementation with actor and critic networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as utils

# Define the Actor network.
class Actor(nn.Module):
    def __init__(self, obs_dim, lr_range, action_dim=5):
        """
        Actor network mapping observation to final action parameters.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.2)           # ← added dropout
        self.out = nn.Linear(128, action_dim)
        self.lr_range = lr_range  # e.g., [0.001, 0.1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)                       # ← apply dropout here
        raw = self.out(x)                         # shape: (batch, 5)
        # Map index 0 to a learning rate in [min_lr, max_lr]
        lr_offset, lr_max = self.lr_range
        lr = lr_offset + (lr_max - lr_offset) * torch.sigmoid(raw[:, 0:1])
        # Convert indices 1–3 to valid mixing ratios via softmax.
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
        self.device = torch.device(config["device"])
        self.config = config

        # networks
        self.actor        = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)

        self.critic1        = Critic(obs_dim, action_dim).to(self.device)
        self.critic1_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic2        = Critic(obs_dim, action_dim).to(self.device)
        self.critic2_target = Critic(obs_dim, action_dim).to(self.device)

        # copy weights
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

        # optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=config["rl"]["actor_lr"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config["rl"]["critic_lr"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config["rl"]["critic_lr"])

        # noise & counters
        self.ou_noise = OUNoise(action_dim)
        self.total_it = 0
        # save for annealing
        self.exploration_noise_initial = config["rl"]["exploration_noise"]
        self.exploration_noise = self.exploration_noise_initial
        self.policy_delay   = config["rl"].get("policy_delay", 2)
        self.policy_noise   = config["rl"].get("policy_noise", 0.2)
        self.noise_clip     = config["rl"].get("noise_clip", 0.5)
        self.max_updates    = config["rl"]["off_policy_updates"]

        # hyperparams
        self.gamma = config["rl"]["gamma"]
        self.tau   = config["rl"]["tau"]

    def _hard_update(self, tgt, src):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.copy_(s.data)

    def _soft_update(self, tgt, src):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.copy_( t.data * (1 - self.tau) + s.data * self.tau )

    def select_action(self, state, noise_enable=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s).cpu().numpy().flatten()
        self.actor.train()

        if noise_enable:
            n = self.ou_noise.noise() * self.exploration_noise
            a[1:4] += n[1:4]
            a[4:5] += n[4:5]

        # clip & renormalize
        mn, mx = self.config["curriculum"]["learning_rate_range"]
        a[0] = float(np.clip(a[0], mn, mx))
        mix = np.maximum(a[1:4], 0.0)
        s = mix.sum()
        a[1:4] = mix / s if s>0 else np.ones(3)/3
        a[4] = float(np.clip(a[4], 0,1))
        return a
    
    def critic_update_only(self, replay_buffer, batch_size):
        """
        Perform a critic-only update step for *both* critic1 and critic2.
        Returns:
            {'critic1_loss': float, 'critic2_loss': float}
        """
        self.total_it += 1

        # 1) Sample (with PER if enabled)
        if self.config["rl"].get("per_enabled", False):
            state, action, reward, next_state, done, weights, idxs = replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            idxs, weights = None, None

        # 2) To tensors
        s  = torch.FloatTensor(state).to(self.device)
        a  = torch.FloatTensor(action).to(self.device)
        r  = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_state).to(self.device)
        d  = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # 3) Compute common TD‑target using both target critics
        with torch.no_grad():
            na = self.actor_target(ns)
            noise = (torch.randn_like(na) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = na + noise

            min_lr, max_lr = self.config["curriculum"]["learning_rate_range"]
            na[:, 0:1] = na[:, 0:1].clamp(min_lr, max_lr)   # learning rate
            na[:, 1:4] = na[:, 1:4].clamp(0.0, 1.0)          # mixing ratios
            na[:, 4:5] = na[:, 4:5].clamp(0.0, 1.0)          # sample usage

            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1 - d) * torch.min(tq1, tq2)

        # 4) Current Q & losses
        cq1 = self.critic1(s, a)
        cq2 = self.critic2(s, a)
        td1 = cq1 - y
        td2 = cq2 - y

        if weights is not None:
            loss1 = (td1.pow(2) * weights).mean()
            loss2 = (td2.pow(2) * weights).mean()
        else:
            loss1 = F.smooth_l1_loss(cq1, y)
            loss2 = F.smooth_l1_loss(cq2, y)

        # 5) Backprop critic1
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        # 6) Backprop critic2
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # 7) PER priority update if used (use td1)
        if idxs is not None:
            eps = float(self.config["rl"].get("per_epsilon", 1e-6))
            new_prios = td1.detach().abs().cpu().numpy().flatten() + eps
            replay_buffer.update_priorities(idxs, new_prios)

        return {
            "critic1_loss": loss1.item(),
            "critic2_loss": loss2.item()
        }

    def update(self, replay_buffer, batch_size):
        self.total_it += 1

        # 1) sample
        if self.config["rl"].get("per_enabled", False):
            state, action, reward, next_state, done, weights, idxs = replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
            # anneal β
            replay_buffer.beta = min(1.0,
                replay_buffer.beta + (1.0 - self.config["rl"]["per_beta"]) * (self.total_it/self.max_updates)
            )
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            weights = None

        # 2) to tensors
        s = torch.FloatTensor(state).to(self.device)
        a = torch.FloatTensor(action).to(self.device)
        r = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        ns= torch.FloatTensor(next_state).to(self.device)
        d = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # 3) compute target actions with smoothing
        with torch.no_grad():
            na = self.actor_target(ns)
            noise = (torch.randn_like(na)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            na = na + noise   

            min_lr, max_lr = self.config["curriculum"]["learning_rate_range"]
            na[:, 0:1] = na[:, 0:1].clamp(min_lr, max_lr)   # learning rate
            na[:, 1:4] = na[:, 1:4].clamp(0.0, 1.0)          # mixing ratios
            na[:, 4:5] = na[:, 4:5].clamp(0.0, 1.0)          # sample usage

            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1-d) * torch.min(tq1, tq2)

        # 4) critic losses
        cq1 = self.critic1(s,a)
        cq2 = self.critic2(s,a)
        td1 = cq1 - y; td2 = cq2 - y
        if weights is not None:
            loss1 = (td1.pow(2)*weights).mean()
            loss2 = (td2.pow(2)*weights).mean()
        else:
            loss1 = F.smooth_l1_loss(cq1, y)
            loss2 = F.smooth_l1_loss(cq2, y)

        # 5) optimize critics
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        utils.clip_grad_norm_(self.critic1.parameters(),1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss2.backward()
        utils.clip_grad_norm_(self.critic2.parameters(),1.0)
        self.critic2_optimizer.step()

        # 6) delayed actor & target updates
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            utils.clip_grad_norm_(self.actor.parameters(),1.0)
            self.actor_optimizer.step()

            # soft‑update all targets
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        # 7) anneal exploration noise
        self.exploration_noise = max(
            0.05,
            self.exploration_noise_initial * (1 - self.total_it/self.max_updates)
        )

        # 8) update priorities
        if self.config["rl"].get("per_enabled", False):
            new_prios = (td1.abs().detach().cpu().numpy().flatten() + self.config["rl"]["per_epsilon"])
            replay_buffer.update_priorities(idxs, new_prios)

        return {
            "actor_loss":  actor_loss.item()  if actor_loss is not None else None,
            "critic1_loss": loss1.item(),
            "critic2_loss": loss2.item()
        }



