#==========================================
# File: rl_agent.py
#==========================================

"""
DDPG Agent with twin critics. Updated PER handling:
- Works with CPU buffers (move batches to self.device).
- β annealing is linear: beta(t) = beta0 + (1-beta0) * t/T
- Fix weights shape (use (B,1) once).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.optim.lr_scheduler import StepLR


class Actor(nn.Module):
    def __init__(self, obs_dim, lr_range, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(128, action_dim)
        self.lr_range = lr_range

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        raw = self.out(x)
        lr_min, lr_max = self.lr_range
        lr  = lr_min + (lr_max - lr_min) * torch.sigmoid(raw[:, 0:1])
        mix = F.softmax(raw[:, 1:4], dim=-1)
        usage = torch.sigmoid(raw[:, 4:5])
        return torch.cat([lr, mix, usage], dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, device="cuda:0"):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device if not isinstance(device, str) else torch.device(device)
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim, device=self.device) * self.mu

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state = self.state + dx
        return self.state


class DDPGAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.device = torch.device(config["device"])
        self.config = config

        self.actor        = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, config["curriculum"]["learning_rate_range"], action_dim).to(self.device)
        self.critic1        = Critic(obs_dim, action_dim).to(self.device)
        self.critic1_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic2        = Critic(obs_dim, action_dim).to(self.device)
        self.critic2_target = Critic(obs_dim, action_dim).to(self.device)

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        # optimizers & schedulers
        self.actor_optimizer   = optim.Adam(self.actor.parameters(),  lr=config["rl"]["actor_lr"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config["rl"]["critic_lr"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config["rl"]["critic_lr"])

        step  = config["rl"].get("lr_decay_steps", 200000)
        gamma = config["rl"].get("lr_decay_rate", 0.5)
        self.actor_scheduler   = StepLR(self.actor_optimizer,   step_size=step, gamma=gamma)
        self.critic1_scheduler = StepLR(self.critic1_optimizer, step_size=step, gamma=gamma)
        self.critic2_scheduler = StepLR(self.critic2_optimizer, step_size=step, gamma=gamma)

        self.ou_noise = OUNoise(action_dim, device=self.device)
        self.total_it = 0

        # hyperparams
        self.gamma = config["rl"]["gamma"]
        self.tau   = config["rl"]["tau"]
        self.policy_delay = config["rl"].get("policy_delay", 2)
        self.policy_noise = config["rl"].get("policy_noise", 0.2)
        self.noise_clip   = config["rl"].get("noise_clip", 0.5)
        self.max_updates  = max(1, int(config["rl"]["off_policy_updates"]))

        # exploration noise anneal
        self.exploration_noise_initial = config["rl"]["exploration_noise"]
        self.exploration_noise_decay_steps = config["rl"].get("exploration_noise_decay_steps", 300000)
        self.exploration_noise = self.exploration_noise_initial

    def _hard_update(self, tgt, src):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.copy_(s.data)

    def _soft_update(self, tgt, src):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.copy_(t.data * (1 - self.tau) + s.data * self.tau)

    def select_action(self, state, noise_enable=True):
        if torch.is_tensor(state):
            s = state.unsqueeze(0).to(self.device).float()
        else:
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s).squeeze(0)
        self.actor.train()

        if noise_enable:
            n = self.ou_noise.noise() * self.exploration_noise
            a[1:4] += n[1:4]
            a[4:5] += n[4:5]

        # Clip & renormalize
        lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
        a[0] = a[0].clamp(lr_min, lr_max)
        mix = a[1:4].clamp(min=0.0)
        ssum = mix.sum()
        a[1:4] = mix / ssum if ssum > 0 else torch.ones(3, device=self.device) / 3
        a[4] = a[4].clamp(0.0, 1.0)
        return a

    def _prepare_batch(self, batch):
        # batch comes from CPU buffer; move to device and add dims
        state, action, reward, next_state, done = batch
        s  = state.to(self.device).float()
        a  = action.to(self.device).float()
        r  = reward.to(self.device).float().unsqueeze(1)
        ns = next_state.to(self.device).float()
        d  = done.to(self.device).float().unsqueeze(1)
        return s, a, r, ns, d

    def critic_update_only(self, replay_buffer, batch_size):
        self.total_it += 1

        # PER beta anneal before sampling
        if self.config["rl"].get("per_enabled", False) and hasattr(replay_buffer, "set_beta"):
            beta0 = float(self.config["rl"].get("per_beta", 0.6))
            t = min(1.0, self.total_it / self.max_updates)
            replay_buffer.set_beta(beta0 + (1.0 - beta0) * t)

        if self.config["rl"].get("per_enabled", False):
            batch = replay_buffer.sample(batch_size)
            state, action, reward, next_state, done, weights, idxs = batch
            s, a, r, ns, d = self._prepare_batch((state, action, reward, next_state, done))
            w = weights.to(self.device).unsqueeze(1)
        else:
            s, a, r, ns, d = self._prepare_batch(replay_buffer.sample(batch_size))
            idxs, w = None, None

        with torch.no_grad():
            na = self.actor_target(ns)
            noise = (torch.randn_like(na) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = na + noise
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
            na[:, 0:1] = na[:, 0:1].clamp(lr_min, lr_max)
            na[:, 1:4] = na[:, 1:4].clamp(0.0, 1.0)
            
            # renormalize the 3-way mixture to the probability simplex
            mix = na[:, 1:4].clamp(min=0.0)
            sums = mix.sum(dim=1, keepdim=True).clamp(min=1e-6)
            na[:, 1:4] = mix / sums
            
            na[:, 4:5] = na[:, 4:5].clamp(0.0, 1.0)
            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1 - d) * torch.min(tq1, tq2)

        cq1 = self.critic1(s, a); cq2 = self.critic2(s, a)
        td1 = cq1 - y; td2 = cq2 - y
        
        # --- Conservative Q-Learning (small coefficient) ---
        cql_alpha = float(self.config["rl"].get("cql_alpha", 1e-3))  # 1e-4..3e-3
        if cql_alpha > 0.0:
            K = int(self.config["rl"].get("cql_num_samples", 10))
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]

            # Sample K random actions per state in the legal box → (B,K,A)
            rand = torch.rand(s.size(0), K, a.size(1), device=s.device)
            rand[..., 0:1] = lr_min + (lr_max - lr_min) * rand[..., 0:1]        # LR in range
            mix = rand[..., 1:4].clamp(min=0.0)
            mix = mix / mix.sum(dim=-1, keepdim=True).clamp(min=1e-6)           # project to simplex
            rand[..., 1:4] = mix
            rand[..., 4:5] = rand[..., 4:5].clamp(0.0, 1.0)                      # usage in [0,1]

            # Q(s, a_rand) → (B,K,1)
            s_tile = s.unsqueeze(1).expand(-1, K, -1).reshape(-1, s.size(1))
            rand_flat = rand.reshape(-1, a.size(1))
            q_rand1 = self.critic1(s_tile, rand_flat).reshape(-1, K, 1).squeeze(-1)
            q_rand2 = self.critic2(s_tile, rand_flat).reshape(-1, K, 1).squeeze(-1)

            # Per-state logsumexp over K
            lse1 = torch.logsumexp(q_rand1, dim=1).mean()
            lse2 = torch.logsumexp(q_rand2, dim=1).mean()

            # "Data/policy" term: encourage Q(s, a_data) & Q(s, π(s)) to be large
            with torch.no_grad():
                a_pi = self.actor(s)
                # clamp & re-normalize (same rules)
                a_pi[:, 0:1] = a_pi[:, 0:1].clamp(lr_min, lr_max)
                mix_pi = a_pi[:, 1:4].clamp(min=0.0)
                a_pi[:, 1:4] = mix_pi / mix_pi.sum(dim=1, keepdim=True).clamp(min=1e-6)
                a_pi[:, 4:5] = a_pi[:, 4:5].clamp(0.0, 1.0)

            q_data1 = self.critic1(s, a).mean()
            q_pi1   = self.critic1(s, a_pi).mean()
            q_data2 = self.critic2(s, a).mean()
            q_pi2   = self.critic2(s, a_pi).mean()

            # Conservative penalty
            cql1 = (lse1 - 0.5 * (q_data1 + q_pi1))
            cql2 = (lse2 - 0.5 * (q_data2 + q_pi2))
        else:
            cql1 = cq1.new_zeros(())
            cql2 = cq2.new_zeros(())

        if w is not None:
            loss1 = (td1.pow(2) * w).mean() + cql_alpha * cql1
            loss2 = (td2.pow(2) * w).mean() + cql_alpha * cql2
        else:
            loss1 = F.smooth_l1_loss(cq1, y) + cql_alpha * cql1
            loss2 = F.smooth_l1_loss(cq2, y) + cql_alpha * cql2

        self.critic1_optimizer.zero_grad(); loss1.backward()
        utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step(); self.critic1_scheduler.step()

        self.critic2_optimizer.zero_grad(); loss2.backward()
        utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step(); self.critic2_scheduler.step()
        self.actor_scheduler.step()

        if idxs is not None:
            eps = float(self.config["rl"].get("per_epsilon", 1e-6))
            new_prios = 0.5 * (td1.detach().abs() + td2.detach().abs())
            new_prios = new_prios.flatten().cpu() + eps
            replay_buffer.update_priorities(idxs, new_prios)

        return {"critic1_loss": loss1.item(), "critic2_loss": loss2.item()}

    def update(self, replay_buffer, batch_size):
        self.total_it += 1

        # PER beta anneal before sampling
        if self.config["rl"].get("per_enabled", False) and hasattr(replay_buffer, "set_beta"):
            beta0 = float(self.config["rl"].get("per_beta", 0.6))
            t = min(1.0, self.total_it / self.max_updates)
            replay_buffer.set_beta(beta0 + (1.0 - beta0) * t)

        if self.config["rl"].get("per_enabled", False):
            state, action, reward, next_state, done, weights, idxs = replay_buffer.sample(batch_size)
            s, a, r, ns, d = self._prepare_batch((state, action, reward, next_state, done))
            w = weights.to(self.device).unsqueeze(1)
        else:
            s, a, r, ns, d = self._prepare_batch(replay_buffer.sample(batch_size))
            idxs, w = None, None

        with torch.no_grad():
            na = self.actor_target(ns)
            noise = (torch.randn_like(na) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = na + noise
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
            na[:, 0:1] = na[:, 0:1].clamp(lr_min, lr_max)
            na[:, 1:4] = na[:, 1:4].clamp(0.0, 1.0)
            
            # renormalize the 3-way mixture to the probability simplex
            mix = na[:, 1:4].clamp(min=0.0)
            sums = mix.sum(dim=1, keepdim=True).clamp(min=1e-6)
            na[:, 1:4] = mix / sums
            
            na[:, 4:5] = na[:, 4:5].clamp(0.0, 1.0)
            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1 - d) * torch.min(tq1, tq2)

        cq1 = self.critic1(s, a); cq2 = self.critic2(s, a)
        td1 = cq1 - y; td2 = cq2 - y
        
        # --- Conservative Q-Learning (small coefficient) ---
        cql_alpha = float(self.config["rl"].get("cql_alpha", 1e-3))  # 1e-4..3e-3
        if cql_alpha > 0.0:
            K = int(self.config["rl"].get("cql_num_samples", 10))
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]

            # Sample K random actions per state in the legal box → (B,K,A)
            rand = torch.rand(s.size(0), K, a.size(1), device=s.device)
            rand[..., 0:1] = lr_min + (lr_max - lr_min) * rand[..., 0:1]        # LR in range
            mix = rand[..., 1:4].clamp(min=0.0)
            mix = mix / mix.sum(dim=-1, keepdim=True).clamp(min=1e-6)           # project to simplex
            rand[..., 1:4] = mix
            rand[..., 4:5] = rand[..., 4:5].clamp(0.0, 1.0)                      # usage in [0,1]

            # Q(s, a_rand) → (B,K,1)
            s_tile = s.unsqueeze(1).expand(-1, K, -1).reshape(-1, s.size(1))
            rand_flat = rand.reshape(-1, a.size(1))
            q_rand1 = self.critic1(s_tile, rand_flat).reshape(-1, K, 1).squeeze(-1)
            q_rand2 = self.critic2(s_tile, rand_flat).reshape(-1, K, 1).squeeze(-1)

            # Per-state logsumexp over K
            lse1 = torch.logsumexp(q_rand1, dim=1).mean()
            lse2 = torch.logsumexp(q_rand2, dim=1).mean()

            # "Data/policy" term: encourage Q(s, a_data) & Q(s, π(s)) to be large
            with torch.no_grad():
                a_pi = self.actor(s)
                # clamp & re-normalize (same rules)
                a_pi[:, 0:1] = a_pi[:, 0:1].clamp(lr_min, lr_max)
                mix_pi = a_pi[:, 1:4].clamp(min=0.0)
                a_pi[:, 1:4] = mix_pi / mix_pi.sum(dim=1, keepdim=True).clamp(min=1e-6)
                a_pi[:, 4:5] = a_pi[:, 4:5].clamp(0.0, 1.0)

            q_data1 = self.critic1(s, a).mean()
            q_pi1   = self.critic1(s, a_pi).mean()
            q_data2 = self.critic2(s, a).mean()
            q_pi2   = self.critic2(s, a_pi).mean()

            # Conservative penalty
            cql1 = (lse1 - 0.5 * (q_data1 + q_pi1))
            cql2 = (lse2 - 0.5 * (q_data2 + q_pi2))
        else:
            cql1 = cq1.new_zeros(())
            cql2 = cq2.new_zeros(())

        if w is not None:
            loss1 = (td1.pow(2) * w).mean() + cql_alpha * cql1
            loss2 = (td2.pow(2) * w).mean() + cql_alpha * cql2
        else:
            loss1 = F.smooth_l1_loss(cq1, y) + cql_alpha * cql1
            loss2 = F.smooth_l1_loss(cq2, y) + cql_alpha * cql2    
            

        # Optimize critics
        self.critic1_optimizer.zero_grad(); loss1.backward()
        utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step(); self.critic1_scheduler.step()

        self.critic2_optimizer.zero_grad(); loss2.backward()
        utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step(); self.critic2_scheduler.step()

        # Delayed actor + targets
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            a_curr = self.actor(s)
            q_term = -self.critic1(s, a_curr).mean()

            # (i) Entropy on the 3-way mixture (already added previously)
            mix = a_curr[:, 1:4].clamp(min=1e-8)
            mix_entropy = -(mix * mix.log()).sum(dim=1).mean()
            ent_beta_mix = float(self.config["rl"].get("entropy_beta", 1e-3))  # 1e-3..3e-3

            # (ii) Entropy on the scalar usage (Bernoulli entropy)
            usage = a_curr[:, 4:5].clamp(1e-6, 1 - 1e-6)
            usage_entropy = -(usage * usage.log() + (1 - usage) * (1 - usage).log()).mean()
            ent_beta_usage = float(self.config["rl"].get("usage_entropy_beta", 1e-3))

            # (iii) Soften LR extremes with a small quadratic around the mid-point of its range
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
            lr_center = 0.5 * (lr_min + lr_max)
            lr_span   = max(1e-12, (lr_max - lr_min))
            lr = a_curr[:, 0:1]
            lr_pen = ((lr - lr_center) / lr_span).pow(2).mean()
            lr_lambda = float(self.config["rl"].get("lr_center_penalty", 1e-3))  # 1e-3..5e-3

            # Optional nudge to keep usage near a target (default off)
            usage_tgt  = float(self.config["rl"].get("target_usage", 0.5))
            usage_lmbd = float(self.config["rl"].get("usage_penalty", 0.0))      # 0.0 = disabled
            usage_pen  = ((usage - usage_tgt) ** 2).mean()

            actor_loss = (
                q_term
                - ent_beta_mix   * mix_entropy
                - ent_beta_usage * usage_entropy
                + lr_lambda      * lr_pen
                + usage_lmbd     * usage_pen
            )

            self.actor_optimizer.zero_grad(); actor_loss.backward()
            utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        self.actor_scheduler.step()

        # PER priority update
        if idxs is not None:
            eps = float(self.config["rl"].get("per_epsilon", 1e-6))
            new_prios = 0.5 * (td1.detach().abs() + td2.detach().abs())
            new_prios = new_prios.flatten().cpu() + eps
            replay_buffer.update_priorities(idxs, new_prios)

        return {
            "actor_loss":  actor_loss.item() if actor_loss is not None else None,
            "critic1_loss": loss1.item(),
            "critic2_loss": loss2.item(),
        }
