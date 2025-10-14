#==========================================
# File: rl_agent.py
#==========================================

"""
DDPG Agent with twin critics. Updated PER handling:
- Works with CPU buffers (move batches to self.device).
- β annealing is linear: beta(t) = beta0 + (1-beta0) * t/T
- Fix weights shape (use (B,1) once).
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def _project_mixture(mix, eps):
    """
    Project mixture logits to the simplex with an optional floor epsilon.
    """
    s = mix.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    mix = mix / s
    if eps > 0:
        k = mix.size(-1)
        mix = (1.0 - k * eps) * mix + eps
    return mix


class Actor(nn.Module):
    def __init__(self, obs_dim, lr_range, action_dim=5, mix_temp=2.0, mix_floor=0.05):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(128, action_dim)
        self.lr_range = lr_range
        self.mix_temp = max(1e-6, float(mix_temp))
        self.mix_floor = float(mix_floor)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        raw = self.out(x)
        lr_min, lr_max = self.lr_range
        lr  = lr_min + (lr_max - lr_min) * torch.sigmoid(raw[:, 0:1])
        logits = raw[:, 1:4] / self.mix_temp
        mix = F.softmax(logits, dim=-1)
        mix = _project_mixture(mix, self.mix_floor)
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

        self.mix_temp = float(self.config["rl"].get("mix_temp", 2.0))
        self.mix_floor = float(self.config["rl"].get("mix_floor", 0.05))

        lr_range = config["curriculum"]["learning_rate_range"]
        self.actor        = Actor(obs_dim, lr_range, action_dim, mix_temp=self.mix_temp, mix_floor=self.mix_floor).to(self.device)
        self.actor_target = Actor(obs_dim, lr_range, action_dim, mix_temp=self.mix_temp, mix_floor=self.mix_floor).to(self.device)
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

        sched_cfg = config["rl"]
        sched_type = str(sched_cfg.get("lr_scheduler", "step")).lower()
        decay_steps = max(1, int(sched_cfg.get("lr_decay_steps", 200000)))
        gamma = float(sched_cfg.get("lr_decay_rate", 0.5))
        eta_min = float(sched_cfg.get("lr_min_lr", 0.0))

        def _make_scheduler(optimizer):
            if sched_type == "cosine":
                return CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
            return StepLR(optimizer, step_size=decay_steps, gamma=gamma)

        self.actor_scheduler   = _make_scheduler(self.actor_optimizer)
        self.critic1_scheduler = _make_scheduler(self.critic1_optimizer)
        self.critic2_scheduler = _make_scheduler(self.critic2_optimizer)

        self.ou_noise = OUNoise(action_dim, device=self.device)
        self.total_it = 0

        # hyperparams
        self.gamma = config["rl"]["gamma"]
        self.tau   = config["rl"]["tau"]
        self.policy_delay = config["rl"].get("policy_delay", 2)
        self.policy_noise = config["rl"].get("policy_noise", 0.2)
        self.noise_clip   = config["rl"].get("noise_clip", 0.5)
        self.max_updates  = max(1, int(config["rl"]["off_policy_updates"]))

        # --- TD3+BC knobs ---
        self.td3bc_lambda = float(self.config["rl"].get("td3bc_lambda", 0.0))
        self.td3bc_qfilter = bool(self.config["rl"].get("td3bc_qfilter", True))

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

    def _project_action(self, a):
        lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
        a[:, 0:1] = a[:, 0:1].clamp(lr_min, lr_max)
        mix = a[:, 1:4].clamp(min=0.0)
        mix = _project_mixture(mix, eps=self.mix_floor)
        a[:, 1:4] = mix
        a[:, 4:5] = a[:, 4:5].clamp(0.0, 1.0)
        return a

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

        a = self._project_action(a.unsqueeze(0)).squeeze(0)
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
            t = min(1.0, (self.total_it / self.max_updates) ** 0.5)
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
            na = self._project_action(na)
            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1 - d) * torch.min(tq1, tq2)

        cq1 = self.critic1(s, a); cq2 = self.critic2(s, a)
        td1 = cq1 - y; td2 = cq2 - y
        
        # --- Conservative Q-Learning (small coefficient) ---
        cql_alpha = float(self.config["rl"].get("cql_alpha", 1e-3))
        if cql_alpha > 0.0:
            K = int(self.config["rl"].get("cql_num_samples", 10))
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]

            # Random K samples
            rand = torch.rand(s.size(0), K, a.size(1), device=s.device)
            rand[..., 0:1] = lr_min + (lr_max - lr_min) * rand[..., 0:1]
            rand_mix = _project_mixture(rand[..., 1:4].clamp(min=0.0), eps=self.mix_floor)
            rand[..., 1:4] = rand_mix
            rand[..., 4:5] = rand[..., 4:5].clamp(0.0, 1.0)

            # Also include current policy action and a noisy variant
            with torch.no_grad():
                a_pi = self.actor(s)
                a_pi = self._project_action(a_pi)

                a_noisy = a_pi + (torch.randn_like(a_pi) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a_noisy = self._project_action(a_noisy)

            cand = torch.cat([rand, a_pi.unsqueeze(1), a_noisy.unsqueeze(1)], dim=1)
            B, Kp2, A = cand.shape
            s_rep = s.unsqueeze(1).expand(-1, Kp2, -1).reshape(-1, s.size(1))
            a_rep = cand.reshape(-1, A)

            q_rand1 = self.critic1(s_rep, a_rep).reshape(B, Kp2, 1).squeeze(-1)
            q_rand2 = self.critic2(s_rep, a_rep).reshape(B, Kp2, 1).squeeze(-1)

            min_q_rand = torch.min(q_rand1, q_rand2)  # [B, K+2]
            lse = torch.logsumexp(min_q_rand, dim=1) - math.log(min_q_rand.size(1))
            lse = lse.mean()

            q_data = torch.min(self.critic1(s, a), self.critic2(s, a)).mean()
            q_pi = torch.min(self.critic1(s, a_pi), self.critic2(s, a_pi)).mean()
            gap = torch.relu(q_pi - q_data)

            cql_pen = (lse - q_data) + 0.25 * gap
        else:
            cql_pen = s.new_zeros(())

        if w is not None:
            loss1 = (td1.pow(2) * w).mean() + cql_alpha * cql_pen
            loss2 = (td2.pow(2) * w).mean() + cql_alpha * cql_pen
        else:
            loss1 = F.smooth_l1_loss(cq1, y) + cql_alpha * cql_pen
            loss2 = F.smooth_l1_loss(cq2, y) + cql_alpha * cql_pen

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
            t = min(1.0, (self.total_it / self.max_updates) ** 0.5)
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
            na = self._project_action(na)
            tq1 = self.critic1_target(ns, na)
            tq2 = self.critic2_target(ns, na)
            y   = r + self.gamma * (1 - d) * torch.min(tq1, tq2)

        cq1 = self.critic1(s, a); cq2 = self.critic2(s, a)
        td1 = cq1 - y; td2 = cq2 - y
        
        # --- Conservative Q-Learning (small coefficient) ---
        cql_alpha = float(self.config["rl"].get("cql_alpha", 1e-3))
        if cql_alpha > 0.0:
            K = int(self.config["rl"].get("cql_num_samples", 10))
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]

            # Random K samples
            rand = torch.rand(s.size(0), K, a.size(1), device=s.device)
            rand[..., 0:1] = lr_min + (lr_max - lr_min) * rand[..., 0:1]
            rand_mix = _project_mixture(rand[..., 1:4].clamp(min=0.0), eps=self.mix_floor)
            rand[..., 1:4] = rand_mix
            rand[..., 4:5] = rand[..., 4:5].clamp(0.0, 1.0)

            # Also include current policy action and a noisy variant
            with torch.no_grad():
                a_pi = self.actor(s)
                a_pi = self._project_action(a_pi)

                a_noisy = a_pi + (torch.randn_like(a_pi) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a_noisy = self._project_action(a_noisy)

            cand = torch.cat([rand, a_pi.unsqueeze(1), a_noisy.unsqueeze(1)], dim=1)
            B, Kp2, A = cand.shape
            s_rep = s.unsqueeze(1).expand(-1, Kp2, -1).reshape(-1, s.size(1))
            a_rep = cand.reshape(-1, A)

            q_rand1 = self.critic1(s_rep, a_rep).reshape(B, Kp2, 1).squeeze(-1)
            q_rand2 = self.critic2(s_rep, a_rep).reshape(B, Kp2, 1).squeeze(-1)

            min_q_rand = torch.min(q_rand1, q_rand2)  # [B, K+2]
            lse = torch.logsumexp(min_q_rand, dim=1) - math.log(min_q_rand.size(1))
            lse = lse.mean()

            q_data = torch.min(self.critic1(s, a), self.critic2(s, a)).mean()
            q_pi = torch.min(self.critic1(s, a_pi), self.critic2(s, a_pi)).mean()
            gap = torch.relu(q_pi - q_data)

            cql_pen = (lse - q_data) + 0.25 * gap
        else:
            cql_pen = s.new_zeros(())

        if w is not None:
            loss1 = (td1.pow(2) * w).mean() + cql_alpha * cql_pen
            loss2 = (td2.pow(2) * w).mean() + cql_alpha * cql_pen
        else:
            loss1 = F.smooth_l1_loss(cq1, y) + cql_alpha * cql_pen
            loss2 = F.smooth_l1_loss(cq2, y) + cql_alpha * cql_pen


        # Optimize critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        (loss1 + loss2).backward()
        utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic1_optimizer.step(); self.critic1_scheduler.step()
        self.critic2_optimizer.step(); self.critic2_scheduler.step()

        # Delayed actor + targets
        a_data = a.detach()
        actor_loss = None
        a_curr = None
        if self.total_it % self.policy_delay == 0:
            a_curr = self.actor(s)
            a_curr = self._project_action(a_curr)
            q1_curr = self.critic1(s, a_curr)
            q_term = -q1_curr.mean()

            # (i) Entropy on the 3-way mixture
            mix = a_curr[:, 1:4].clamp(min=1e-8)
            mix_entropy = -(mix * mix.log()).sum(dim=1).mean()
            ent_beta_mix = float(self.config["rl"].get("entropy_beta", 1e-3))

            # (ii) Bernoulli entropy on usage
            usage = a_curr[:, 4:5].clamp(1e-6, 1 - 1e-6)
            usage_entropy = -(usage * usage.log() + (1 - usage) * (1 - usage).log()).mean()
            ent_beta_usage = float(self.config["rl"].get("usage_entropy_beta", 1e-2))

            # (iii) LR center penalty
            lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
            lr_center = 0.5 * (lr_min + lr_max)
            lr_span = max(1e-12, (lr_max - lr_min))
            lr = a_curr[:, 0:1]
            lr_pen = ((lr - lr_center) / lr_span).pow(2).mean()
            lr_lambda = float(self.config["rl"].get("lr_center_penalty", 1e-3))

            # (iv) Optional usage target penalty
            usage_tgt = float(self.config["rl"].get("target_usage", 0.5))
            usage_lmbd = float(self.config["rl"].get("usage_penalty", 5e-3))
            usage_pen = ((usage - usage_tgt) ** 2).mean()

            # --- TD3+BC term (disabled if lambda=0) ---
            td3bc_term = 0.0
            if self.td3bc_lambda > 0.0:
                if self.td3bc_qfilter:
                    with torch.no_grad():
                        q_data = self.critic1(s, a_data)
                        mask = (q_data >= q1_curr).float()
                        if mask.mean() < 1e-3:
                            mask = torch.ones_like(mask)
                    td3bc_term = ((a_curr - a_data).pow(2).mean(dim=1) * mask.squeeze(-1)).mean()
                else:
                    td3bc_term = (a_curr - a_data).pow(2).mean()

            actor_loss = (
                q_term
                - ent_beta_mix * mix_entropy
                - ent_beta_usage * usage_entropy
                + lr_lambda * lr_pen
                + usage_lmbd * usage_pen
                + float(self.td3bc_lambda) * td3bc_term
            )

            # Optional action MMD regularization
            mmd_lambda = float(self.config["rl"].get("mmd_lambda", 0.0))
            if mmd_lambda > 0.0:
                sigma = float(self.config["rl"].get("mmd_sigma", 0.2))

                def rbf(x, y):
                    x2 = (x ** 2).sum(dim=1, keepdim=True)
                    y2 = (y ** 2).sum(dim=1, keepdim=True).t()
                    xy = x @ y.t()
                    d2 = x2 + y2 - 2 * xy
                    return torch.exp(-d2 / (2 * sigma ** 2))

                Kpp = rbf(a_curr, a_curr).mean()
                Kqq = rbf(a_data, a_data).mean()
                Kpq = rbf(a_curr, a_data).mean()
                mmd = Kpp + Kqq - 2 * Kpq
                actor_loss = actor_loss + mmd_lambda * mmd

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
        else:
            with torch.no_grad():
                a_curr = self.actor(s)
                a_curr = self._project_action(a_curr)

        self.actor_scheduler.step()

        # PER priority update
        if idxs is not None:
            eps = float(self.config["rl"].get("per_epsilon", 1e-6))
            new_prios = 0.5 * (td1.detach().abs() + td2.detach().abs())
            new_prios = new_prios.flatten().cpu() + eps
            replay_buffer.update_priorities(idxs, new_prios)

        # OOD metric: L2 distance to batch data action (proxy)
        ood_l2 = (a_curr.detach() - a_data).pow(2).sum(dim=1).sqrt().mean().item()

        return {
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
            "critic1_loss": loss1.item(),
            "critic2_loss": loss2.item(),
            "ood_l2": ood_l2,
        }
