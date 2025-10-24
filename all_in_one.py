#==========================================
# File: utils.py
#==========================================

#==========================================
# File: utils.py
#==========================================

"""
Utility functions for computing loss histograms, constructing observation vectors,
and other helper methods.
"""

import torch
import matplotlib.pyplot as plt

# --- tweakable max clip and alpha ---
_MAX_LOSS = 13.8
_ALPHA    = 2.0

# cache edges per (num_bins, device)
_edges_cache = {}

def _get_bin_edges(num_bins: int, device: torch.device):
    key = (num_bins, str(device))
    if key not in _edges_cache:
        rel = torch.linspace(0.0, 1.0, steps=num_bins+1, device=device)
        _edges_cache[key] = (rel ** _ALPHA) * _MAX_LOSS
    return _edges_cache[key]

def compute_loss_histogram(losses, num_bins: int, device="cuda:0"):
    """
    Vectorized 1D histogram on GPU via torch.bincount.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # to GPU tensor
    if not torch.is_tensor(losses):
        losses = torch.tensor(losses, device=device, dtype=torch.float32)
    else:
        losses = losses.to(device).float()
    losses = losses.flatten().clamp(0.0, _MAX_LOSS)

    # bucket boundaries
    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    # bin indices
    bins = torch.bucketize(losses, boundaries)

    # fast count
    hist = torch.bincount(bins, minlength=num_bins).to(device, torch.float32)
    tot  = hist.sum()
    if tot > 0:
        hist /= tot

    return hist, edges

def compute_dual_loss_histograms(losses_correct, losses_incorrect, num_bins: int, device="cuda:0"):
    """
    Joint correct/incorrect histograms, vectorized.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # pack into GPU tensors
    lc = torch.as_tensor(losses_correct or [], device=device, dtype=torch.float32).flatten()
    li = torch.as_tensor(losses_incorrect or [], device=device, dtype=torch.float32).flatten()
    lc = lc.clamp(0.0, _MAX_LOSS)
    li = li.clamp(0.0, _MAX_LOSS)

    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    bins_c = torch.bucketize(lc, boundaries)
    bins_i = torch.bucketize(li, boundaries)

    hist_c = torch.bincount(bins_c, minlength=num_bins).to(device, torch.float32)
    hist_i = torch.bincount(bins_i, minlength=num_bins).to(device, torch.float32)

    total = hist_c + hist_i
    tot_sum = total.sum()
    if tot_sum > 0:
        hist_c /= tot_sum
        hist_i /= tot_sum

    return hist_c, hist_i, edges

def plot_histogram(hist, edges=None, title=None, filename=None):
    """
    Plot a normalized histogram (Tensor) with optional variable-width bins.
    """
    plt.figure()
    hist_vals = hist.cpu().tolist()
    
    if edges is not None:
        edges_cpu = edges.cpu()
        centers = ((edges_cpu[:-1] + edges_cpu[1:]) / 2).tolist()
        widths  = (edges_cpu[1:] - edges_cpu[:-1]).tolist()
        plt.bar(centers, hist_vals, width=widths, align='center')
        plt.xticks(edges_cpu.tolist(), rotation=45)
    else:
        positions = list(range(len(hist_vals)))
        plt.bar(positions, hist_vals)

    if title:
        plt.title(title)
    plt.xlabel("Loss bins")
    plt.ylabel("Normalized frequency")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()


def select_top_percent(dataset, percent):
    """Select transitions from the top ``percent``% highest-reward trajectories."""
    rewards = dataset["rewards"].cpu()
    dones = dataset["dones"].cpu()

    ep_indices = []
    ep_returns = []
    start = 0
    ret = 0.0
    for i, r in enumerate(rewards):
        ret += float(r)
        if dones[i]:
            ep_indices.append((start, i + 1))
            ep_returns.append(ret)
            start = i + 1
            ret = 0.0
    if start < len(rewards):
        ep_indices.append((start, len(rewards)))
        ep_returns.append(ret)

    k = max(1, int(len(ep_returns) * percent / 100.0))
    top_idx = sorted(range(len(ep_returns)), key=lambda i: ep_returns[i], reverse=True)[:k]

    selected = []
    for idx in top_idx:
        s, e = ep_indices[idx]
        selected.extend(range(s, e))

    selected = torch.tensor(selected, dtype=torch.long)
    return {
        "states": dataset["states"][selected],
        "actions": dataset["actions"][selected],
        "rewards": dataset["rewards"][selected],
        "next_states": dataset["next_states"][selected],
        "dones": dataset["dones"][selected],
    }


def behavior_clone(policy, expert_data, epochs=5, batch_size=64, weight_decay=1e-4, device=None):
    """Fit ``policy`` to ``expert_data`` using supervised learning."""
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = device or next(policy.parameters()).device
    device = torch.device(device)

    ds = TensorDataset(expert_data["states"].to(device), expert_data["actions"].to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    policy.train()
    optimiser = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for s, a in loader:
            pred = policy(s)
            loss = loss_fn(pred, a)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()


def tune_value_function(policy, value_net, data_loader, config):
    """Train ``value_net`` off-policy with a CQL regularizer."""
    import torch.nn.functional as F
    import torch.optim as optim

    device = next(value_net.parameters()).device
    gamma = config.get("gamma", 0.99)
    alpha = config.get("cql_alpha", 1.0)

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()

    optimiser = optim.Adam(value_net.parameters(), lr=config.get("critic_lr", 3e-4))

    for states, actions, rewards, next_states, dones in data_loader:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)
        if dones.dtype == torch.bool:
            dones = dones.float()

        with torch.no_grad():
            next_a = policy(next_states)
            target_q = rewards + gamma * (1 - dones) * value_net(next_states, next_a)

        current_q = value_net(states, actions)
        td_loss = F.mse_loss(current_q, target_q)

        rand_a = torch.rand_like(actions)
        rand_a[:, 0:1] = rand_a[:, 0:1].clamp(*config.get("learning_rate_range", (0.0, 1.0)))
        rand_a[:, 1:4] = torch.softmax(rand_a[:, 1:4], dim=1)
        rand_a[:, 4:5] = rand_a[:, 4:5].clamp(0.0, 1.0)
        q_rand = value_net(states, rand_a)
        q_policy = value_net(states, policy(states))
        cql_loss = (torch.logsumexp(q_rand, dim=0).mean() - q_policy.mean()) * alpha

        loss = td_loss + cql_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


#==========================================
# File: replay_buffer.py
#==========================================

#==========================================
# File: replay_buffer.py
#==========================================

"""
Replay buffers (standard & Prioritized) designed to STORE ON CPU to avoid GPU OOM.
Includes helpers to stream very large datasets into the buffer in shards.
"""

from __future__ import annotations
import random
from typing import Dict, Iterable, Tuple

import torch
from tqdm import tqdm


Transition = Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]


def _to_cpu_tensor(x) -> torch.Tensor:
    """Ensure a float32 CPU tensor (except bool flags handled at call sites)."""
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.as_tensor(x, dtype=torch.float32, device="cpu")


class ReplayBuffer:
    """
    Simple FIFO replay buffer. **All data are stored on CPU**.
    Only sampled batches should be moved to GPU by the learner.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: list[Transition] = []
        self.position = 0  # write pointer

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition. Everything is stored on CPU tensors.
        """
        entry: Transition = (
            _to_cpu_tensor(state),
            _to_cpu_tensor(action),
            float(reward),
            _to_cpu_tensor(next_state),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idxs))
        # Assemble CPU tensors; the agent will .to(device) them as needed.
        return (
            torch.stack(states, dim=0),                               # (B, ...)
            torch.stack(actions, dim=0),                              # (B, ...)
            torch.as_tensor(rewards, dtype=torch.float32),            # (B,)
            torch.stack(next_states, dim=0),                          # (B, ...)
            torch.as_tensor(dones, dtype=torch.float32),              # (B,)
        )

    def __len__(self):
        return len(self.buffer)


class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (proportional or rank-based).
    Stored on CPU. Sampling returns (state, action, reward, next_state, done, isw, idxs)

    Args:
        alpha: priority exponent (0 -> uniform, 1 -> greedy)
        beta0: initial importance-sampling exponent
        per_type: "proportional" or "rank"
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta0: float = 0.4,
                 epsilon: float = 1e-6, per_type: str = "proportional"):
        super().__init__(capacity)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.beta = float(beta0)      # current annealed beta
        self.epsilon = float(epsilon)
        self.per_type = per_type
        self.priorities = torch.zeros(capacity, dtype=torch.float32)  # CPU

    def set_beta(self, new_beta: float):
        self.beta = float(new_beta)

    def push(self, state, action, reward, next_state, done):
        super().push(state, action, reward, next_state, done)
        idx = (self.position - 1) % self.capacity
        if len(self.buffer) > 0:
            max_p = self.priorities[:len(self.buffer)].max()
            max_p = float(max_p) if max_p > 0 else 1.0
        else:
            max_p = 1.0
        self.priorities[idx] = max_p

    def _probs(self) -> torch.Tensor:
        N = len(self.buffer)
        if N == 0:
            return torch.zeros(0, dtype=torch.float32)
        prios = self.priorities[:N]
        if self.per_type == "proportional":
            probs = (prios + self.epsilon).clamp_min(0) ** self.alpha
        else:
            # rank-based: convert to ranks (1=highest)
            ranks = torch.argsort(prios, descending=True).argsort().to(torch.float32) + 1.0
            probs = (1.0 / ranks) ** self.alpha
        s = probs.sum()
        if s <= 0:
            probs = torch.full_like(probs, 1.0 / N)
        else:
            probs = probs / s
        return probs

    def sample(self, batch_size: int):
        N = len(self.buffer)
        probs = self._probs()
        # multinomial on CPU
        idxs = torch.multinomial(probs, batch_size, replacement=True) if N > 0 else torch.empty(0, dtype=torch.long)
        weights = (N * probs[idxs]).clamp_min(1e-12) ** (-self.beta)
        weights = weights / weights.max().clamp_min(1e-12)  # normalize to 1

        # Gather transitions
        batch = [self.buffer[i] for i in idxs.tolist()]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states, dim=0),
            torch.stack(actions, dim=0),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.stack(next_states, dim=0),
            torch.as_tensor(dones, dtype=torch.float32),
            weights,      # (B,) on CPU
            idxs,         # (B,) long on CPU
        )

    def update_priorities(self, indices: Iterable[int | torch.Tensor], new_prios):
        """
        Vectorized priority update. new_prios can be list/ndarray/tensor.
        """
        if isinstance(indices, torch.Tensor):
            idxs = indices.detach().cpu().long()
        else:
            idxs = torch.as_tensor(list(indices), dtype=torch.long, device="cpu")
        newp = torch.as_tensor(new_prios, dtype=torch.float32, device="cpu").flatten()
        if newp.numel() == 1:
            newp = newp.repeat(len(idxs))
        self.priorities[idxs] = newp


# ------- Helpers to build/stream buffers from a (very large) dataset -------

def _iter_shards(n_total: int, shard_size: int):
    for start in range(0, n_total, shard_size):
        end = min(start + shard_size, n_total)
        yield start, end


def stream_into_buffer(
    buffer: ReplayBuffer | PERBuffer,
    data: Dict[str, torch.Tensor],
    shard_size: int = 200_000,
    shuffle: bool = True,
    *,
    show_progress: bool = False,
    progress_desc: str | None = None,
):
    """
    Stream a big dataset dict into the buffer in shards to limit peak memory.
    Expects tensors on CPU (or moveable to CPU). Uses FIFO overwrite when full.
    """
    states      = data["states"]
    actions     = data["actions"]
    rewards     = data["rewards"]
    next_states = data["next_states"]
    dones       = data["dones"]

    n = len(states)
    indices = torch.arange(n)
    if shuffle:
        indices = indices[torch.randperm(n)]

    progress = None
    if show_progress:
        progress = tqdm(total=n, desc=progress_desc or "Replay Buffer", leave=False)

    try:
        for s0, s1 in _iter_shards(n, shard_size):
            idx = indices[s0:s1]
            for i in idx.tolist():
                buffer.push(states[i], actions[i], float(rewards[i]), next_states[i], bool(dones[i]))
            if progress is not None:
                progress.update(int(idx.numel()))
    finally:
        if progress is not None:
            progress.close()


def build_replay_buffer_streaming(
    data: Dict[str, torch.Tensor],
    capacity: int,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_epsilon: float = 1e-6,
    per_type: str = "proportional",
    shard_size: int = 200_000,
    *,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> ReplayBuffer | PERBuffer:
    """
    Create a CPU buffer (PER or uniform) and stream *all* data into it in shards.
    Only up to `capacity` most-recent items will be retained (FIFO overwrite).
    """
    buffer: ReplayBuffer | PERBuffer
    if use_per:
        buffer = PERBuffer(capacity, alpha=per_alpha, beta0=per_beta, epsilon=per_epsilon, per_type=per_type)
    else:
        buffer = ReplayBuffer(capacity)

    stream_into_buffer(
        buffer,
        data,
        shard_size=shard_size,
        shuffle=True,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return buffer


def save_replay_buffer(buffer: ReplayBuffer | PERBuffer, path: str, metadata: Dict | None = None) -> None:
    """Persist the replay buffer to disk along with optional metadata."""
    payload = {
        "buffer": buffer,
        "metadata": metadata or {},
        "buffer_class": buffer.__class__.__name__,
    }
    torch.save(payload, path)


def load_replay_buffer(path: str) -> Tuple[ReplayBuffer | PERBuffer, Dict]:
    """Load a replay buffer and associated metadata from disk."""
    payload = torch.load(path, map_location="cpu")
    return payload["buffer"], payload.get("metadata", {})


#==========================================
# File: rl_agent.py
#==========================================

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
        """
        Purely functional projection (no in-place writes) so autograd stays happy.
        Works both with and without gradients.
        """
        lr_min, lr_max = self.config["curriculum"]["learning_rate_range"]
        lr   = a[..., 0:1].clamp(lr_min, lr_max)
        mix0 = a[..., 1:4].clamp(min=0.0)
        mix  = _project_mixture(mix0, eps=self.mix_floor)
        use  = a[..., 4:5].clamp(0.0, 1.0)
        return torch.cat([lr, mix, use], dim=-1)

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
            # Step actor scheduler only when we actually stepped the actor optimizer
            self.actor_scheduler.step()
        else:
            with torch.no_grad():
                a_curr = self.actor(s)
                a_curr = self._project_action(a_curr)

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


#==========================================
# File: curriculum.py
#==========================================

"""
Module that implements the curriculum training pipeline functions.
This module contains a simple burn-in phase for loss collection and a curriculum training routine 
that uses hyperparameters provided by the RL agent.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import _get_bin_edges, _MAX_LOSS

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

def eval_loader(model, loader, device, num_bins):
    """
    Compute per-sample correct/incorrect loss histograms on GPU.
    Returns (hist_correct, hist_incorrect), each length num_bins.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    hist_c = torch.zeros(num_bins, device=device)
    hist_i = torch.zeros(num_bins, device=device)

    # shared binning
    edges      = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    max_num_batches = int(len(loader) * 0.5)
    with torch.no_grad():
        for batch_num, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            losses  = ce_loss(outputs, labels)
            preds   = outputs.argmax(dim=1)

            # split correct vs incorrect
            lc = losses[preds == labels].clamp(0.0, _MAX_LOSS).flatten()
            li = losses[preds != labels].clamp(0.0, _MAX_LOSS).flatten()

            # bucketize + count
            bc = torch.bucketize(lc, boundaries)
            bi = torch.bucketize(li, boundaries)

            hist_c += torch.bincount(bc, minlength=num_bins).to(device, torch.float32)
            hist_i += torch.bincount(bi, minlength=num_bins).to(device, torch.float32)
            
            if batch_num > max_num_batches:
                break
            

    total = hist_c + hist_i
    S = total.sum()
    if S > 0:
        hist_c /= S
        hist_i /= S

    return hist_c, hist_i

def run_phase_training(model, easy_loader, medium_loader, hard_loader, hyperparams, device):
    """
    Runs a single phase of curriculum training using the provided hyperparameters.
    
    Hyperparameters dictionary (hyperparams) should contain:
        - 'training_samples': int, number of samples to use in this phase.
        - 'learning_rate': float, the learning rate for this phase.
        - 'mixture_ratio': list of 3 floats, the probability-like mixing ratios for [easy, medium, hard].
        - 'phase_batch_size': int, batch size for this phase.
    
    Args:
        model (nn.Module): The model to train.
        easy_loader, medium_loader, hard_loader (DataLoader): Data loaders for the easy, medium, and hard datasets.
        hyperparams (dict): Hyperparameters for this phase.
        device (torch.device): Computation device.
        
    Returns:
        reward (float): The macro accuracy achieved after phase training,
                        computed as the average of accuracies on the easy, medium, and hard datasets.
    """
    phase_batch_size = hyperparams.get("phase_batch_size", 1024)
    criterion = torch.nn.CrossEntropyLoss()

    # Create a mixed training loader based on the provided mixing ratios.
    current_mixture = hyperparams["mixture_ratio"]
    mixed_loader = get_mixed_loader(
        easy_loader.dataset,
        medium_loader.dataset,
        hard_loader.dataset,
        current_mixture,
        num_samples=hyperparams["training_samples"],
        batch_size=phase_batch_size
    )
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model.train()
    phase_samples = 0
    pbar = tqdm(mixed_loader, desc="Phase Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        phase_samples += imgs.size(0)
        pbar.set_postfix(loss=loss.item())
        if phase_samples >= hyperparams["training_samples"]:
            break

    # Evaluate accuracy on each curriculum subset and compute macro accuracy.
    easy_acc = evaluate_accuracy(model, easy_loader, device)
    med_acc = evaluate_accuracy(model, medium_loader, device)
    hard_acc = evaluate_accuracy(model, hard_loader, device)
    macro_acc = (easy_acc + med_acc + hard_acc) / 3.0

    return macro_acc

def run_curriculum_training(model, easy_loader, medium_loader, hard_loader, hyperparams, val_loader, device):
    """
    Runs a simplified curriculum training process using the hyperparameters
    output by the RL agent.
    
    Args:
        model (nn.Module): The model to train.
        easy_loader, medium_loader, hard_loader (DataLoader): Data loaders for each curriculum subset.
        hyperparams (dict): Dictionary with keys:
            - 'training_samples': list of training sample counts (one per phase)
            - 'learning_rates': list of learning rates (one per phase)
            - 'mixture_ratio': list of three lists (each with 3 values) for easy, medium, and hard mixing per phase.
            - 'phase_batch_size': optional batch size (default 512)
        val_loader (DataLoader): Validation loader.
        device (torch.device): Device to perform training.
        
    Returns:
        reward (float): The true macro accuracy achieved on the validation set.
    """
    phase_batch_size = hyperparams.get("phase_batch_size", 512)
    criterion = nn.CrossEntropyLoss()
    
    # For simplicity, run training for a fixed number of phases.
    for phase in range(3):
        # For the mixture ratios, use the set corresponding to the current phase.
        current_mixture = hyperparams["mixture_ratio"][phase]
        mixed_loader = get_mixed_loader(easy_loader.dataset,
                                        medium_loader.dataset,
                                        hard_loader.dataset,
                                        current_mixture,
                                        num_samples=hyperparams["training_samples"][phase],
                                        batch_size=phase_batch_size)
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rates"][phase])
        model.train()
        phase_samples = 0
        pbar = tqdm(mixed_loader, desc=f"Curriculum Phase {phase+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            phase_samples += imgs.size(0)
            pbar.set_postfix(loss=loss.item())
            if phase_samples >= hyperparams["training_samples"][phase]:
                break
            
    # After training, evaluate the model on the validation set.
    acc = evaluate_accuracy(model, val_loader, device)
    return acc

def get_mixed_loader(easy_ds, medium_ds, hard_ds, mixture, num_samples, batch_size=64):
    """
    Create a DataLoader that samples from the concatenation of the easy, medium, and hard datasets
    with weights given by the mixture ratios.
    
    Args:
        easy_ds, medium_ds, hard_ds (Dataset): The three datasets.
        mixture (list): List of three mixture ratios for [easy, medium, hard].
        num_samples (int): Number of samples to draw.
        batch_size (int): Batch size.
        
    Returns:
        DataLoader: The mixed DataLoader.
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
    concat_ds = ConcatDataset([easy_ds, medium_ds, hard_ds])
    weights = [mixture[0]] * len(easy_ds) + [mixture[1]] * len(medium_ds) + [mixture[2]] * len(hard_ds)
    sampler = WeightedRandomSampler(weights, num_samples=max(num_samples, 100), replacement=True)
    loader = DataLoader(concat_ds, batch_size=batch_size, sampler=sampler)
    return loader

def evaluate_accuracy(model, loader, device):
    """
    Evaluate model accuracy on the provided loader.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader.
        device (torch.device): Device to perform evaluation.
        
    Returns:
        accuracy (float): The classification accuracy in percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total




















#==========================================
# File: curriculum_env.py
#==========================================

import torch
import random
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn

from curriculum import eval_loader, run_phase_training
from rl_agent import _project_mixture


def build_cnn_model(n_convs, conv_ch, n_fcs, fc_units, activation_cls, dropout_rate,
                    input_channels=1, input_size=28, num_classes=10):
    """
    Dynamically build a CNN with the given hyperparameters.
    """
    layers = []
    in_ch = input_channels
    out_size = input_size
    for _ in range(n_convs):
        layers.append(nn.Conv2d(in_ch, conv_ch, kernel_size=3, padding=1))
        layers.append(activation_cls())
        layers.append(nn.MaxPool2d(2))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        in_ch = conv_ch
        out_size //= 2
    layers.append(nn.Flatten())
    in_features = in_ch * out_size * out_size
    for _ in range(n_fcs):
        layers.append(nn.Linear(in_features, fc_units))
        layers.append(activation_cls())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_features = fc_units
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def build_mlp_model(hidden_layers, activation_cls, input_dim=28*28, num_classes=10):
    """Dynamically build an MLP with the given hidden layer sizes."""
    layers = [nn.Flatten()]
    in_features = input_dim
    for units in hidden_layers:
        layers.append(nn.Linear(in_features, units))
        layers.append(activation_cls())
        in_features = units
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


class CurriculumEnv:
    """
    Custom environment with cached DataLoaders and models,
    avoiding deep copies by resetting subsets in place.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.batch_size = config["curriculum"]["student_batch_size"]
        dl_cfg = config["curriculum"]
        self.loader_workers = int(dl_cfg.get("dataloader_workers", 4))
        self.loader_pin_memory = bool(dl_cfg.get("dataloader_pin_memory", True))
        self.loader_persistent = bool(dl_cfg.get("dataloader_persistent_workers", self.loader_workers > 0))
        self.num_bins = config["observation"]["num_bins"]

        # Fraction bounds
        fr = config["fractions"]
        self.easy_lower = fr["easy_lower"]
        self.easy_upper = fr["easy_upper"]
        self.medium_lower = fr.get("medium_lower", 0.05)
        self.hard_min = fr.get("hard_min", 0.02)

        # Model search space
        ms = config["model_space"]
        self.n_convs_choices = ms["n_convs_choices"]
        self.conv_channels_choices = ms["conv_channels_choices"]
        self.n_fcs_choices = ms["n_fcs_choices"]
        self.fc_units_choices = ms["fc_units_choices"]
        self.activation_names = ms["activations"]
        self.dropout_rates = ms["dropout_rates"]

        # Transforms
        mean, std = (0.1307,), (0.3081,)
        self.easy_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.medium_transform = T.Compose([
            T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2,0.2,0.2),
            T.ToTensor(), T.Normalize(mean, std)
        ])
        self.hard_transform = T.Compose([
            T.RandomHorizontalFlip(0.5), T.ColorJitter(0.3,0.3,0.3),
            T.RandomRotation(15), T.GaussianBlur(3),
            T.ToTensor(), T.Normalize(mean, std)
        ])

        # Load base datasets
        data_path = config["paths"]["data_path"]
        self.full_easy_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.easy_transform)
        self.full_medium_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.medium_transform)
        self.full_hard_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.hard_transform)

        # Create empty Subsets
        self.easy_subset = Subset(self.full_easy_ds, [])
        self.medium_subset = Subset(self.full_medium_ds, [])
        self.hard_subset = Subset(self.full_hard_ds, [])

        # Hyperparams
        self.train_samples_max = config["curriculum"]["train_samples_max"]
        self.lr_range = config["curriculum"]["learning_rate_range"]
        self.max_phases = config["curriculum"]["max_phases"]

        # Initialize model cache
        self._init_model()

        # Now perform first reset to build loaders and warm-up
        self.reset()

    def _generate_splits(self):
        total = len(self.full_easy_ds)
        idxs = list(range(total))
        random.shuffle(idxs)
        n_easy = int(self.easy_frac * total)
        n_medium = int(self.medium_frac * total)
        return idxs[:n_easy], idxs[n_easy:n_easy+n_medium], idxs[n_easy+n_medium:]

    def _init_model(self):
        model_type = self.config.get("model_type", "cnn")
        if model_type == "mlp":
            if hasattr(self, "model_config"):
                cfg = self.model_config
                act = cfg.get("activation", nn.ReLU)
                self.model = build_mlp_model(cfg["hidden_layers"], act).to(self.device)
            else:
                self.model = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU(), nn.Linear(128,10)).to(self.device)
        else:
            if hasattr(self, "model_config"):
                cfg = self.model_config
                self.model = build_cnn_model(cfg["n_convs"], cfg["conv_ch"], cfg["n_fcs"],
                                             cfg["fc_units"], cfg["activation"], cfg["dropout"]).to(self.device)
            else:
                self.model = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU(), nn.Linear(128,10)).to(self.device)

    def get_observation(self):
        ec, ei = eval_loader(self.model, self.easy_loader,   self.device, self.num_bins)
        mc, mi = eval_loader(self.model, self.medium_loader, self.device, self.num_bins)
        hc, hi = eval_loader(self.model, self.hard_loader,   self.device, self.num_bins)

        counts = [len(self.easy_subset), len(self.medium_subset), len(self.hard_subset)]
        total = sum(counts)
        rel = torch.tensor([c/total for c in counts], device=self.device)

        obs = torch.cat([ec, ei, mc, mi, hc, hi, rel], dim=0)
        phase = torch.tensor(self.current_phase/self.max_phases, device=self.device).unsqueeze(0)
        avail = torch.tensor(self.remaining_samples/self.train_samples_max, device=self.device).unsqueeze(0)
        return torch.cat([obs, phase, avail], dim=0)

    def reset(self, easy_frac: float | None = None, medium_frac: float | None = None):
        """Reset the environment. Optionally specify dataset fractions."""
        if easy_frac is None or medium_frac is None:
            easy = random.uniform(self.easy_lower, self.easy_upper)
            max_med = min(easy, 1.0 - easy - self.hard_min)
            min_med = max(self.medium_lower, (1.0 - easy) / 2)
            self.easy_frac = easy
            self.medium_frac = (
                (min_med + max_med) / 2 if max_med <= min_med else random.uniform(min_med, max_med)
            )
        else:
            self.easy_frac = easy_frac
            self.medium_frac = medium_frac

        # Update subset indices
        e_idx, m_idx, h_idx = self._generate_splits()
        self.easy_subset.indices = e_idx
        self.medium_subset.indices = m_idx
        self.hard_subset.indices = h_idx

        # Build DataLoaders now that subsets are non-empty
        persistent_workers = self.loader_persistent and self.loader_workers > 0
        dl_args = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_workers,
            pin_memory=self.loader_pin_memory,
            persistent_workers=persistent_workers,
        )
        self.easy_loader = DataLoader(self.easy_subset, **dl_args)
        self.medium_loader = DataLoader(self.medium_subset, **dl_args)
        self.hard_loader = DataLoader(self.hard_subset, **dl_args)
        self.warmup_loader = DataLoader(ConcatDataset([self.easy_subset, self.medium_subset, self.hard_subset]), **dl_args)

        model_type = self.config.get("model_type", "cnn")
        if model_type == "mlp":
            depth = random.randint(1, 3)
            widths = [random.randint(32, 128) for _ in range(depth)]
            self.model_config = {
                "hidden_layers": widths,
                "activation": getattr(nn, random.choice(self.activation_names))
            }
        else:
            self.model_config = {
                "n_convs": random.choice(self.n_convs_choices),
                "conv_ch": random.choice(self.conv_channels_choices),
                "n_fcs": random.choice(self.n_fcs_choices),
                "fc_units": random.choice(self.fc_units_choices),
                "activation": getattr(nn, random.choice(self.activation_names)),
                "dropout": random.choice(self.dropout_rates)
            }
        self._init_model()

        # Reset counters
        self.current_phase = 0
        self.remaining_samples = self.train_samples_max

        # Warm-up
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=sum(self.lr_range)/2)
        criterion = nn.CrossEntropyLoss()
        max_batches = max(1, int(0.25 * len(self.warmup_loader)))
        for i, (x,y) in enumerate(self.warmup_loader):
            if i>=max_batches: break
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad(); loss = criterion(self.model(x), y); loss.backward(); opt.step()

        return self.get_observation()

    def step(self, action):
        a = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.float32, device=self.device)
        lr, mix, frac = float(a[0]), a[1:4], float(a[4])
        mix = _project_mixture(
            mix.clamp(min=0.0).unsqueeze(0),
            eps=float(self.config["rl"].get("mix_floor", 0.05))
        ).squeeze(0)
        num = int(max(0.0, min(1.0, frac)) * self.remaining_samples)

        hp = {
            "training_samples": num,
            "learning_rate": lr,
            "mixture_ratio": mix.tolist(),
            "phase_batch_size": self.batch_size,
        }

        macro_acc = run_phase_training(self.model, self.easy_loader, self.medium_loader, self.hard_loader, hp, self.device)

        self.remaining_samples -= num
        self.current_phase += 1
        done = (self.current_phase >= self.max_phases) or (self.remaining_samples <= 0) or (frac <= 0)

        terminal_scale = 1.0  # set to 10.0 if you want a bigger payout
        r = float(macro_acc) * terminal_scale if done else 0.0

        return self.get_observation(), r, done


#==========================================
# File: off_policy_train.py
#==========================================

#==========================================
# File: off_policy_train.py
#==========================================


import os
import yaml
import torch
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import trange, tqdm
import time

import wandb

from rl_agent import DDPGAgent, _project_mixture
from curriculum_env import CurriculumEnv
from replay_buffer import (
    build_replay_buffer_streaming,
    load_replay_buffer,
    save_replay_buffer,
)
from utils import select_top_percent, behavior_clone



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def rollout_ga_candidate(env, candidate_vec, num_phases):
    unit = candidate_vec.numel() // num_phases
    obs = env.reset()
    done = False
    total_r = 0.0
    for p in range(num_phases):
        action = candidate_vec[p * unit:(p + 1) * unit]
        obs, r, done = env.step(action)
        total_r += float(r)
        if done:
            break
    return total_r


# ----- plotting helpers (unchanged) -----
def breakdown_state(state, num_bins):
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32)
    return {
        'easy_correct_hist':   state[0:num_bins].tolist(),
        'easy_incorrect_hist': state[num_bins:2*num_bins].tolist(),
        'medium_correct_hist': state[2*num_bins:3*num_bins].tolist(),
        'medium_incorrect_hist': state[3*num_bins:4*num_bins].tolist(),
        'hard_correct_hist':   state[4*num_bins:5*num_bins].tolist(),
        'hard_incorrect_hist': state[5*num_bins:6*num_bins].tolist(),
        'relative_sizes':      state[6*num_bins:6*num_bins+3].tolist(),
        'extra':               state[6*num_bins+3:6*num_bins+5].tolist(),
    }

def breakdown_action(action):
    if not torch.is_tensor(action):
        action = torch.tensor(action, dtype=torch.float32)
    return {
        'learning_rate': action[0].tolist(),
        'mixing_ratios': action[1:4].tolist(),
        'sample_usage_fraction': action[4].tolist(),
    }

def plot_episode_figure(episode, group_name, num_bins, output_dir):
    min_val, max_val, alpha = 0.0, 13.8, 2.0
    rel = torch.linspace(0, 1, num_bins + 1)
    edges = (min_val + (max_val - min_val) * (rel ** alpha)).tolist()
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    widths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]

    states = episode["states"]; actions = episode["actions"]; rewards = episode["rewards"]
    num_phases = len(states)

    fig = plt.figure(figsize=(20, num_phases * 3 + 3))
    fig.patch.set_facecolor("#fffaf0")
    fig.suptitle("\U0001F389 Fun Episode Analysis \U0001F389", fontsize=18, fontweight="bold", y=0.995)

    gs_top = gridspec.GridSpec(nrows=num_phases, ncols=4, top=0.90, bottom=0.55, wspace=0.4, hspace=0.6)
    gs_bot = gridspec.GridSpec(nrows=1, ncols=4, top=0.50, bottom=0.05, wspace=0.5)

    for i in range(num_phases):
        s = states[i]; r = rewards[i]
        sb = breakdown_state(s, num_bins)

        ax0 = fig.add_subplot(gs_top[i, 0]); ax0.set_facecolor("#f5f5f5")
        ax0.bar(centers, sb["easy_correct_hist"], widths, align="center", color="green", hatch="//", alpha=0.7, label="Correct")
        ax0.bar(centers, sb["easy_incorrect_hist"], widths, align="center", color="red", hatch="xx", alpha=0.7, label="Incorrect")
        if i == 0: ax0.set_title("Easy Loss Hist", fontsize=10, fontweight="bold"); ax0.legend(fontsize=8)
        ax0.set_ylabel(f"P{i+1}\nR:{r:.2f}", fontsize=9); ax0.grid(True, linestyle="--", alpha=0.5)
        ax0.tick_params(axis="both", labelsize=8, rotation=45); ax0.set_xticks(edges)

        ax1 = fig.add_subplot(gs_top[i, 1]); ax1.set_facecolor("#f5f5f5")
        ax1.bar(centers, sb["medium_correct_hist"], widths, color="green", hatch="//", alpha=0.7)
        ax1.bar(centers, sb["medium_incorrect_hist"], widths, color="red", hatch="xx", alpha=0.7)
        if i == 0: ax1.set_title("Medium Loss Hist", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5); ax1.tick_params(axis="both", labelsize=8, rotation=45); ax1.set_xticks(edges)

        ax2 = fig.add_subplot(gs_top[i, 2]); ax2.set_facecolor("#f5f5f5")
        ax2.bar(centers, sb["hard_correct_hist"], widths, color="green", hatch="//", alpha=0.7)
        ax2.bar(centers, sb["hard_incorrect_hist"], widths, color="red", hatch="xx", alpha=0.7)
        if i == 0: ax2.set_title("Hard Loss Hist", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5); ax2.tick_params(axis="both", labelsize=8, rotation=45); ax2.set_xticks(edges)

        ax3 = fig.add_subplot(gs_top[i, 3]); ax3.set_facecolor("#f5f5f5")
        info = sb["relative_sizes"] + sb["extra"]
        ax3.bar(range(5), info, color=["blue", "orange", "purple", "cyan", "magenta"], alpha=0.8)
        if i == 0: ax3.set_title("State Info", fontsize=10, fontweight="bold")
        ax3.set_xticks(range(5)); ax3.set_xticklabels(["Easy", "Med", "Hard", "PhaseRatio", "AvailRatio"], rotation=45, fontsize=8)
        ax3.tick_params(axis="both", labelsize=8); ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.text(2, max(info)*1.05 if info else 0.0, f"R={r:.1f}", ha="center", fontsize=8, color="darkred")

    phases = list(range(1, num_phases + 1))
    lrs, usage, mixrs = [], [], []
    for a in actions:
        if torch.is_tensor(a): a = a.detach().cpu()
        lrs.append(float(a[0])); usage.append(float(a[4])); mixrs.append([float(x) for x in a[1:4]])
    rews = [float(r) for r in rewards]
    # if rews: rews[-1] = rews[-1] / 10.0

    ax_lr = fig.add_subplot(gs_bot[0, 0]); ax_lr.set_facecolor("#f5f5f5")
    ax_lr.plot(phases, lrs, marker="D", linestyle="-", color="blue", markersize=6)
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Phase"); ax_lr.set_ylabel("LR"); ax_lr.set_xticks(phases); ax_lr.grid(True, linestyle=":", alpha=0.6)

    ax_us = fig.add_subplot(gs_bot[0, 1]); ax_us.set_facecolor("#f5f5f5")
    ax_us.plot(phases, usage, marker="D", linestyle="-", color="orange", markersize=6)
    ax_us.set_title("Sample Usage"); ax_us.set_xlabel("Phase"); ax_us.set_ylabel("Usage"); ax_us.set_xticks(phases); ax_us.grid(True, linestyle=":", alpha=0.6)

    ax_mx = fig.add_subplot(gs_bot[0, 2]); ax_mx.set_facecolor("#f5f5f5")
    bar_w = 0.6
    for idx, mr in enumerate(mixrs):
        bottom = 0.0
        ax_mx.bar(idx, mr[0], bottom=bottom, width=bar_w, color="green", label="Easy" if idx == 0 else ""); bottom += mr[0]
        ax_mx.bar(idx, mr[1], bottom=bottom, width=bar_w, color="yellow", label="Med" if idx == 0 else ""); bottom += mr[1]
        ax_mx.bar(idx, mr[2], bottom=bottom, width=bar_w, color="red", label="Hard" if idx == 0 else "")
    ax_mx.set_xticks(range(num_phases)); ax_mx.set_xticklabels([f"P{p}" for p in phases])
    ax_mx.set_title("Mixing Ratios"); ax_mx.set_xlabel("Phase"); ax_mx.set_ylabel("Ratio"); ax_mx.legend(fontsize=8); ax_mx.grid(True, linestyle=":", alpha=0.6)

    ax_rw = fig.add_subplot(gs_bot[0, 3]); ax_rw.set_facecolor("#f5f5f5")
    ax_rw.plot(phases, rews, marker="D", linestyle="-", color="magenta", markersize=6)
    ax_rw.set_title("Reward"); ax_rw.set_xlabel("Phase"); ax_rw.set_ylabel("Reward"); ax_rw.set_xticks(phases); ax_rw.grid(True, linestyle=":", alpha=0.6)

    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.98, hspace=0.6, wspace=0.4)
    fname = os.path.join(output_dir, f"{group_name}_episode_{episode['index']}_detailed.png")
    plt.savefig(fname); plt.close(fig)
    print(f"Saved detailed figure for {group_name} episode {episode['index']} to {fname}")


# ----- Main Training and Evaluation -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="path to your config YAML")
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    results_dir = os.path.join("results", "off_policy_v6")
    os.makedirs(results_dir, exist_ok=True)

    if wandb is not None:
        wandb.init(project="off_policy_training", config=config)
    else:
        print("wandb not installed; proceeding without online logging")

    # Hourly checkpoint overwrites
    hourly_actor   = os.path.join(results_dir, "off_policy_actor_latest.pth")
    hourly_critic1 = os.path.join(results_dir, "off_policy_critic1_latest.pth")
    hourly_critic2 = os.path.join(results_dir, "off_policy_critic2_latest.pth")
    last_hour_save = time.time()

    # Load dataset to CPU
    dataset_path = config["paths"]["pretrain_path"]
    data = torch.load(dataset_path, map_location="cpu")
    states, actions = data["states"], data["actions"]
    rewards, next_states, dones = data["rewards"], data["next_states"], data["dones"]
    
    # --- Reward normalization: dataset has final phase reward ×10; scale terminals down ---
    if isinstance(dones, torch.Tensor) and dones.dtype == torch.bool:
        rewards = rewards.clone()
        # zero out all non-terminal rewards
        rewards[~dones] = 0.0
        # scale terminal rewards if you want (here ×10)
        rewards[dones] = rewards[dones] / 10.0
    else:
        # if dones is float 0/1
        dones_f = dones.to(torch.float32)
        rewards = rewards.clone()
        rewards = torch.where(dones_f > 0.5, rewards / 10.0, torch.zeros_like(rewards))

    # Project offline dataset actions onto the updated manifold so critics see reachable actions.
    with torch.no_grad():
        projected_actions = actions.clone()
        lr_min, lr_max = config["curriculum"]["learning_rate_range"]
        projected_actions[:, 0:1] = projected_actions[:, 0:1].clamp(lr_min, lr_max)
        projected_actions[:, 1:4] = _project_mixture(
            projected_actions[:, 1:4].clamp(min=0.0),
            eps=float(config["rl"].get("mix_floor", 0.05)),
        )
        projected_actions[:, 4:5] = projected_actions[:, 4:5].clamp(0.0, 1.0)
        actions = projected_actions
    
    dataset = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}
    print(f"Loaded dataset from {dataset_path} with {len(states)} transitions")

    # Environment & agent on user-specified device
    env = CurriculumEnv(config)
    obs_dim = len(env.reset()); action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, config)

    # --- State normalization (mean/std over the offline set) ---
    with torch.no_grad():
        s_mean = states.mean(dim=0)
        s_std  = states.std(dim=0, unbiased=False).clamp_min(1e-6)
        states = (states - s_mean) / s_std
        next_states = (next_states - s_mean) / s_std
        
    def _norm_obs(x):
        return (x - s_mean.to(x.device)) / s_std.to(x.device)

    dataset = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}

    # --- Behavior Cloning ---
    try:
        expert = select_top_percent(dataset, percent=int(config["rl"].get("bc_top_percent", 20)))
        behavior_clone(agent.actor, expert, epochs=int(config["rl"].get("bc_epochs", 5)),
                        batch_size=int(config["rl"].get("bc_batch_size", 256)), device=agent.device)
        agent._hard_update(agent.actor_target, agent.actor)
        print("Behavior cloning warm-start complete.")
    except Exception as e:
        print(f"BC warm-start skipped due to error: {e}")

    # --------- Build a CPU-resident PER buffer via streaming ingestion ----------
    shard_size = int(config["rl"].get("ingest_shard_size", 200_000))
    cache_path = config.get("paths", {}).get("replay_cache_path")
    use_cache = bool(cache_path)
    replay_buffer = None

    # Normalize cache path to repo-relative path if provided
    if use_cache:
        cache_path = os.path.expanduser(cache_path)
        cache_path = os.path.abspath(cache_path)

    per_enabled = bool(config["rl"].get("per_enabled", True))
    per_alpha = float(config["rl"].get("per_alpha", 0.4))
    per_beta = float(config["rl"].get("per_beta", 0.6))
    per_epsilon = float(config["rl"].get("per_epsilon", 1e-6))
    per_type = str(config["rl"].get("per_type", "proportional"))
    buffer_capacity = int(config["rl"]["buffer_size"])

    def _cache_matches(meta: dict, expected: dict) -> bool:
        for key, value in expected.items():
            if meta.get(key) != value:
                return False
        return True

    dataset_mtime = None
    try:
        dataset_mtime = os.path.getmtime(dataset_path)
    except (OSError, FileNotFoundError):
        dataset_mtime = None

    expected_meta = {
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_mtime": dataset_mtime,
        "dataset_length": int(len(states)),
        "buffer_capacity": buffer_capacity,
        "per_enabled": per_enabled,
        "per_alpha": per_alpha,
        "per_beta": per_beta,
        "per_epsilon": per_epsilon,
        "per_type": per_type,
        "shard_size": shard_size,
    }

    if use_cache and os.path.exists(cache_path):
        try:
            cached_buffer, cached_meta = load_replay_buffer(cache_path)
            if _cache_matches(cached_meta, expected_meta):
                replay_buffer = cached_buffer
                print(f"Loaded replay buffer cache from {cache_path}")
            else:
                print("Replay buffer cache metadata mismatch; rebuilding cache.")
        except Exception as cache_err:
            print(f"Failed to load replay buffer cache ({cache_err}); rebuilding.")

    if replay_buffer is None:
        replay_buffer = build_replay_buffer_streaming(
            dataset,
            capacity=buffer_capacity,          # you can lower this in YAML if RAM is tight
            use_per=per_enabled,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            per_type=per_type,
            shard_size=shard_size,
            show_progress=True,
            progress_desc="Initializing replay buffer",
        )
        if use_cache:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            try:
                save_replay_buffer(replay_buffer, cache_path, expected_meta)
                print(f"Saved replay buffer cache to {cache_path}")
            except Exception as cache_save_err:
                print(f"Warning: failed to save replay buffer cache ({cache_save_err})")

    if per_enabled and hasattr(replay_buffer, "set_beta"):
        replay_buffer.set_beta(per_beta)

    print(f"Replay buffer initialized with {len(replay_buffer)} transitions (capacity {config['rl']['buffer_size']}).")

    # Probe states for action variance tracking (on agent device)
    probe_batch_size = 256
    perm = torch.randperm(len(states))
    idxs = perm[:probe_batch_size]
    variance_states_tensor = states[idxs].to(agent.device).float()

    # Training schedule
    num_updates = int(config["rl"].get("off_policy_updates", 1_000_000))
    evaluation_interval  = max(1, num_updates // 200)   
    checkpoint_interval  = max(1, num_updates // 200)   # save every 5%

    print(f"Starting off-policy training for {num_updates} updates")

    # Logs
    actor_losses, critic1_losses, critic2_losses = [], [], []
    actor_lrs, critic1_lrs, critic2_lrs = [], [], []
    reward_progress, reward_stds, eval_updates = [], [], []
    action_var_history, update_steps = [], []
    num_bins = int(config["observation"]["num_bins"])

    best_mean_reward = -float("inf")
    no_improve = 0
    early_stop = config["rl"].get("early_stop_patience")
    last_actor_loss = 0.0
    num_eval_eps = int(config["rl"].get("num_eval_episodes", 1))
    stop_training = False

    for update in trange(num_updates, desc="Off-policy Training"):
        if len(replay_buffer) >= int(config["rl"]["batch_size"]):
            metrics = agent.update(replay_buffer, int(config["rl"]["batch_size"]))
            if metrics is not None:
                if metrics["actor_loss"] is not None:
                    last_actor_loss = metrics["actor_loss"]
                actor_losses.append(last_actor_loss)
                critic1_losses.append(metrics["critic1_loss"])
                critic2_losses.append(metrics["critic2_loss"])
                actor_lrs.append(agent.actor_optimizer.param_groups[0]["lr"])
                critic1_lrs.append(agent.critic1_optimizer.param_groups[0]["lr"])
                critic2_lrs.append(agent.critic2_optimizer.param_groups[0]["lr"])
                if wandb is not None:
                    wandb.log({
                        "actor_loss": last_actor_loss,
                        "critic1_loss": metrics["critic1_loss"],
                        "critic2_loss": metrics["critic2_loss"],
                        "actor_lr": actor_lrs[-1],
                        "critic1_lr": critic1_lrs[-1],
                        "critic2_lr": critic2_lrs[-1],
                        "ood_l2": metrics.get("ood_l2", None),
                        "update": update,
                    })
                else:
                    if metrics.get("ood_l2") is not None:
                        print(f"Update {update}: ood_l2={metrics['ood_l2']:.4f}")

        with torch.no_grad():
            pred_actions = agent.actor(variance_states_tensor).cpu()
        var = torch.var(pred_actions, dim=0, unbiased=False)
        action_var_history.append(var.tolist()); update_steps.append(update)

        # Hourly rolling checkpoint
        if time.time() - last_hour_save >= 3600:
            torch.save(agent.actor.state_dict(), hourly_actor)
            torch.save(agent.critic1.state_dict(), hourly_critic1)
            torch.save(agent.critic2.state_dict(), hourly_critic2)
            print(f"Hourly checkpoint saved at update {update}")
            last_hour_save = time.time()

        # Periodic evaluation + plots
        if update % evaluation_interval == 0:
            rewards_this_ckpt = []
            first_episode = None
            for ep_i in range(num_eval_eps):
                eval_states, eval_actions, eval_rewards = [], [], []
                obs_eval = env.reset(); done = False
                obs_eval = _norm_obs(obs_eval)
                agent.ou_noise.reset() 
                with tqdm(total=env.max_phases, desc=f"Eval {update} Ep{ep_i}", leave=False) as pbar:
                    while not done:
                        action_eval = agent.select_action(obs_eval, noise_enable=False)
                        eval_states.append(obs_eval); eval_actions.append(action_eval)
                        obs_eval, reward, done = env.step(action_eval)
                        obs_eval = _norm_obs(obs_eval)
                        eval_rewards.append(reward); pbar.update(1)
                total_reward = float(sum(eval_rewards))
                rewards_this_ckpt.append(total_reward)
                if ep_i == 0:
                    first_episode = {
                        "index": update,
                        "states": eval_states,
                        "actions": eval_actions,
                        "rewards": eval_rewards,
                        "total_reward": total_reward,
                        "episode_length": len(eval_states),
                    }

            mean_reward = float(np.mean(rewards_this_ckpt))
            std_reward  = float(np.std(rewards_this_ckpt))
            reward_progress.append(mean_reward); reward_stds.append(std_reward); eval_updates.append(update)
            print(f"Checkpoint at update {update}/{num_updates} - Eval reward {mean_reward:.2f} ± {std_reward:.2f}")
            with torch.no_grad():
                a_eval = agent.actor(variance_states_tensor).cpu()
                mix = a_eval[:, 1:4].clamp(min=1e-8)
                mix_H = (-(mix * mix.log()).sum(dim=1)).mean().item()
                mix_min = mix.min().item()
                mix_max = mix.max().item()
                usage_mean = a_eval[:, 4].mean().item()
            if wandb is not None:
                wandb.log({"eval_mean_reward": mean_reward, "eval_std_reward": std_reward, "update": update})
                wandb.log({"mix_entropy": mix_H, "mix_min": mix_min, "mix_max": mix_max, "usage_mean": usage_mean})

            # Optional: compare vs GA elites if provided
            ga_paths = config.get("compare_models", {}).get("GA_Elites", None)
            if ga_paths:
                try:
                    ga = torch.load(ga_paths, map_location="cpu")
                    elites = ga.get("population")
                    num_phases = int(config["curriculum"].get("max_phases", 3))
                    scores = []
                    for i in range(min(3, elites.size(0))):
                        env_comp = CurriculumEnv(config)
                        scores.append(rollout_ga_candidate(env_comp, elites[i], num_phases))
                    ga_mean = float(np.mean(scores))
                    if wandb is not None:
                        wandb.log({"ga_elite_mean_reward": ga_mean, "update": update})
                    print(f"GA elite mean reward (sample): {ga_mean:.2f}")
                except Exception as e:
                    print(f"GA compare skipped: {e}")

            eval_episode = first_episode

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(agent.actor.state_dict(),  os.path.join(results_dir, "best_actor.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(results_dir, "best_critic1.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(results_dir, "best_critic2.pth"))
                no_improve = 0
            else:
                no_improve += 1

            if early_stop is not None and no_improve >= int(early_stop):
                print(f"Early stopping at update {update} due to no improvement")
                stop_training = True
                break

            # Plots
            plot_episode_figure(eval_episode, f"eval_{update}", num_bins, results_dir)

            fig_loss, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(8, 10))
            ax_actor.plot(actor_losses, color='blue'); ax_actor.set_title("Actor Loss Progression")
            ax_actor.set_xlabel("Update Steps"); ax_actor.set_ylabel("Loss")
            ax_critic.plot(critic1_losses, label="Critic1")
            ax_critic.plot(critic2_losses, label="Critic2"); ax_critic.legend()
            ax_critic.set_title("Critic Loss Progression"); ax_critic.set_xlabel("Update Steps"); ax_critic.set_ylabel("Loss")
            fig_loss.tight_layout(); fig_loss.savefig(os.path.join(results_dir, f"training_losses_{update}.png")); plt.close(fig_loss)

            plt.figure(); plt.errorbar(eval_updates, reward_progress, yerr=reward_stds, marker='o', capsize=3)
            plt.xlabel("Update Steps"); plt.ylabel("Reward"); plt.title("Periodic Reward Evaluation")
            plt.savefig(os.path.join(results_dir, f"reward_progress_{update}.png")); plt.close()

            fig_lr = plt.figure()
            plt.plot(actor_lrs, label='Actor'); plt.plot(critic1_lrs, label='Critic1'); plt.plot(critic2_lrs, label='Critic2')
            plt.xlabel('Update Steps'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Progression'); plt.legend()
            fig_lr.tight_layout(); fig_lr.savefig(os.path.join(results_dir, f"learning_rates_{update}.png")); plt.close(fig_lr)

            if update % checkpoint_interval == 0:
                torch.save(agent.actor.state_dict(),  os.path.join(results_dir, f"off_policy_actor_{update}.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(results_dir, f"off_policy_critic1_{update}.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(results_dir, f"off_policy_critic2_{update}.pth"))

            action_var_array = torch.tensor(action_var_history)
            names = ["learning_rate","mix_easy","mix_med","mix_hard","sample_usage"]
            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8,12), sharex=True)
            for idx, name in enumerate(names):
                axs[idx].plot(update_steps, action_var_array[:, idx].numpy())
                axs[idx].set_ylabel("Var"); axs[idx].set_title(name); axs[idx].grid(True)
            axs[-1].set_xlabel("Update Step")
            plt.tight_layout(); plt.savefig(os.path.join(results_dir, "action_variance.png")); plt.close()

        if stop_training:
            break

    # Final full evaluation
    eval_states, eval_actions, eval_rewards = [], [], []
    obs_eval = env.reset(); done = False
    obs_eval = _norm_obs(obs_eval)
    agent.ou_noise.reset() 
    with tqdm(total=env.max_phases, desc="Final Eval", leave=False) as pbar:
        while not done:
            action_eval = agent.select_action(obs_eval, noise_enable=False)
            eval_states.append(obs_eval); eval_actions.append(action_eval)
            obs_eval, reward, done = env.step(action_eval)
            obs_eval = _norm_obs(obs_eval)
            eval_rewards.append(reward); pbar.update(1)
    total_reward = float(sum(eval_rewards))
    print(f"Final evaluation episode total reward: {total_reward}")

    final_index = update if stop_training else num_updates
    eval_episode = {
        "index": final_index, "states": eval_states, "actions": eval_actions, "rewards": eval_rewards,
        "total_reward": total_reward, "episode_length": len(eval_states)
    }
    plot_episode_figure(eval_episode, "final_eval", num_bins, results_dir)

    # Save finals
    final_actor   = os.path.join(results_dir, "off_policy_actor_model_final.pth")
    final_critic1 = os.path.join(results_dir, "off_policy_critic1_model_final.pth")
    final_critic2 = os.path.join(results_dir, "off_policy_critic2_model_final.pth")
    torch.save(agent.actor.state_dict(), final_actor)
    torch.save(agent.critic1.state_dict(), final_critic1)
    torch.save(agent.critic2.state_dict(), final_critic2)
    print(f"Saved final actor → {final_actor}")
    print(f"Saved final critic 1 → {final_critic1}")
    print(f"Saved final critic 2 → {final_critic2}")

    # Final plots
    fig_loss, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(8, 10))
    ax_actor.plot(actor_losses, color='blue'); ax_actor.set_title("Actor Loss Progression")
    ax_actor.set_xlabel("Update Steps"); ax_actor.set_ylabel("Loss")
    ax_critic.plot(critic1_losses, color='red', label="Critic 1")
    ax_critic.plot(critic2_losses, color='green', label="Critic 2"); ax_critic.legend()
    ax_critic.set_title("Critic Loss Progression"); ax_critic.set_xlabel("Update Steps"); ax_critic.set_ylabel("Loss")
    fig_loss.tight_layout(); fig_loss.savefig(os.path.join(results_dir, "training_losses_final.png")); plt.close(fig_loss)

    plt.figure(); plt.errorbar(eval_updates, reward_progress, yerr=reward_stds, marker='o', capsize=3)
    plt.xlabel("Update Steps"); plt.ylabel("Reward"); plt.title("Periodic Reward Evaluation")
    plt.savefig(os.path.join(results_dir, "reward_progress_final.png")); plt.close()

    plt.figure()
    plt.plot(actor_lrs, label='Actor'); plt.plot(critic1_lrs, label='Critic1'); plt.plot(critic2_lrs, label='Critic2')
    plt.xlabel('Update Steps'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Progression'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(results_dir, 'learning_rates_final.png')); plt.close()

    if wandb is not None:
        wandb.log({"final_total_reward": total_reward})
        wandb.finish()


if __name__ == "__main__":
    main()


#==========================================
# File: config_off_policy_seq.yaml
#==========================================

# Compute device for training. Change to "cpu" if no GPU is available.
device: "cuda:4"

fractions:
  easy_lower: 0.34       # easy fraction ∈ [0.34, 0.95]
  easy_upper: 0.95
  medium_lower: 0.03    # medium ≥ 0
  hard_min: 0.02         # ensure hard ≥ 0.02

# Define the model search space for dynamic architectures
model_space:
  n_convs_choices: [1]
  conv_channels_choices: [8, 16]
  n_fcs_choices: [1, 2]
  fc_units_choices: [128, 256]
  activations: ["ReLU"]
  dropout_rates: [0.0, 0.2, 0.5]

curriculum:
  train_samples_max: 30000
  learning_rate_range: [0.001, 0.01]
  max_phases: 3
  student_batch_size: 512

rl:
  ea_pop_size:      124
  ea_generations:   124
  ea_top_k:         3
  ea_mutation_rate: 0.1

  pretrain_bc_iters:       100000
  pretrain_critic_iters:   100000
  off_policy_updates:      100000000

  policy_delay: 4
  policy_noise: 0.2
  noise_clip:   0.1
  mix_temp: 2.0
  mix_floor: 0.05

  on_policy_episodes:            10000
  on_policy_updates_per_episode: 1
  batch_size:                    512
  probe_from_ea:                 true
  probe_batch_size:              128
  num_parallel_students:         1000

  actor_lr:  0.0003
  critic_lr: 0.001
  lr_scheduler: "cosine"
  lr_decay_steps: 200000
  lr_decay_rate: 0.5
  lr_min_lr: 1.0e-5

  gamma: 0.995
  tau:   0.001

  exploration_noise: 0.2
  exploration_noise_decay_steps: 300000

  per_enabled: true
  per_type:    "proportional"
  per_alpha:   0.4
  per_beta:    0.6
  per_epsilon: 1e-6

  elite_fraction: 0.02
  num_eval_episodes: 5
  early_stop_patience: 10

  use_behavioral_cloning:    true
  bc_trajectory_selection:   "top_k"
  bc_top_k:                  250
  bc_top_percent:            0.5 # Top 0.5% 
  bc_epochs: 5
  bc_batch_size: 256
  bc_reward_threshold:       0.5
  bc_percentile:             0.75
  use_triplet_loss:          true
  triplet_margin:            0.2
  pretrain_critic_offpolicy: true

  seed_replay_buffer: true
  buffer_size:         14136870

  entropy_beta: 2.0e-2    # try 1e-3 to 3e-3
  usage_entropy_beta: 1.0e-2      # encourage usage away from 0/1
  lr_center_penalty:  1.0e-3     # discourage LR hugging the bounds
  usage_penalty:      5.0e-3      
  target_usage:       0.5
  td3bc_lambda: 7.5        # 0.0 disables TD3+BC; typical 1..10
  td3bc_qfilter: false     # only apply BC where Q(data) >= Q(policy)
  cql_alpha:          0.005         # try {0.001, 0.003, 0.01}
  cql_num_samples: 10      # you already have this; keep 10-20
  mmd_lambda: 0.0
  mmd_sigma: 0.2

observation:
  num_bins: 16

paths:
  save_path:                  "results/curriculum_rl"
  data_path:                  "./data"
  pretrain_path:              "/data/naddeok/spcl/merged_history.pt"
  off_policy_actor_model:     "results/off_policy/off_policy_actor_model_final.pth"
  off_policy_critic1_model:   "results/off_policy/off_policy_critic1_model_final.pth"
  off_policy_critic2_model:   "results/off_policy/off_policy_critic2_model_final.pth"

  on_policy_dir:       "results/seq_on_policy"
  on_policy_actor_model:    "results/seq_on_policy/seq_on_policy_actor_model_final.pth"
  on_policy_critic1_model:  "results/seq_on_policy/seq_on_policy_critic1_model_final.pth"
  on_policy_critic2_model:  "results/seq_on_policy/seq_on_policy_critic2_model_final.pth"
  on_policy_parallel_dir: "results/on_policy_parallel"
  replay_cache_path: null   # set to a filepath (e.g. "./cache/replay_buffer.pt") to enable replay caching

compare_models:
  Off Policy Start:
    actor:   "results/off_policy/off_policy_actor_0.pth"
    critic1: "results/off_policy/off_policy_critic1_0.pth"
    critic2: "results/off_policy/off_policy_critic2_0.pth"
  Off Policy End:
    actor:   "results/off_policy/off_policy_actor_model_final.pth"
    critic1: "results/off_policy/off_policy_critic1_model_final.pth"
    critic2: "results/off_policy/off_policy_critic2_model_final.pth"

seed: 42





#==========================================
# File: analyze_history.py
#==========================================

#!/usr/bin/env python3
"""
analyze_history.py

This script loads a PyTorch .pt file (saved by generate_dataset.py) that contains training history 
(transitions from an evolutionary dataset), groups transitions into episodes using the 
done flag as an episode terminator, and then analyzes the episodes. The script:
  - Computes the total reward per episode.
  - Selects 5 episodes with the lowest total rewards, 5 episodes around the median, 
    and 5 episodes with the highest total rewards.
  - For each selected episode, it produces a text summary and also generates a detailed figure:
      * TOP BLOCK: For each phase (transition) in the episode, a row of 4 subplots:
            - Column 0: Combined "Easy" loss histogram (green for correct, red for incorrect).
            - Column 1: Combined "Medium" loss histogram.
            - Column 2: Combined "Hard" loss histogram.
            - Column 3: State Info bar chart (relative sizes and extra features).
         Each row is annotated with that phase’s reward.
      * BOTTOM BLOCK: Aggregated evolution across phases with 4 subplots:
            1. Learning Rate evolution.
            2. Sample Usage evolution.
            3. Mixing Ratios as a stacked bar plot (one bar per phase).
            4. Reward evolution (with the last phase reward divided by 10).
  - Also plots and saves a histogram of all episode total rewards.

Usage:
    python analyze_history.py \
        --pt_file evolutionary_dataset.pt \
        [--num_bins 64] \
        [--output_dir history]

If no arguments are given, it defaults to:
    pt_file: results/curriculum_rl/evolutionary_dataset.pt
    num_bins: 16
    output_dir: history
"""

import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_data(pt_file: str):
    """Load the saved transitions from a .pt file."""
    data = torch.load(pt_file, map_location='cpu')
    return data['states'], data['actions'], data['rewards'], data['dones']

def group_episodes_by_done(states, actions, rewards, dones):
    """
    Groups transitions into episodes by accumulating consecutive transitions
    until a transition with done == True is encountered.
    Each episode dictionary now includes an 'index' field.
    """
    episodes = []
    current_states = []
    current_actions = []
    current_rewards = []
    episode_counter = 0
    for s, a, r, d in zip(states, actions, rewards, dones):
        current_states.append(s)
        current_actions.append(a)
        current_rewards.append(r)
        if d:
            aggregated_reward = sum(float(x) for x in current_rewards)
            episodes.append({
                'index': episode_counter,
                'states': current_states.copy(),
                'actions': current_actions.copy(),
                'rewards': current_rewards.copy(),
                'total_reward': aggregated_reward,
                'episode_length': len(current_states)
            })
            episode_counter += 1
            current_states = []
            current_actions = []
            current_rewards = []
            
            # if episode_counter >= 100:
            #     break
    # Group any remaining transitions as an incomplete episode.
    if current_states:
        aggregated_reward = sum(float(x) for x in current_rewards)
        episodes.append({
            'index': episode_counter,
            'states': current_states.copy(),
            'actions': current_actions.copy(),
            'rewards': current_rewards.copy(),
            'total_reward': aggregated_reward,
            'episode_length': len(current_states)
        })
    return episodes

def select_episode_groups(episodes):
    """
    Sorts episodes by total reward and selects:
       - 5 episodes with the lowest total rewards,
       - 5 episodes around the median,
       - 5 episodes with the highest total rewards.
    """
    sorted_eps = sorted(episodes, key=lambda ep: ep['total_reward'])
    num_eps = len(sorted_eps)
    low = sorted_eps[:5] if num_eps >= 5 else sorted_eps
    high = sorted_eps[-5:] if num_eps >= 5 else sorted_eps
    median_index = num_eps // 2
    start = max(0, median_index - 2)
    end = start + 5
    if end > num_eps:
        end = num_eps
        start = max(0, end - 5)
    median = sorted_eps[start:end]
    return low, median, high

def breakdown_state(state, num_bins):
    """
    Breaks down the state vector into its constituent parts.
    Assumes the following layout:
      - Indices 0:num_bins                  : Easy Correct Histogram
      - Indices num_bins:2*num_bins         : Easy Incorrect Histogram
      - Indices 2*num_bins:3*num_bins       : Medium Correct Histogram
      - Indices 3*num_bins:4*num_bins       : Medium Incorrect Histogram
      - Indices 4*num_bins:5*num_bins       : Hard Correct Histogram
      - Indices 5*num_bins:6*num_bins       : Hard Incorrect Histogram
      - Indices 6*num_bins:6*num_bins+3     : Relative dataset sizes (3 values)
      - Indices 6*num_bins+3:6*num_bins+5   : Extra state features (2 values)
    Total length = 6*num_bins + 5.
    """
    breakdown = {}
    breakdown['easy_correct_hist']    = state[0:num_bins]
    breakdown['easy_incorrect_hist']  = state[num_bins:2*num_bins]
    breakdown['medium_correct_hist']  = state[2*num_bins:3*num_bins]
    breakdown['medium_incorrect_hist'] = state[3*num_bins:4*num_bins]
    breakdown['hard_correct_hist']     = state[4*num_bins:5*num_bins]
    breakdown['hard_incorrect_hist']   = state[5*num_bins:6*num_bins]
    breakdown['relative_sizes']       = state[6*num_bins:6*num_bins+3]
    breakdown['extra']                = state[6*num_bins+3:6*num_bins+5]
    return breakdown

def breakdown_action(action):
    """
    Breaks down the 5-dimensional action vector into:
      - Learning rate: action[0]
      - Mixing ratios for (Easy, Medium, Hard): action[1:4]
      - Sample usage fraction: action[4]
    """
    return {
        'learning_rate': action[0],
        'mixing_ratios': action[1:4],
        'sample_usage_fraction': action[4],
    }

def save_episode_details(episodes, group_name, num_bins, output_dir):
    """
    Saves a text file summarizing each episode’s detailed breakdown for a specified group.
    """
    filename = os.path.join(output_dir, f"{group_name}_episodes.txt")
    with open(filename, "w") as f:
        f.write(f"--- {group_name.capitalize()} Episodes Analysis ---\n\n")
        for ep in episodes:
            f.write(f"Episode {ep['index']} - Total Reward: {ep['total_reward']}, Length: {ep['episode_length']}\n")
            for i, (state, action, reward) in enumerate(zip(ep['states'], ep['actions'], ep['rewards'])):
                f.write(f"  Phase {i+1} (Reward: {reward}):\n")
                sb = breakdown_state(state, num_bins)
                ab = breakdown_action(action)
                f.write("    State breakdown:\n")
                for key, value in sb.items():
                    f.write(f"      {key}: {value}\n")
                f.write("    Action breakdown:\n")
                f.write(f"      Learning Rate: {ab['learning_rate']:.4f}\n")
                f.write(f"      Mixing Ratios: {ab['mixing_ratios']}\n")
                f.write(f"      Sample Usage Fraction: {ab['sample_usage_fraction']:.4f}\n")
            f.write("\n")
    print(f"Saved {group_name} episodes analysis to {filename}")

def plot_reward_distribution(episodes, output_dir):
    """Plots and saves a histogram of total rewards for all episodes."""
    rewards = [ep['total_reward'] for ep in episodes]
    plt.figure()
    plt.hist(rewards, bins=20, edgecolor='black', color='skyblue')
    plt.title("Episode Total Reward Distribution")
    plt.xlabel("Total Reward")
    plt.ylabel("Number of Episodes")
    plot_path = os.path.join(output_dir, "episode_reward_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved reward distribution plot to {plot_path}")

def plot_episode_figure(episode, group_name, num_bins, output_dir):
    """
    For a given episode, creates a detailed figure with playful styling:
      - Adds a fun emoji title and light background.
      - TOP BLOCK: one row per phase:
          * Columns 0–2: loss histograms with original colors (green/red), plus hatch patterns.
          * Column 3: state info bar chart with original 'colors_info'.
          * Grids, light axis background, and phase annotations keep it lively.
      - BOTTOM BLOCK: evolution plots with:
          1. LR (blue, diamond markers) +
             rocket 🚀 annotation at the max point.
          2. Sample usage (orange, diamond markers).
          3. Mixing ratios (stacked bars in original green/yellow/red).
          4. Reward (magenta, diamond markers) +
             star ★ annotation at the final point.
    """
    import torch
    # Recompute bin edges
    min_val, max_val, alpha = 0.0, 13.8, 2.0
    rel = torch.linspace(0, 1, num_bins + 1)
    edges = (min_val + (max_val - min_val) * (rel ** alpha)).tolist()
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    widths  = [edges[i+1] - edges[i]       for i in range(len(edges)-1)]

    num_phases = len(episode['states'])
    fig = plt.figure(figsize=(20, num_phases * 3 + 3))
    fig.patch.set_facecolor('#fffaf0')
    fig.suptitle("🎉 Fun Episode Analysis 🎉", fontsize=18, fontweight='bold', y=0.995)

    gs_top = gridspec.GridSpec(
        nrows=num_phases, ncols=4,
        top=0.90, bottom=0.55, wspace=0.4, hspace=0.6,
        height_ratios=[1] * num_phases
    )
    gs_bot = gridspec.GridSpec(
        nrows=1, ncols=4,
        top=0.50, bottom=0.05, wspace=0.5
    )

    # --- TOP BLOCK ---
    for i in range(num_phases):
        s  = episode['states'][i]
        r  = episode['rewards'][i]
        sb = breakdown_state(s, num_bins)

        # Easy losses
        ax0 = fig.add_subplot(gs_top[i, 0])
        ax0.set_facecolor('#f5f5f5')
        ax0.bar(centers, sb['easy_correct_hist'],   widths,
                align='center', color='green', hatch='//', alpha=0.7, label='Correct')
        ax0.bar(centers, sb['easy_incorrect_hist'], widths,
                align='center', color='red',   hatch='xx', alpha=0.7, label='Incorrect')
        if i == 0:
            ax0.set_title("Easy Loss Hist", fontsize=10, fontweight='bold')
            ax0.legend(fontsize=8)
        ax0.set_ylabel(f"P{i+1}\nR:{r:.2f}", fontsize=9)
        ax0.grid(True, linestyle='--', alpha=0.5)
        ax0.tick_params(axis='both', labelsize=8, rotation=45)
        ax0.set_xticks(edges)

        # Medium losses
        ax1 = fig.add_subplot(gs_top[i, 1])
        ax1.set_facecolor('#f5f5f5')
        ax1.bar(centers, sb['medium_correct_hist'],   widths, color='green', hatch='//', alpha=0.7)
        ax1.bar(centers, sb['medium_incorrect_hist'], widths, color='red',   hatch='xx', alpha=0.7)
        if i == 0: ax1.set_title("Medium Loss Hist", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.tick_params(axis='both', labelsize=8, rotation=45)
        ax1.set_xticks(edges)

        # Hard losses
        ax2 = fig.add_subplot(gs_top[i, 2])
        ax2.set_facecolor('#f5f5f5')
        ax2.bar(centers, sb['hard_correct_hist'],   widths, color='green', hatch='//', alpha=0.7)
        ax2.bar(centers, sb['hard_incorrect_hist'], widths, color='red',   hatch='xx', alpha=0.7)
        if i == 0: ax2.set_title("Hard Loss Hist", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.tick_params(axis='both', labelsize=8, rotation=45)
        ax2.set_xticks(edges)

        # State Info
        ax3 = fig.add_subplot(gs_top[i, 3])
        ax3.set_facecolor('#f5f5f5')
        info = sb['relative_sizes'].tolist() + sb['extra'].tolist()
        colors_info = ['blue','orange','purple','cyan','magenta']
        ax3.bar(range(5), info, color=colors_info, alpha=0.8)
        if i == 0: ax3.set_title("State Info", fontsize=10, fontweight='bold')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Easy','Med','Hard','PhaseRatio','AvailRatio'], rotation=45, fontsize=8)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.text(2, max(info)*1.05, f"R={r:.1f}", ha='center', fontsize=8, color='darkred')

    # --- BOTTOM BLOCK ---
    phases = list(range(1, num_phases + 1))
    lrs    = [episode['actions'][i][0] for i in range(num_phases)]
    usage  = [episode['actions'][i][4] for i in range(num_phases)]
    mixrs  = [episode['actions'][i][1:4] for i in range(num_phases)]
    rews   = [episode['rewards'][i]        for i in range(num_phases)]
    # if rews: rews[-1] /= 10.0

    # Learning rate
    ax_lr = fig.add_subplot(gs_bot[0, 0])
    ax_lr.set_facecolor('#f5f5f5')
    ax_lr.plot(phases, lrs, marker='D', linestyle='-', color='blue', markersize=6)
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Phase"); ax_lr.set_ylabel("LR")
    ax_lr.set_xticks(phases); ax_lr.grid(True, linestyle=':', alpha=0.6)

    # Sample usage
    ax_us = fig.add_subplot(gs_bot[0, 1])
    ax_us.set_facecolor('#f5f5f5')
    ax_us.plot(phases, usage, marker='D', linestyle='-', color='orange', markersize=6)
    ax_us.set_title("Sample Usage"); ax_us.set_xlabel("Phase"); ax_us.set_ylabel("Usage")
    ax_us.set_xticks(phases); ax_us.grid(True, linestyle=':', alpha=0.6)

    # Mixing ratios
    ax_mx = fig.add_subplot(gs_bot[0, 2])
    ax_mx.set_facecolor('#f5f5f5')
    bar_w = 0.6
    for idx, mr in enumerate(mixrs):
        bottom = 0.0
        ax_mx.bar(idx, mr[0], bottom=bottom, width=bar_w, color='green',  label='Easy'  if idx==0 else "")
        bottom += mr[0]
        ax_mx.bar(idx, mr[1], bottom=bottom, width=bar_w, color='yellow', label='Med'   if idx==0 else "")
        bottom += mr[1]
        ax_mx.bar(idx, mr[2], bottom=bottom, width=bar_w, color='red',    label='Hard'  if idx==0 else "")
    ax_mx.set_xticks(range(num_phases))
    ax_mx.set_xticklabels([f"P{p}" for p in phases])
    ax_mx.set_title("Mixing Ratios"); ax_mx.set_xlabel("Phase"); ax_mx.set_ylabel("Ratio")
    ax_mx.legend(fontsize=8); ax_mx.grid(True, linestyle=':', alpha=0.6)

    # Reward evolution
    ax_rw = fig.add_subplot(gs_bot[0, 3])
    ax_rw.set_facecolor('#f5f5f5')
    ax_rw.plot(phases, rews, marker='D', linestyle='-', color='magenta', markersize=6)
    ax_rw.set_title("Reward"); ax_rw.set_xlabel("Phase"); ax_rw.set_ylabel("Reward")
    ax_rw.set_xticks(phases); ax_rw.grid(True, linestyle=':', alpha=0.6)

    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.98, hspace=0.6, wspace=0.4)

    out_f = os.path.join(output_dir, f"{group_name}_episode_{episode['index']}_detailed_fun.png")
    plt.savefig(out_f)
    plt.close(fig)
    print(f"Saved fun detailed figure for {group_name} episode {episode['index']} to {out_f}")

def save_episodes(episodes, path='episodes.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(episodes, f)

def load_episodes(path='episodes.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def main():
    pt_file = "/data/naddeok/spcl/merged_history.pt" # "seq_evo_results/history/current.pt" # "seq_evo_results/eval_parts/eval_gen0_part0.pt" #  "vec_evo_results_parallel/fixed_history/history_gen33.pt" # 
    # pkl_file = "vec_evo_results_parallel/episodes.pkl"
    output_dir = "/data/naddeok/spcl/merged_history/"
    num_bins = 16

    os.makedirs(output_dir, exist_ok=True)

    # Load data.
    states, actions, rewards, dones = load_data(pt_file)
    print(f"Loaded {states.shape[0]} transitions from {pt_file}")

    # Group transitions into episodes using the done flag.
    episodes = group_episodes_by_done(states, actions, rewards, dones)
    print(f"Grouped into {len(episodes)} episodes based on done flags.")
    
    # save_episodes(episodes, path=pkl_file)
    # episodes = load_episodes(path=pkl_file)

    # Select low, median, and high groups.
    low_eps, median_eps, high_eps = select_episode_groups(episodes)

    # Save text summaries.
    save_episode_details(low_eps, "low", num_bins, output_dir)
    save_episode_details(median_eps, "median", num_bins, output_dir)
    save_episode_details(high_eps, "high", num_bins, output_dir)

    # Plot reward distribution for all episodes.
    plot_reward_distribution(episodes, output_dir)

    # Generate detailed figures for each chosen episode.
    for group_name, group_eps in zip(["low", "median", "high"], [low_eps, median_eps, high_eps]):
        for ep in group_eps:
            plot_episode_figure(ep, group_name, num_bins, output_dir)

    print("Analysis complete.")

if __name__ == "__main__":
    main()


#==========================================
# File: eval_population.py
#==========================================

# eval_population.py
#!/usr/bin/env python3
import argparse
import torch
from tqdm import trange

from population_utils import (
    set_seed,
    load_config,
    evaluate_candidate,
    evaluate_candidate_parallel,
)
from curriculum_env import CurriculumEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True, help="path to your config YAML")
    p.add_argument("--pop_file",       required=True, help=".pt c`ntaining ‘population’")
    p.add_argument("--start_idx",    type=int,  required=True, help="first candidate index")
    p.add_argument("--num_candidates", type=int, required=True, help="how many candidates")
    p.add_argument("--out_file",       required=True, help="where to write this slice’s .pt")
    p.add_argument("--num_models", type=int, default=1, help="number of models to evaluate per candidate")
    p.add_argument("--model_type", choices=["cnn", "mlp"], default=None,
                   help="override model type from config")
    p.add_argument("--parallel", action="store_true",
                   help="use evaluate_candidate_parallel (for multi-model eval)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model_type:
        cfg["model_type"] = args.model_type
    set_seed(cfg.get("seed", 42))
    cfg["device"]    = "cuda:0"
    cfg["device_id"] = 0

    data       = torch.load(args.pop_file, map_location="cpu")
    population = data["population"]
    cand_dim   = population.size(1)
    num_phases = int(cfg["curriculum"].get("max_phases", 3))

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    agg_rewards, indices = [], []

    env = None
    if not args.parallel:
        env = CurriculumEnv(cfg)

    # tqdm over the slice
    for local_idx in trange(args.num_candidates,
                            desc=f"Eval gen slice {args.start_idx}-{args.start_idx+args.num_candidates-1}",
                            unit="cand"):
        idx = args.start_idx + local_idx
        if args.parallel:
            transitions, total_r = evaluate_candidate_parallel(
                cfg,
                population[idx],
                cand_dim,
                num_phases,
                args.num_models,
            )
        else:
            transitions, total_r = evaluate_candidate(
                env,
                population[idx],
                cand_dim,
                num_phases,
            )
        for (s, a, r, ns, d) in transitions:
            all_s .append(s)
            all_a .append(a)
            all_r .append(r)
            all_ns.append(ns)
            all_d .append(d)
        agg_rewards.append(total_r)
        indices    .append(idx)

    # save
    torch.save({
        'candidate_indices'  : torch.tensor(indices,    dtype=torch.int64),
        'aggregated_rewards' : torch.tensor(agg_rewards, dtype=torch.float32),
        'states'             : torch.stack(all_s),
        'actions'            : torch.stack(all_a),
        'rewards'            : torch.tensor(all_r,       dtype=torch.float32),
        'next_states'        : torch.stack(all_ns),
        'dones'              : torch.tensor(all_d,       dtype=torch.bool),
    }, args.out_file)
    print(f"Wrote eval slice → {args.out_file}")

if __name__ == "__main__":
    main()






#==========================================
# File: init_population.py
#==========================================

# init_population.py
#!/usr/bin/env python3
import argparse
import os
import torch

from population_utils import set_seed, load_config, initialize_population

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--output",    required=True,
                   help=".pt file to write generation‑0 population")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    num_phases    = cfg["curriculum"].get("max_phases", 3)
    unit          = 5
    candidate_dim = num_phases * unit
    pop_size      = cfg["rl"].get("ea_pop_size", 100)
    lr_range      = tuple(cfg["curriculum"]["learning_rate_range"])
    macro_actions = cfg["rl"].get("macro_actions", None)

    population = initialize_population(
        pop_size, candidate_dim, num_phases, macro_actions, lr_range
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'population': population}, args.output)
    print(f"Saved initial population ({pop_size} × {candidate_dim}) to {args.output}")

if __name__ == "__main__":
    main()






#==========================================
# File: evolve_population.py
#==========================================

#!/usr/bin/env python3
import argparse
import os
import glob
import torch
import random

from population_utils import load_config, mutate, crossover

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True)
    p.add_argument("--pop_file",          required=True)
    p.add_argument("--eval_dir",          required=True)
    p.add_argument("--gen",      type=int,required=True)
    p.add_argument("--output_population",required=True)
    p.add_argument("--history_dir",       required=True)
    args = p.parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    num_phases = cfg["curriculum"].get("max_phases", 3)
    lr_range   = tuple(cfg["curriculum"]["learning_rate_range"])
    top_k      = cfg["rl"].get("ea_top_k", 10)
    mut_rate   = cfg["rl"].get("ea_mutation_rate", 0.1)

    # Load current population
    pop_data   = torch.load(args.pop_file, map_location="cpu")
    population = pop_data["population"]                    # Tensor [pop_size, cand_dim]
    pop_size, cand_dim = population.shape

    # Prepare to collect rewards and history
    reward_map       = torch.zeros(pop_size, dtype=torch.float32)
    states_list      = []
    actions_list     = []
    rewards_list     = []
    next_states_list = []
    dones_list       = []

    # Gather per-candidate eval files
    pattern = os.path.join(args.eval_dir, f"eval_gen{args.gen}_part*.pt")
    for fn in sorted(glob.glob(pattern)):
        data = torch.load(fn, map_location="cpu")
        idxs = data["candidate_indices"]              # Tensor of indices
        rwd  = data["aggregated_rewards"]            # Tensor of rewards
        reward_map[idxs] = rwd

        # Accumulate history segments
        states_list.append(data["states"])
        actions_list.append(data["actions"])
        rewards_list.append(data["rewards"])
        next_states_list.append(data["next_states"])
        dones_list.append(data["dones"])

    # Concatenate all phases of this generation
    hs = torch.cat(states_list,      dim=0)
    ha = torch.cat(actions_list,     dim=0)
    hr = torch.cat(rewards_list,     dim=0)
    hn = torch.cat(next_states_list, dim=0)
    hd = torch.cat(dones_list,       dim=0)

    # Save this generation's history
    os.makedirs(args.history_dir, exist_ok=True)
    hist_fn = os.path.join(args.history_dir, f"history_gen{args.gen}.pt")
    torch.save({
        "states":      hs,
        "actions":     ha,
        "rewards":     hr,
        "next_states": hn,
        "dones":       hd
    }, hist_fn)
    print(f"Saved gen {args.gen} history to {hist_fn}")

    # Select top_k candidates by reward
    sel_idxs = torch.topk(reward_map, k=top_k, largest=True).indices
    selected = population[sel_idxs]  # Tensor [top_k, cand_dim]

    # Build next generation
    new_pop = []
    while len(new_pop) < pop_size:
        p1, p2 = random.sample(list(selected), 2)
        child  = crossover(p1, p2, cand_dim, num_phases, lr_range)
        child  = mutate(child, mut_rate, cand_dim, num_phases, lr_range)
        new_pop.append(child)
    new_pop = torch.stack(new_pop, dim=0)  # Tensor [pop_size, cand_dim]

    # Save the new population
    os.makedirs(os.path.dirname(args.output_population), exist_ok=True)
    torch.save({"population": new_pop}, args.output_population)
    print(f"Produced generation {args.gen+1} population → {args.output_population}")

if __name__ == "__main__":
    main()




#==========================================
# File: population_utils.py
#==========================================

# population_utils.py
#!/usr/bin/env python3

import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

from curriculum import get_mixed_loader
from curriculum_env import CurriculumEnv
from utils import _get_bin_edges, _MAX_LOSS

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def safe_normalize(arr: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a non-negative tensor so it sums to 1. If sum==0, returns uniform.
    """
    arr = torch.clamp(arr, min=0.0)
    total = arr.sum()
    if total.item() > 0.0:
        return arr / total
    return torch.full_like(arr, 1.0 / arr.numel())


def initialize_population(pop_size: int,
                          candidate_dim: int,
                          num_phases: int,
                          macro_actions: dict,
                          lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    population = []
    if macro_actions:
        macro_vals = [torch.tensor(v, dtype=torch.float32).flatten()
                      for v in macro_actions.values()]
        while len(population) < pop_size:
            base = random.choice(macro_vals).clone()
            if base.numel() != candidate_dim:
                base = base.repeat(num_phases)[:candidate_dim]
            cand = base + torch.randn(candidate_dim) * 0.1
            for p in range(num_phases):
                i = p * unit
                cand[i] = cand[i].clamp(lr_range[0], lr_range[1])
                cand[i+1:i+4] = safe_normalize(cand[i+1:i+4])
                cand[i+4] = cand[i+4].clamp(0.0, 1.0)
            population.append(cand)
    else:
        for _ in range(pop_size):
            cand = torch.zeros(candidate_dim)
            for p in range(num_phases):
                i = p * unit
                cand[i] = random.uniform(*lr_range)
                cand[i+1:i+4] = safe_normalize(torch.rand(3))
                cand[i+4] = random.random()
            population.append(cand)
    return torch.stack(population)

def mutate(child: torch.Tensor,
           mutation_rate: float,
           candidate_dim: int,
           num_phases: int,
           lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    m = child + torch.randn_like(child) * mutation_rate
    for p in range(num_phases):
        i = p * unit
        m[i] = m[i].clamp(lr_range[0], lr_range[1])
        m[i+1:i+4] = safe_normalize(m[i+1:i+4])
        m[i+4] = m[i+4].clamp(0.0, 1.0)
    return m

def crossover(p1: torch.Tensor,
              p2: torch.Tensor,
              candidate_dim: int,
              num_phases: int,
              lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    mask = torch.rand(candidate_dim) < 0.5
    
    child = p1.clone()
    child[mask] = p2[mask]
    for p in range(num_phases):
        i = p * unit
        child[i] = child[i].clamp(lr_range[0], lr_range[1])
        child[i+1:i+4] = safe_normalize(child[i+1:i+4])
        child[i+4] = child[i+4].clamp(0.0, 1.0)
    return child

def evaluate_candidate(env: CurriculumEnv,
                       candidate: torch.Tensor,
                       candidate_dim: int,
                       num_phases: int):
    """
    Evaluate a multi-phase candidate, with a tqdm over phases.
    Returns transitions list and total reward.
    """
    transitions = []
    total_reward = 0.0
    print("Initializing State")
    state = env.reset()
    print("State Initialized")
    unit = candidate_dim // num_phases
    done = False
    for p in trange(num_phases, desc="Eval phases", unit="phase", leave=False):
        action = candidate[state.new_zeros(())] if False else None  # placeholder
        
        idx = len(transitions) * unit
        action = candidate[idx:idx+unit]
        
        nxt, r, done = env.step(action)
        transitions.append((state, action, r, nxt, done))
        total_reward += r
        state = nxt
        if done:
            break

    return transitions, total_reward


class PaddedBatchedMLP(nn.Module):
    """Minimal padded MLP supporting a batch of differently-sized networks."""

    def __init__(self, configs, init_models=None, input_dim=28*28, output_dim=10):
        super().__init__()
        self.configs = configs
        self.num_models = len(configs)
        self.depths = [len(c) for c in configs]
        self.max_depth = max(self.depths)
        self.max_width = max(max(c) for c in configs)

        dims = [input_dim] + [self.max_width]*self.max_depth + [output_dim]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1], dims[i]))
            for i in range(len(dims)-1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1]))
            for i in range(len(dims)-1)
        ])

        # either copy from provided models or random init
        if init_models:
            for mi, model in enumerate(init_models):
                li = 0
                for m in model:
                    if isinstance(m, nn.Linear):
                        out, inp = m.weight.size()
                        self.weights[li].data[mi, :out, :inp].copy_(m.weight.data)
                        self.biases[li].data[mi, :out].copy_(m.bias.data)
                        li += 1
        else:
            for mi, cfg in enumerate(configs):
                full = [input_dim] + cfg + [output_dim]
                for li in range(len(full)-1):
                    out, inp = full[li+1], full[li]
                    self.weights[li].data[mi, :out, :inp].normal_(0.0, 0.1)
                    self.biases[li].data[mi, :out].normal_(0.0, 0.1)

        for li in range(self.max_depth):
            mask = torch.tensor([li < d for d in self.depths], dtype=torch.bool)
            self.register_buffer(f"mask_{li}", mask.view(-1,1,1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0).expand(self.num_models, -1, -1)
        for li, (W, B) in enumerate(zip(self.weights, self.biases)):
            y = torch.bmm(W.to(x.device), x.transpose(1,2)).transpose(1,2) + B.to(x.device).unsqueeze(1)
            if li == self.max_depth:
                x = y
            else:
                if li == 0:
                    x = F.relu(y)
                else:
                    mask = getattr(self, f"mask_{li}")
                    x = torch.where(mask, F.relu(y), x)
        return x


def _run_phase_training_batched(model, easy_loader, medium_loader, hard_loader, hp, device):
    """Train a PaddedBatchedMLP for one phase and return per-model rewards."""
    phase_batch_size = hp.get("phase_batch_size", 1024)
    criterion = nn.CrossEntropyLoss(reduction="none")

    mixed = get_mixed_loader(
        easy_loader.dataset,
        medium_loader.dataset,
        hard_loader.dataset,
        hp["mixture_ratio"],
        num_samples=hp["training_samples"],
        batch_size=phase_batch_size,
    )

    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    model.train()
    phase_samples = 0
    for imgs, labels in mixed:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        outs = model(imgs)
        lbl = labels.expand(model.num_models, -1)
        # Use reshape instead of view to avoid contiguity issues
        losses = criterion(outs.reshape(-1, outs.size(-1)), lbl.reshape(-1))
        loss_pm = losses.view(model.num_models, -1).mean(dim=1)
        loss_pm.sum().backward()
        opt.step()
        phase_samples += imgs.size(0)
        if phase_samples >= hp["training_samples"]:
            break

    def acc(loader):
        correct = torch.zeros(model.num_models, device=device)
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).argmax(dim=2)
                correct += (preds == yb).sum(dim=1).to(correct.dtype)
                total += yb.size(0)
        return correct / total * 100.0

    ea = acc(easy_loader)
    ma = acc(medium_loader)
    ha = acc(hard_loader)
    return ((ea + ma + ha) / 3.0).tolist()


def eval_loader_batched(model, loader, device, num_bins):
    """Vectorized ``eval_loader`` for ``PaddedBatchedMLP``.

    The normalization step divides each histogram row by the total count for
    that row.  ``totals.sum`` returns a ``(N, 1)`` tensor, so we squeeze the
    trailing dimension before using it as a boolean mask to select the rows to
    normalize.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    hist_c = torch.zeros(model.num_models, num_bins, device=device)
    hist_i = torch.zeros(model.num_models, num_bins, device=device)

    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    max_batches = int(len(loader) * 0.5)
    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outs = model(imgs)  # [num_models, batch, C]
            lbl = labels.expand(model.num_models, -1)
            losses = ce_loss(outs.reshape(-1, outs.size(-1)), lbl.reshape(-1))
            losses = losses.reshape(model.num_models, -1)
            preds = outs.argmax(dim=2)
            correct = preds.eq(labels)
            for mi in range(model.num_models):
                lc = losses[mi][correct[mi]].clamp(0.0, _MAX_LOSS)
                li = losses[mi][~correct[mi]].clamp(0.0, _MAX_LOSS)
                bc = torch.bucketize(lc, boundaries)
                bi_idx = torch.bucketize(li, boundaries)
                hist_c[mi] += torch.bincount(bc, minlength=num_bins).float()
                hist_i[mi] += torch.bincount(bi_idx, minlength=num_bins).float()
            if bi > max_batches:
                break

    totals = hist_c + hist_i
    S = totals.sum(dim=1, keepdim=True).clamp(min=1.0)
    hist_c /= S
    hist_i /= S
    return hist_c, hist_i


def evaluate_candidate_parallel(cfg: dict,
                                candidate: torch.Tensor,
                                candidate_dim: int,
                                num_phases: int,
                                num_models: int):
    """Evaluate a candidate using multiple random MLP models in parallel."""
    base_env = CurriculumEnv(cfg)
    base_env.reset()

    model_cfgs = []
    for _ in range(num_models):
        depth = random.randint(1, 3)
        widths = [random.randint(32, 128) for _ in range(depth)]
        model_cfgs.append(widths)

    depth_groups = {}
    for idx, cfgs in enumerate(model_cfgs):
        depth_groups.setdefault(len(cfgs), []).append(idx)

    group_models = {
        d: PaddedBatchedMLP([model_cfgs[i] for i in idxs]).to(base_env.device)
        for d, idxs in depth_groups.items()
    }

    def get_obs_group(vec_model):
        ec, ei = eval_loader_batched(vec_model, base_env.easy_loader, base_env.device, base_env.num_bins)
        mc, mi = eval_loader_batched(vec_model, base_env.medium_loader, base_env.device, base_env.num_bins)
        hc, hi = eval_loader_batched(vec_model, base_env.hard_loader, base_env.device, base_env.num_bins)
        counts = [len(base_env.easy_subset), len(base_env.medium_subset), len(base_env.hard_subset)]
        total = sum(counts)
        rel = torch.tensor([c/total for c in counts], device=base_env.device).view(1,-1)
        rel = rel.expand(vec_model.num_models, -1)
        obs = torch.cat([ec, ei, mc, mi, hc, hi, rel], dim=1)
        phase = torch.full((vec_model.num_models,1), base_env.current_phase/base_env.max_phases, device=base_env.device)
        avail = torch.full((vec_model.num_models,1), base_env.remaining_samples/base_env.train_samples_max, device=base_env.device)
        return torch.cat([obs, phase, avail], dim=1)

    states = [None for _ in range(num_models)]
    for d, idxs in depth_groups.items():
        obs = get_obs_group(group_models[d])
        for li, gi in enumerate(idxs):
            states[gi] = obs[li]

    unit = candidate_dim // num_phases
    remaining = base_env.train_samples_max
    total_rewards = [0.0 for _ in range(num_models)]
    transitions = []

    done = False
    for p in range(num_phases):
        idx = p * unit
        action = candidate[idx:idx+unit]
        lr, mix, frac = float(action[0]), action[1:4], float(action[4])
        num = int(frac * remaining)
        hp = {"training_samples": num, "learning_rate": lr,
              "mixture_ratio": mix.tolist(), "phase_batch_size": base_env.batch_size}

        rewards = [0.0 for _ in range(num_models)]
        for d, idxs in depth_groups.items():
            vec = group_models[d]
            r = _run_phase_training_batched(vec, base_env.easy_loader,
                                            base_env.medium_loader,
                                            base_env.hard_loader,
                                            hp, base_env.device)
            for li, gi in enumerate(idxs):
                rewards[gi] = r[li]

        remaining -= num
        base_env.remaining_samples = remaining
        base_env.current_phase += 1
        done = (base_env.current_phase >= base_env.max_phases) or remaining <= 0 or frac <= 0

        next_states = [None for _ in range(num_models)]
        for d, idxs in depth_groups.items():
            obs = get_obs_group(group_models[d])
            for li, gi in enumerate(idxs):
                next_states[gi] = obs[li]

        for j in range(num_models):
            transitions.append((states[j], action, rewards[j], next_states[j], done))
            total_rewards[j] += rewards[j]

        states = next_states
        if done:
            break

    return transitions, sum(total_rewards) / num_models






