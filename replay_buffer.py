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

    for s0, s1 in _iter_shards(n, shard_size):
        idx = indices[s0:s1]
        for i in idx.tolist():
            buffer.push(states[i], actions[i], float(rewards[i]), next_states[i], bool(dones[i]))


def build_replay_buffer_streaming(
    data: Dict[str, torch.Tensor],
    capacity: int,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_epsilon: float = 1e-6,
    per_type: str = "proportional",
    shard_size: int = 200_000,
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

    stream_into_buffer(buffer, data, shard_size=shard_size, shuffle=True)
    return buffer
