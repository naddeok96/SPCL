"""
Simple replay buffer implementation for off-policy RL algorithms.
"""

import random
import torch

class ReplayBuffer:
    def __init__(self, capacity, device):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.
        
        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state.
            done (bool): Whether the episode ended.
        """
        entry = (state, action, torch.tensor([reward], device=self.device),
                 next_state, torch.tensor([done], device=self.device, dtype=torch.float32))
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of samples to return.
            
        Returns:
            tuple: Batch of (state, action, reward, next_state, done) transitions.
        """
        idxs = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idxs))
        return (torch.stack(states),
                torch.stack(actions),
                torch.cat(rewards),
                torch.stack(next_states),
                torch.cat(dones).float())

    def __len__(self):
        return len(self.buffer)

class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    """
    def __init__(self, capacity, device, alpha=0.6, beta=0.4, epsilon=1e-6, per_type="proportional"):
        super().__init__(capacity, device)
        self.priorities = torch.zeros(capacity, device=device)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.per_type = per_type

    def push(self, *args):
        super().push(*args)
        idx = (self.position - 1) % self.capacity
        max_p = self.priorities[:len(self.buffer)].max().item() if len(self.buffer) > 0 else 1.0
        self.priorities[idx] = max_p
        

    def sample(self, batch_size):
        """
        Sample with priorities; returns:
          state, action, reward, next_state, done, weights, indices
        """
        
        N = len(self.buffer)
        if self.per_type == "proportional":
            probs = (self.priorities[:N] + self.epsilon) ** self.alpha
        else:
            # rank-based
            ranks = torch.argsort(self.priorities[:N], descending=True).argsort().float() + 1
            probs = (1.0 / ranks) ** self.alpha
        probs = probs / probs.sum()

        idxs = torch.multinomial(probs, batch_size, replacement=True)
        weights = (N * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()

        batch = [self.buffer[i] for i in idxs.tolist()]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.stack(actions),
                torch.cat(rewards),
                torch.stack(next_states),
                torch.cat(dones).float(),
                weights.unsqueeze(1),
                idxs)

    def update_priorities(self, indices, new_prios):
        """
        After learning, update the stored priorities.
        """
        for i, p in zip(indices, new_prios):
            self.priorities[i] = p


class AugmentedReplayBuffer(ReplayBuffer):
    """Replay buffer that keeps an elite subset of transitions permanently."""

    def __init__(self, capacity, device, elite_fraction=0.1):
        super().__init__(capacity, device)
        self.elite_fraction = elite_fraction
        self.elite_capacity = int(capacity * elite_fraction)
        self.elite = []
        self.random = []
        self.random_pos = 0

    def _push_elite(self, *transition):
        if len(self.elite) < self.elite_capacity:
            self.elite.append(transition)
        else:
            idx = self.random_pos % self.elite_capacity
            self.elite[idx] = transition
            self.random_pos = (self.random_pos + 1) % self.elite_capacity

    def push(self, *args):
        if len(self.random) < self.capacity - self.elite_capacity:
            self.random.append(args)
        else:
            idx = self.position % (self.capacity - self.elite_capacity)
            self.random[idx] = args
        self.position = (self.position + 1) % (self.capacity - self.elite_capacity)

    def sample(self, batch_size):
        pool = self.elite + self.random
        idxs = random.sample(range(len(pool)), batch_size)
        batch = [pool[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.stack(actions),
                torch.cat(rewards),
                torch.stack(next_states),
                torch.cat(dones).float())

    def refresh_random(self, transitions):
        self.random = []
        self.position = 0
        for tr in transitions[: max(0, self.capacity - self.elite_capacity)]:
            self.push(*tr)

    def __len__(self):
        return len(self.elite) + len(self.random)


def build_augmented_replay_buffer(elite_data, all_data, capacity, elite_fraction, device="cpu"):
    """Create an :class:`AugmentedReplayBuffer` seeded with elite and random data."""
    buffer = AugmentedReplayBuffer(capacity, device, elite_fraction)

    def to_list(data):
        return list(zip(data["states"], data["actions"], data["rewards"], data["next_states"], data["dones"]))

    elite_trans = to_list(elite_data)
    for t in elite_trans[:buffer.elite_capacity]:
        s, a, r, ns, d = t
        buffer._push_elite(s.to(device), a.to(device), torch.tensor([float(r)], device=device), ns.to(device), torch.tensor([float(d)], device=device))

    def make_key(tr):
        key = []
        for item in tr:
            if torch.is_tensor(item):
                key.append(item.cpu().numpy().tobytes())
            else:
                key.append(item)
        return tuple(key)

    elite_keys = {make_key(t) for t in elite_trans}

    others = [t for t in to_list(all_data) if make_key(t) not in elite_keys]
    random.shuffle(others)
    buffer.refresh_random(others)

    return buffer

















