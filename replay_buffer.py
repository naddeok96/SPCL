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

















