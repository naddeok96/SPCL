"""
Simple replay buffer implementation for off-policy RL algorithms.
"""

import random
import numpy as np
from numpy.random import choice

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.
        
        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of samples to return.
            
        Returns:
            tuple: Batch of (state, action, reward, next_state, done) transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6, per_type="proportional"):
        super().__init__(capacity)
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.per_type = per_type

    def push(self, state, action, reward, next_state, done):
        # Assign max priority so new transitions are sampled once.
        max_prio = max(self.priorities) if self.priorities else 1.0
        super().push(state, action, reward, next_state, done)
        if len(self.priorities) < self.capacity:
            self.priorities.append(max_prio)
        else:
            self.priorities[self.position - 1] = max_prio

    def sample(self, batch_size):
        """
        Sample with priorities; returns:
          state, action, reward, next_state, done, weights, indices
        """
        
        N = len(self.buffer)
        if self.per_type == "proportional":
            prios = np.array(self.priorities, dtype=float) + float(self.epsilon)
            probs = prios ** self.alpha
        else:
            # rank-based
            sorted_idx = np.argsort(self.priorities)[::-1]
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(1, N+1)
            probs = 1.0 / (ranks ** self.alpha)
        probs = probs / probs.sum()
        indices = choice(N, batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        s, a, r, sp, d = map(np.stack, zip(*batch))
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return s, a, r, sp, d, weights, indices

    def update_priorities(self, indices, new_prios):
        """
        After learning, update the stored priorities.
        """
        for i, p in zip(indices, new_prios):
            self.priorities[i] = p













