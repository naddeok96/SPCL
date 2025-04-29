# population_utils.py
#!/usr/bin/env python3

import yaml
import torch
import random
import numpy as np

from tqdm import trange

from curriculum_env import CurriculumEnv

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0)
    total = arr.sum()
    return arr / total if total > 0 else np.ones_like(arr) / len(arr)

def initialize_population(pop_size: int,
                          candidate_dim: int,
                          num_phases: int,
                          macro_actions: dict,
                          lr_range: tuple) -> np.ndarray:
    unit = candidate_dim // num_phases
    population = []
    if macro_actions:
        macro_vals = [np.array(v).flatten() for v in macro_actions.values()]
        while len(population) < pop_size:
            base = random.choice(macro_vals)
            if base.size != candidate_dim:
                base = np.tile(base, num_phases)
            cand = base + np.random.normal(0, 0.1, candidate_dim)
            for p in range(num_phases):
                i = p * unit
                cand[i] = np.clip(cand[i], *lr_range)
                cand[i+1:i+4] = safe_normalize(cand[i+1:i+4])
                cand[i+4] = np.clip(cand[i+4], 0, 1)
            population.append(cand)
    else:
        for _ in range(pop_size):
            cand = np.zeros(candidate_dim)
            for p in range(num_phases):
                i = p * unit
                cand[i] = random.uniform(*lr_range)
                cand[i+1:i+4] = safe_normalize(np.random.rand(3))
                cand[i+4] = random.random()
            population.append(cand)
    return np.stack(population)

def mutate(child: np.ndarray,
           mutation_rate: float,
           candidate_dim: int,
           num_phases: int,
           lr_range: tuple) -> np.ndarray:
    unit = candidate_dim // num_phases
    m = child + np.random.normal(0, mutation_rate, child.shape)
    for p in range(num_phases):
        i = p * unit
        m[i] = np.clip(m[i], *lr_range)
        m[i+1:i+4] = safe_normalize(m[i+1:i+4])
        m[i+4] = np.clip(m[i+4], 0, 1)
    return m

def crossover(p1: np.ndarray,
              p2: np.ndarray,
              candidate_dim: int,
              num_phases: int,
              lr_range: tuple) -> np.ndarray:
    unit = candidate_dim // num_phases
    child = p1.copy()
    mask = np.random.rand(candidate_dim) < 0.5
    child[mask] = p2[mask]
    for p in range(num_phases):
        i = p * unit
        child[i] = np.clip(child[i], *lr_range)
        child[i+1:i+4] = safe_normalize(child[i+1:i+4])
        child[i+4] = np.clip(child[i+4], 0, 1)
    return child

def evaluate_candidate(env: CurriculumEnv,
                       candidate: np.ndarray,
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
        action = candidate[p * unit:(p + 1) * unit]
        nxt, r, done = env.step(action)
        transitions.append((state, action, r, nxt, done))
        total_reward += r
        state = nxt
        if done:
            break

    return transitions, total_reward
