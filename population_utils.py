# population_utils.py
#!/usr/bin/env python3

import yaml
import torch
import random
import torch

from tqdm import trange

from curriculum_env import CurriculumEnv

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




