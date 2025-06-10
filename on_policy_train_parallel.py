#!/usr/bin/env python3
"""Parallel on-policy training with per-student datasets.

This script follows ``on_policy_train.py`` but keeps multiple student models
inside a vectorized ``PaddedBatchedMLP`` and trains them simultaneously on
separate datasets derived from the RL agent actions.
"""

import os
import yaml
import torch
import random
from typing import List, Dict

import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from rl_agent import DDPGAgent
from replay_buffer import ReplayBuffer, PERBuffer
from curriculum_env import CurriculumEnv
from population_utils import PaddedBatchedMLP, eval_loader_batched


class CombinedDataset(Dataset):
    """Stack multiple datasets item-wise."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.length = min(len(ds) for ds in datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        xs, ys = [], []
        for ds in self.datasets:
            x, y = ds[idx]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.tensor(ys)


class MixedDataset(Dataset):
    """Finite dataset sampled according to a mixture."""

    def __init__(self, easy_ds, medium_ds, hard_ds, mixture, num_samples):
        self.dataset = ConcatDataset([easy_ds, medium_ds, hard_ds])
        weights = [mixture[0]] * len(easy_ds) + [mixture[1]] * len(medium_ds) + [mixture[2]] * len(hard_ds)
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        self.indices = list(sampler)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ParallelMLP(PaddedBatchedMLP):
    """``PaddedBatchedMLP`` variant that accepts per-model batches."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` expected [num_models, batch, 1, 28, 28]
        n, b = x.size(0), x.size(1)
        x = x.view(n, b, -1)
        for li, (W, B) in enumerate(zip(self.weights, self.biases)):
            y = torch.bmm(W[:n], x.transpose(1, 2)).transpose(1, 2) + B[:n].unsqueeze(1)
            if li == self.max_depth:
                x = y
            else:
                if li == 0:
                    x = F.relu(y)
                else:
                    mask = getattr(self, f"mask_{li}")[:n]
                    x = torch.where(mask, F.relu(y), x)
        return x


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(cfg_file: str) -> dict:
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def build_vec_models(num: int) -> (Dict[int, List[int]], Dict[int, List[int]]):
    """Sample random model configs and group them by depth."""
    cfgs: List[List[int]] = []
    for _ in range(num):
        depth = random.randint(1, 3)
        widths = [random.randint(32, 128) for _ in range(depth)]
        cfgs.append(widths)

    depth_groups: Dict[int, List[int]] = {}
    for idx, c in enumerate(cfgs):
        depth_groups.setdefault(len(c), []).append(idx)

    return cfgs, depth_groups


def get_obs(vec_model: ParallelMLP, env: CurriculumEnv) -> torch.Tensor:
    ec, ei = eval_loader_batched(vec_model, env.easy_loader, env.device, env.num_bins)
    mc, mi = eval_loader_batched(vec_model, env.medium_loader, env.device, env.num_bins)
    hc, hi = eval_loader_batched(vec_model, env.hard_loader, env.device, env.num_bins)
    counts = [len(env.easy_subset), len(env.medium_subset), len(env.hard_subset)]
    total = sum(counts)
    rel = torch.tensor([c / total for c in counts], device=env.device).view(1, -1)
    rel = rel.expand(vec_model.num_models, -1)
    obs = torch.cat([ec, ei, mc, mi, hc, hi, rel], dim=1)
    phase = torch.full((vec_model.num_models, 1), env.current_phase / env.max_phases, device=env.device)
    avail = torch.full((vec_model.num_models, 1), env.remaining_samples / env.train_samples_max, device=env.device)
    return torch.cat([obs, phase, avail], dim=1)


def train_group(vec_model: ParallelMLP, env: CurriculumEnv, hyper_list: List[dict]) -> List[float]:
    device = env.device
    batch_size = hyper_list[0].get("phase_batch_size", 1024)
    datasets = [
        MixedDataset(env.easy_loader.dataset, env.medium_loader.dataset, env.hard_loader.dataset,
                     hp["mixture_ratio"], hp["training_samples"])
        for hp in hyper_list
    ]
    loader = DataLoader(CombinedDataset(datasets), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(vec_model.parameters(), lr=sum(hp["learning_rate"] for hp in hyper_list) / len(hyper_list))
    crit = nn.CrossEntropyLoss(reduction="none")
    vec_model.train()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        outs = vec_model(imgs)
        loss = crit(outs.reshape(-1, outs.size(-1)), labels.reshape(-1))
        loss_pm = loss.view(vec_model.num_models, -1).mean(dim=1)
        loss_pm.sum().backward()
        opt.step()

    ea, _ = eval_loader_batched(vec_model, env.easy_loader, device, env.num_bins)
    ma, _ = eval_loader_batched(vec_model, env.medium_loader, device, env.num_bins)
    ha, _ = eval_loader_batched(vec_model, env.hard_loader, device, env.num_bins)
    return ((ea + ma + ha) / 3.0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["paths"]["on_policy_parallel_dir"], exist_ok=True)
    ckpt_dir = cfg["paths"]["on_policy_parallel_dir"]

    set_seed(cfg.get("seed", 42))

    env = CurriculumEnv(cfg)
    obs_dim = env.num_bins * 6 + 5
    action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, cfg)

    num_students = cfg["rl"].get("num_parallel_students", 4)
    model_cfgs, depth_groups = build_vec_models(num_students)
    group_models = {
        d: ParallelMLP([model_cfgs[i] for i in idxs]).to(env.device)
        for d, idxs in depth_groups.items()
    }

    if cfg["rl"].get("per_enabled", False):
        replay_buffer = PERBuffer(
            cfg["rl"]["buffer_size"],
            cfg["device"],
            alpha=cfg["rl"].get("per_alpha", 0.6),
            beta=cfg["rl"].get("per_beta", 0.4),
            epsilon=cfg["rl"].get("per_epsilon", 1e-6),
            per_type=cfg["rl"].get("per_type", "proportional"),
        )
    else:
        replay_buffer = ReplayBuffer(cfg["rl"]["buffer_size"], cfg["device"])

    num_episodes = cfg["rl"].get("on_policy_episodes", 50)
    batch_size = cfg["rl"]["batch_size"]

    for ep in tqdm(range(1, num_episodes + 1), desc="Parallel Episodes"):
        env.reset()

        states = [None for _ in range(num_students)]
        for d, idxs in depth_groups.items():
            obs = get_obs(group_models[d], env)
            for li, gi in enumerate(idxs):
                states[gi] = obs[li]

        s_batch = torch.stack(states).to(agent.device)
        with torch.no_grad():
            actions = agent.actor(s_batch)

        rewards = [0.0 for _ in range(num_students)]
        next_states = [None for _ in range(num_students)]

        remaining = env.train_samples_max
        for d, idxs in depth_groups.items():
            vec = group_models[d]
            hyper_list = []
            for li, gi in enumerate(idxs):
                act = actions[gi]
                lr, mix, frac = float(act[0]), act[1:4], float(act[4])
                num = int(frac * remaining)
                hyper_list.append({
                    "training_samples": num,
                    "learning_rate": lr,
                    "mixture_ratio": mix.tolist(),
                    "phase_batch_size": env.batch_size,
                })
            r = train_group(vec, env, hyper_list)
            for li, gi in enumerate(idxs):
                rewards[gi] = r[li]

        for d, idxs in depth_groups.items():
            obs = get_obs(group_models[d], env)
            for li, gi in enumerate(idxs):
                next_states[gi] = obs[li]

        done = True
        for j in range(num_students):
            replay_buffer.push(states[j].cpu(), actions[j].cpu(), rewards[j], next_states[j].cpu(), done)

        if len(replay_buffer) >= batch_size:
            agent.update(replay_buffer, batch_size)

        if ep % max(1, int(num_episodes * 0.1)) == 0:
            for d, idxs in depth_groups.items():
                vec = group_models[d]
                for li, gi in enumerate(idxs):
                    layers = []
                    in_dim = 28 * 28
                    for width in model_cfgs[gi]:
                        layers.append(nn.Linear(in_dim, width))
                        layers.append(nn.ReLU())
                        in_dim = width
                    layers.append(nn.Linear(in_dim, 10))
                    single = nn.Sequential(nn.Flatten(), *layers).to(env.device)
                    li2 = 0
                    for m in single:
                        if isinstance(m, nn.Linear):
                            w = vec.weights[li2][li, :m.out_features, :m.in_features]
                            b = vec.biases[li2][li, :m.out_features]
                            m.weight.data.copy_(w)
                            m.bias.data.copy_(b)
                            li2 += 1
                    torch.save(single.state_dict(), os.path.join(ckpt_dir, f"student_{gi}_ep{ep}.pt"))


if __name__ == "__main__":
    main()
