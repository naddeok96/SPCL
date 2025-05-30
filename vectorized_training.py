#!/usr/bin/env python3
"""Simple vectorized MLP training used by run_evolution.sh."""
import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vectorized_mlp_utils import PaddedBatchedMLP


def make_mlp(widths, idx, input_dim, output_dim):
    dims = [input_dim] + list(widths) + [output_dim]
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    cnt = 0
    seed_base = 1000
    for m in model:
        if isinstance(m, nn.Linear):
            base = seed_base + idx*100 + cnt*2
            torch.manual_seed(base)
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.manual_seed(base + 1)
            nn.init.normal_(m.bias, mean=0.0, std=0.1)
            cnt += 1
    return model


def train_vectorized(model_configs, epochs, batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    loader = DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    input_dim = 28*28
    output_dim = 10
    depth_groups = defaultdict(list)
    for idx, cfg in enumerate(model_configs):
        depth_groups[len(cfg)].append(idx)
    trained_state = {}
    for depth, group_idxs in depth_groups.items():
        cfgs = [model_configs[i] for i in group_idxs]
        model = PaddedBatchedMLP(cfgs, group_idxs, input_dim, output_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                outs = model(xb.unsqueeze(0).expand(len(group_idxs), -1, -1))
                loss_pm = F.cross_entropy(
                    outs.view(-1, output_dim),
                    yb.repeat(len(group_idxs)),
                    reduction="none"
                ).view(len(group_idxs), -1).mean(dim=1)
                loss_pm.sum().backward()
                opt.step()
                opt.zero_grad()
        for li, gi in enumerate(group_idxs):
            single = make_mlp(model_configs[gi], gi, input_dim, output_dim).to(device)
            ptr = 0
            for m in single:
                if isinstance(m, nn.Linear):
                    w = model.weights[ptr][li, :m.out_features, :m.in_features]
                    b = model.biases[ptr][li, :m.out_features]
                    m.weight.data.copy_(w)
                    m.bias.data.copy_(b)
                    ptr += 1
            trained_state[gi] = single.state_dict()
    return trained_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON list of layer widths for each model")
    p.add_argument("--output", required=True, help="Path to save state dicts")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.config, "r") as f:
        configs = json.load(f)
    states = train_vectorized(configs, args.epochs, args.batch_size, torch.device(args.device))
    torch.save(states, args.output)
    print(f"Saved trained models → {args.output}")


if __name__ == "__main__":
    main()
