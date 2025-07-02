"""
behavior_cloning_train.py

This script performs the behavior cloning and critic tuning steps from
``off_policy_train.py`` as a standalone procedure.  Losses during the actor
(supervised behavioral cloning) stage and the critic tuning stage are recorded
and plotted to ``results/behavior_cloning``.
"""

import os
import yaml
import random
import argparse
import itertools
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from rl_agent import DDPGAgent
from curriculum_env import CurriculumEnv
from utils import select_top_percent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def behavior_clone_logged(policy, expert_data, iters=1000, batch_size=64, weight_decay=1e-4, device=None):
    """Train ``policy`` on ``expert_data`` while logging MSE losses."""
    import torch.nn as nn
    import torch.optim as optim

    device = device or next(policy.parameters()).device
    device = torch.device(device)

    ds = TensorDataset(expert_data["states"].to(device), expert_data["actions"].to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    data_iter = itertools.cycle(loader)

    policy.train()
    opt = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(iters):
        s, a = next(data_iter)
        pred = policy(s)
        loss = loss_fn(pred, a)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def tune_value_function_logged(policy, value_net, data_loader, config, iters=1000):
    """Tuned value function with CQL regularizer while logging loss."""
    import torch.nn.functional as F
    import torch.optim as optim

    device = next(value_net.parameters()).device
    gamma = config.get("gamma", 0.99)
    alpha = config.get("cql_alpha", 1.0)

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()

    optimiser = optim.Adam(value_net.parameters(), lr=config.get("critic_lr", 3e-4))

    data_iter = itertools.cycle(data_loader)
    losses = []
    for _ in range(iters):
        states, actions, rewards, next_states, dones = next(data_iter)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

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

        losses.append(loss.item())
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="path to config YAML"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    results_dir = os.path.join("results", "behavior_cloning")
    os.makedirs(results_dir, exist_ok=True)

    dataset_path = config["paths"]["pretrain_path"]
    data = torch.load(dataset_path, map_location="cpu")
    dataset = {
        "states": data["states"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "next_states": data["next_states"],
        "dones": data["dones"],
    }

    env = CurriculumEnv(config)
    obs_dim = len(env.reset())
    action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, config)

    top_percent = config["rl"].get("bc_top_percent", 10)
    elite_data = select_top_percent(dataset, top_percent)

    bc_iters = config["rl"].get("pretrain_bc_iters", 1000)
    bc_losses = behavior_clone_logged(agent.actor, elite_data, iters=bc_iters, batch_size=config["rl"]["batch_size"])

    ds = TensorDataset(dataset["states"], dataset["actions"], dataset["rewards"], dataset["next_states"], dataset["dones"])
    loader = DataLoader(ds, batch_size=config["rl"]["batch_size"], shuffle=True)

    critic_iters = config["rl"].get("pretrain_critic_iters", 1000)
    c1_losses = tune_value_function_logged(agent.actor, agent.critic1, loader, config["rl"], iters=critic_iters)
    c2_losses = tune_value_function_logged(agent.actor, agent.critic2, loader, config["rl"], iters=critic_iters)

    torch.save(agent.actor.state_dict(), os.path.join(results_dir, "actor.pth"))
    torch.save(agent.critic1.state_dict(), os.path.join(results_dir, "critic1.pth"))
    torch.save(agent.critic2.state_dict(), os.path.join(results_dir, "critic2.pth"))

    plt.figure()
    plt.plot(bc_losses)
    plt.xlabel("Update")
    plt.ylabel("MSE Loss")
    plt.title("Behavior Cloning Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "behavior_cloning_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(c1_losses, label="Critic1")
    plt.plot(c2_losses, label="Critic2")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Critic Tuning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "critic_tuning_loss.png"))
    plt.close()


if __name__ == "__main__":
    main()
