#!/usr/bin/env python3
"""
compare_models.py

Evaluate and compare arbitrary actor + critic checkpoints defined in config.yaml.

Expect your config.yaml to include a `compare_models` section, e.g.:

compare_models:
  First Model:
    actor:  "results/off_policy/actor_before_training.pth"
    critic: "results/off_policy/critic_before_training.pth"
  Off-Policy-End:
    actor:  "results/off_policy/off_policy_actor_model_final.pth"
    critic: "results/off_policy/off_policy_critic_model_final.pth"
  On-Policy-Final:
    actor:  "results/on_policy/actor_final.pth"
    critic: "results/on_policy/critic_final.pth"

This script will skip any entry missing actor or critic paths, and evaluate all remaining models.
It uses `tqdm` to display progress bars during actor rollouts, model evaluation, and probe state collection.

Outputs:
  * Detailed episode plots via `plot_episode_figure`
  * Bar chart of mean ± std returns
  * Histograms of return distributions
  * Bar chart & histograms of critic Q-value distributions on probe states
"""
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from curriculum_env import CurriculumEnv
from rl_agent import DDPGAgent
from off_policy_train import plot_episode_figure


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_actor(env, agent, episodes):
    """
    Run `episodes` full rollouts (no exploration noise),
    returning list of total rewards and the first episode trajectory.
    Uses tqdm to show progress.
    """
    returns = []
    first_ep = None
    for ep in trange(episodes, desc="Actor eval"):  # progress bar
        obs = env.reset()
        done = False
        total_r = 0.0
        states, actions, rewards = [], [], []
        while not done:
            a = agent.select_action(obs, noise_enable=False)
            states.append(obs)
            actions.append(a)
            obs, r, done = env.step(a)
            rewards.append(r)
            total_r += r
        returns.append(total_r)
        if ep == 0:
            first_ep = {"index": 0,
                        "states":  np.array(states),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards)}
    return returns, first_ep


def evaluate_critic(agent, probe_states):
    """
    Compute critic Q-values for given probe_states and agent's policy.
    Returns array of Q-values.
    """
    device = agent.device
    with torch.no_grad():
        s_t = torch.FloatTensor(probe_states).to(device)
        a_t = agent.actor(s_t).to(device)
        q_t = agent.critic(s_t, a_t)
    return q_t.cpu().numpy().flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml",
                   help="Path to config.yaml containing `compare_models` dict")
    p.add_argument("--eval_episodes", "-n", type=int, default=20,
                   help="Number of episodes per model for actor evaluation")
    p.add_argument("--probe_size", "-m", type=int, default=256,
                   help="Number of probe states for critic evaluation")
    args = p.parse_args()

    config = load_config(args.config)
    model_defs = config.get("compare_models", {})
    if not isinstance(model_defs, dict) or not model_defs:
        print("No models defined under 'compare_models'. Exiting.")
        return

    # Setup environment
    env = CurriculumEnv(config)
    obs_dim = len(env.reset())
    action_dim = 5

    # Prepare probe states for critic eval: sample from EA data if available
    probe_states = None
    pretrained = config.get("paths", {}).get("pretrain_path")
    if pretrained and os.path.exists(pretrained):
        data = np.load(pretrained)
        all_states = data.get("states")
        if all_states is not None and len(all_states) >= args.probe_size:
            idx = np.random.choice(len(all_states), args.probe_size, replace=False)
            probe_states = all_states[idx]
    if probe_states is None:
        # fallback: roll random policy
        probe_states = []
        st = env.reset()
        for _ in trange(args.probe_size, desc="Collect probe states"):
            a = np.random.uniform(size=action_dim)
            st, _, done = env.step(a)
            probe_states.append(st)
            if done:
                st = env.reset()
        probe_states = np.array(probe_states)

    results = {}
    # Iterate over models with a progress bar
    for name, paths in tqdm(model_defs.items(), desc="Models"):  
        actor_ckpt = paths.get("actor")
        critic_ckpt = paths.get("critic")
        if not actor_ckpt or not critic_ckpt:
            print(f"Skipping '{name}' — actor or critic path missing.")
            continue
        if not os.path.exists(actor_ckpt) or not os.path.exists(critic_ckpt):
            print(f"Skipping '{name}' — checkpoint file not found.")
            continue

        agent = DDPGAgent(obs_dim, action_dim, config)
        agent.actor.load_state_dict(torch.load(actor_ckpt, map_location=agent.device))
        agent.critic.load_state_dict(torch.load(critic_ckpt, map_location=agent.device))
        agent.actor.eval(); agent.critic.eval()

        actor_rets, first_ep = evaluate_actor(env, agent, args.eval_episodes)
        critic_vals = evaluate_critic(agent, probe_states)
        results[name] = {
            "actor_rets": actor_rets,
            "first_ep": first_ep,
            "critic_vals": critic_vals
        }

    if not results:
        print("No models evaluated. Exiting.")
        return

    out_dir = os.path.join("results", "model_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # Actor: detailed first episode plots
    for name, data in results.items():
        plot_episode_figure(
            data["first_ep"],
            f"compare_{name}",
            config["observation"]["num_bins"],
            out_dir
        )

    # Actor: bar + hist returns
    labels, means, stds = [], [], []
    for name, data in results.items():
        arr = np.array(data["actor_rets"])
        labels.append(name); means.append(arr.mean()); stds.append(arr.std())
    x = np.arange(len(labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Total Reward")
    plt.title(f"Actor Return Comparison (n={args.eval_episodes})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "actor_returns_bar.png")); plt.close()
    for name, data in results.items():
        plt.figure(figsize=(5,4))
        plt.hist(data["actor_rets"], bins=20, alpha=0.7)
        plt.title(f"{name} Actor Returns")
        plt.xlabel("Return"); plt.ylabel("Freq")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_actor_hist.png")); plt.close()

    # Critic: bar + hist Q-values
    labels, c_means, c_stds = [], [], []
    for name, data in results.items():
        arr = data["critic_vals"]
        labels.append(name); c_means.append(arr.mean()); c_stds.append(arr.std())
    x = np.arange(len(labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, c_means, yerr=c_stds, capsize=5)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Q-Value")
    plt.title(f"Critic Q-Value Comparison (n={args.probe_size})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "critic_q_bar.png")); plt.close()
    for name, data in results.items():
        plt.figure(figsize=(5,4))
        plt.hist(data["critic_vals"], bins=20, alpha=0.7)
        plt.title(f"{name} Critic Q-Values")
        plt.xlabel("Q-Value"); plt.ylabel("Freq")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_critic_hist.png")); plt.close()

    print(f"Saved comparison assets under '{out_dir}'")

if __name__ == "__main__":
    main()
