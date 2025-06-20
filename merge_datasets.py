#!/usr/bin/env python3
"""Merge multiple evolutionary datasets without command-line arguments.

This script combines ``.pt`` history files produced by the evolutionary
pipeline (``run_evolution.sh`` or ``run_evolution_parallel.sh``). All
paths are configured below; adjust them for your setup.
"""

import os
import glob
import torch
import matplotlib.pyplot as plt

# ==== Configuration ====
OUTPUT_PATH = "merged_history.pt"  # where to save the merged dataset
HISTOGRAM_PATH = "merged_reward_hist.png"  # reward distribution plot
FINAL_REWARD_HIST_PATH = "final_reward_hist.png"  # final reward distribution plot
HISTORY_DIRS = [
    "vec_evo_results_parallel/history",
    "seq_evo_results/history",
]


def load_dataset(path: str) -> dict:
    """Load a saved evolutionary dataset."""
    return torch.load(path, map_location="cpu")


def main() -> None:
    # Gather all history files from the configured directories
    input_files = []
    for d in HISTORY_DIRS:
        input_files.extend(sorted(glob.glob(os.path.join(d, "*.pt"))))

    if not input_files:
        print("No history files found. Check HISTORY_DIRS paths.")
        return

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for path in input_files:
        data = load_dataset(path)
        states.append(data["states"])
        actions.append(data["actions"])
        rewards.append(data["rewards"])
        next_states.append(data["next_states"])
        dones.append(data["dones"])

    S = torch.cat(states, dim=0)
    A = torch.cat(actions, dim=0)
    R = torch.cat(rewards, dim=0)
    NS = torch.cat(next_states, dim=0)
    D = torch.cat(dones, dim=0)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    torch.save({
        "states": S,
        "actions": A,
        "rewards": R,
        "next_states": NS,
        "dones": D,
    }, OUTPUT_PATH)

    # Basic statistics
    n_trans = S.size(0)
    n_episodes = int(D.sum().item()) + (0 if D[-1] else 1)
    print(f"Saved merged dataset to {OUTPUT_PATH}")
    print(f"Input files merged: {len(input_files)}")
    print(f"Total transitions: {n_trans}")
    print(f"Total episodes: {n_episodes}")
    print(f"Reward range: {R.min().item():.2f} to {R.max().item():.2f}")
    print(f"Average reward: {R.mean().item():.2f}")

    # Compute final rewards (rewards at episode termination).
    final_rewards = R[D]
    if not D[-1]:
        final_rewards = torch.cat((final_rewards, R[-1:].clone()))
    print(f"Final rewards collected: {final_rewards.numel()}")
    print(f"Final reward range: {final_rewards.min().item():.2f} to {final_rewards.max().item():.2f}")
    print(f"Average final reward: {final_rewards.mean().item():.2f}")

    # Plot reward distribution
    plt.figure()
    plt.hist(R.cpu().numpy(), bins=30, edgecolor="black", color="skyblue")
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.savefig(HISTOGRAM_PATH)
    plt.close()
    print(f"Saved reward histogram to {HISTOGRAM_PATH}")

    # Plot final reward distribution
    plt.figure()
    plt.hist(final_rewards.cpu().numpy(), bins=30, edgecolor="black", color="salmon")
    plt.title("Final Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.savefig(FINAL_REWARD_HIST_PATH)
    plt.close()
    print(f"Saved final reward histogram to {FINAL_REWARD_HIST_PATH}")


if __name__ == "__main__":
    main()
