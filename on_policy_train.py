#!/usr/bin/env python3
"""
on_policy_train.py

This script implements on-policy training for the curriculum learning experiment with RL-based
hyperparameter control. It loads a pre-trained off-policy actor & critic (if paths are provided
in config.yaml), then after each episode logs metrics and saves checkpoints and plots under
the on_policy_dir specified in config.
"""

import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

from curriculum_env import CurriculumEnv
from rl_agent import DDPGAgent
from replay_buffer import ReplayBuffer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--load_model", type=str, default=None, help="Override actor model path")
    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure base save path exists
    os.makedirs(config["paths"]["save_path"], exist_ok=True)
    # Use the on-policy subdir from config
    ckpt_dir = config["paths"]["on_policy_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(config.get("seed", 42))

    # Initialize environment and agent
    env = CurriculumEnv(config)
    obs_dim = len(env.reset())
    action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, config)

    # 1) Load actor
    actor_path = args.load_model \
                 or config["paths"].get("off_policy_actor_model", None)
    if actor_path and os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path))
        print(f"Loaded actor from {actor_path}")

    # 2) Load critic (if provided)
    critic_path = config["paths"].get("off_policy_critic_model", None)
    if critic_path and os.path.exists(critic_path):
        agent.critic.load_state_dict(torch.load(critic_path))
        print(f"Loaded critic from {critic_path}")

    replay_buffer = ReplayBuffer(config["rl"]["buffer_size"])

    num_episodes = config.get("on_policy_episodes", 50)
    batch_size   = config["rl"]["batch_size"]

    # Trackers
    episode_rewards   = []
    all_actor_losses  = []
    all_critic_losses = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        # One on-policy step
        action = agent.select_action(state, noise_enable=True)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        ep_reward += reward

        # Perform multiple updates
        actor_losses = []
        critic_losses = []
        if len(replay_buffer) >= batch_size:
            for _ in range(config.get("on_policy_updates_per_episode", 10)):
                metrics = agent.update(replay_buffer, batch_size)
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])

        # Record means (or zero if none)
        all_actor_losses.append(float(np.mean(actor_losses))  if actor_losses  else 0.0)
        all_critic_losses.append(float(np.mean(critic_losses)) if critic_losses else 0.0)
        episode_rewards.append(ep_reward)


        if ep % interval == 0:
            print(f"Episode {ep}/{num_episodes} — Reward: {ep_reward:.2f}%")

            # ── Checkpoint: save models ───────────────────────────────────────────
            torch.save(agent.actor.state_dict(),
                    os.path.join(ckpt_dir, f"actor_ep{ep}.pth"))
            torch.save(agent.critic.state_dict(),
                    os.path.join(ckpt_dir, f"critic_ep{ep}.pth"))

            # ── Plot rewards ─────────────────────────────────────────────────────
            plt.figure()
            plt.plot(episode_rewards, marker='o')
            plt.title("On-policy Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward (%)")
            plt.grid(True)
            plt.savefig(os.path.join(ckpt_dir, f"rewards_ep{ep}.png"))
            plt.close()

            # ── Plot losses ──────────────────────────────────────────────────────
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
            ax1.plot(all_actor_losses, color='blue')
            ax1.set_title("Actor Loss (mean per epi)")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Loss")
            ax2.plot(all_critic_losses, color='red')
            ax2.set_title("Critic Loss (mean per epi)")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Loss")
            fig.tight_layout()
            fig.savefig(os.path.join(ckpt_dir, f"losses_ep{ep}.png"))
            plt.close(fig)

    # ── Final save ─────────────────────────────────────────────────────────
    final_actor = os.path.join(ckpt_dir, "actor_final.pth")
    final_critic = os.path.join(ckpt_dir, "critic_final.pth")
    torch.save(agent.actor.state_dict(),  final_actor)
    torch.save(agent.critic.state_dict(), final_critic)
    print(f"Saved final actor → {final_actor}")
    print(f"Saved final critic → {final_critic}")

if __name__ == "__main__":
    main()
