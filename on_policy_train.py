#!/usr/bin/env python3
"""
on_policy_train.py

This script implements on-policy training for the curriculum learning experiment
with RL-based hyperparameter control. It loads a pre-trained off-policy actor &
critic (paths provided in config.yaml), then after each episode logs metrics and
saves checkpoints and plots under the on_policy_dir specified in config.
"""

import os
import yaml
import torch

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import torch.nn.utils as utils

from off_policy_train import plot_episode_figure
from curriculum_env import CurriculumEnv
from rl_agent import DDPGAgent
from replay_buffer import ReplayBuffer, PERBuffer

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
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure base save path exists
    os.makedirs(config["paths"]["on_policy_dir"], exist_ok=True)
    ckpt_dir = config["paths"]["on_policy_dir"]

    set_seed(config.get("seed", 42))

    # Initialize environment and agent
    env = CurriculumEnv(config)
    obs_dim = len(env.reset())
    action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, config)

    # Load actor
    actor_path = config["paths"].get("on_policy_actor_model", None)
    if actor_path and os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print(f"Loaded actor from {actor_path}")

    # Load critic
    critic_path = config["paths"].get("on_policy_critic_model", None)
    if critic_path and os.path.exists(critic_path):
        agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
        print(f"Loaded critic from {critic_path}")

     # ── Setup replay buffer and optionally preload from EA dataset ────────────
    if config["rl"].get("per_enabled", False):
        replay_buffer = PERBuffer(
            config["rl"]["buffer_size"],
            alpha=config["rl"].get("per_alpha", 0.6),
            beta=config["rl"].get("per_beta", 0.4),
            epsilon=config["rl"].get("per_epsilon", 1e-6),
            per_type=config["rl"].get("per_type", "proportional")
        )
    else:
        replay_buffer = ReplayBuffer(config["rl"]["buffer_size"])

    # If desired, seed the on‑policy buffer from your EA dataset:
    if config["rl"].get("seed_replay_buffer", False):
        ea_path = config["paths"]["pretrain_path"]
        data   = np.load(ea_path)
        states       = data["states"]
        actions      = data["actions"]
        rewards      = data["rewards"]
        next_states  = data["next_states"]
        dones        = data["dones"]
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            replay_buffer.push(s, a, r, ns, d)
        print(f"Seeded {len(replay_buffer)} transitions from EA dataset at '{ea_path}'")

    num_episodes = config["rl"].get("on_policy_episodes", 50)
    batch_size   = config["rl"]["batch_size"]
    interval     = max(1, int(num_episodes * 0.05))

        # ── Pre‑sample states for variance tracking ─────────────────────────────
    probe_batch_size = config["rl"].get("probe_batch_size", 256)
    probe_from_ea   = config["rl"].get("probe_from_ea", False)

    if probe_from_ea:
        # load EA states from the evolutionary dataset (.npz)
        data          = np.load(config["paths"]["pretrain_path"])
        all_states    = data["states"]  # shape (N, state_dim)
        indices       = np.random.choice(len(all_states),
                                         size=probe_batch_size,
                                         replace=False)
        probe_states  = all_states[indices]
    else:
        # roll the env forward to collect diverse states
        probe_states = []
        state = env.reset()
        for _ in tqdm(range(probe_batch_size), desc="Probing"):
            action, _, _ = agent.select_action(state, noise_enable=True), None, None
            next_state, _, done = env.step(action)
            probe_states.append(state)
            state = env.reset() if done else next_state
        probe_states = np.array(probe_states)

    probe_states_tensor = torch.FloatTensor(probe_states).to(agent.device)

    # Trackers
    episode_rewards    = []
    all_actor_losses   = []
    all_critic_losses  = []
    action_var_history = []
    update_steps       = []

    for ep in tqdm(range(1, num_episodes + 1), desc="On‑policy Episodes"):
        state = env.reset()
        ep_reward = 0.0

        # One on-policy step
        action = agent.select_action(state, noise_enable=True)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        ep_reward += reward

        # Perform multiple updates
        actor_losses  = []
        critic_losses = []
        if len(replay_buffer) >= batch_size:
            for _ in range(config["rl"].get("on_policy_updates_per_episode", 10)):
                metrics = agent.update(replay_buffer, batch_size)
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])

        all_actor_losses.append(float(np.mean(actor_losses))  if actor_losses  else 0.0)
        all_critic_losses.append(float(np.mean(critic_losses)) if critic_losses else 0.0)
        episode_rewards.append(ep_reward)

        # Action‑component variance
        with torch.no_grad():
            preds = agent.actor(probe_states_tensor).cpu().numpy()
        var = np.var(preds, axis=0)
        action_var_history.append(var)
        update_steps.append(ep)

        if ep % interval == 0:
            print(f"Episode {ep}/{num_episodes} — Reward: {ep_reward:.2f}%")

            # ── Save models ───────────────────────────────
            torch.save(agent.actor.state_dict(),
                       os.path.join(ckpt_dir, f"actor_ep{ep}.pth"))
            torch.save(agent.critic.state_dict(),
                       os.path.join(ckpt_dir, f"critic_ep{ep}.pth"))

            # ── Plot rewards ──────────────────────────────
            plt.figure()
            plt.plot(episode_rewards, marker='o')
            plt.title("On-policy Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward (%)")
            plt.grid(True)
            plt.savefig(os.path.join(ckpt_dir, f"rewards_ep{ep}.png"))
            plt.close()

            # ── Plot losses ───────────────────────────────
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

            # ── Plot action variance ──────────────────────
            var_array = np.vstack(action_var_history)
            names = ["learning_rate","mix_easy","mix_med","mix_hard","sample_usage"]
            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6,10), sharex=True)
            for idx, name in enumerate(names):
                axs[idx].plot(update_steps, var_array[:,idx])
                axs[idx].set_ylabel(name)
            axs[-1].set_xlabel("Episode")
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f"action_variance_ep{ep}.png"))
            plt.close(fig)

            # ── Periodic evaluation ───────────────────────
            eval_states, eval_actions, eval_rewards = [], [], []
            obs_eval, done = env.reset(), False
            while not done:
                a_eval = agent.select_action(obs_eval, noise_enable=False)
                eval_states.append(obs_eval)
                eval_actions.append(a_eval)
                obs_eval, r, done = env.step(a_eval)
                eval_rewards.append(r)
            print(f"Eval Episode Reward: {sum(eval_rewards):.2f}")

            eval_episode = {
                "index": ep,
                "states":  np.array(eval_states),
                "actions": np.array(eval_actions),
                "rewards": np.array(eval_rewards),
            }
            plot_episode_figure(eval_episode, f"on_policy_eval_{ep}",
                                config["observation"]["num_bins"], ckpt_dir)

    # ── Final save & evaluation ───────────────────────────────
    torch.save(agent.actor.state_dict(),
               os.path.join(ckpt_dir, "actor_final.pth"))
    torch.save(agent.critic.state_dict(),
               os.path.join(ckpt_dir, "critic_final.pth"))
    print("Saved final actor & critic.")

    eval_states, eval_actions, eval_rewards = [], [], []
    obs_eval, done = env.reset(), False
    while not done:
        a_eval = agent.select_action(obs_eval, noise_enable=False)
        eval_states.append(obs_eval)
        eval_actions.append(a_eval)
        obs_eval, r, done = env.step(a_eval)
        eval_rewards.append(r)
    print(f"Final evaluation episode total reward: {sum(eval_rewards):.2f}")

    eval_episode = {
        "index": num_episodes,
        "states":  np.array(eval_states),
        "actions": np.array(eval_actions),
        "rewards": np.array(eval_rewards),
    }
    plot_episode_figure(eval_episode, "on_policy_eval_final",
                        config["observation"]["num_bins"], ckpt_dir)

if __name__ == "__main__":
    main()









