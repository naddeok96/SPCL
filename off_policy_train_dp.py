#==========================================
# File: off_policy_train_dp.py
#==========================================


import os
import yaml
import torch
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import trange, tqdm
import time
from typing import List
from torch.nn.parallel import DataParallel

import wandb

from rl_agent import DDPGAgent, _project_mixture
from curriculum_env import CurriculumEnv
from replay_buffer import (
    build_replay_buffer_streaming,
    load_replay_buffer,
    save_replay_buffer,
)
from utils import select_top_percent, behavior_clone



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _parse_gpu_list(s: str | None) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def rollout_ga_candidate(env, candidate_vec, num_phases):
    unit = candidate_vec.numel() // num_phases
    obs = env.reset()
    done = False
    total_r = 0.0
    for p in range(num_phases):
        action = candidate_vec[p * unit:(p + 1) * unit]
        obs, r, done = env.step(action)
        total_r += float(r)
        if done:
            break
    return total_r


# ----- plotting helpers (unchanged) -----
def breakdown_state(state, num_bins):
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32)
    return {
        'easy_correct_hist':   state[0:num_bins].tolist(),
        'easy_incorrect_hist': state[num_bins:2*num_bins].tolist(),
        'medium_correct_hist': state[2*num_bins:3*num_bins].tolist(),
        'medium_incorrect_hist': state[3*num_bins:4*num_bins].tolist(),
        'hard_correct_hist':   state[4*num_bins:5*num_bins].tolist(),
        'hard_incorrect_hist': state[5*num_bins:6*num_bins].tolist(),
        'relative_sizes':      state[6*num_bins:6*num_bins+3].tolist(),
        'extra':               state[6*num_bins+3:6*num_bins+5].tolist(),
    }

def breakdown_action(action):
    if not torch.is_tensor(action):
        action = torch.tensor(action, dtype=torch.float32)
    return {
        'learning_rate': action[0].tolist(),
        'mixing_ratios': action[1:4].tolist(),
        'sample_usage_fraction': action[4].tolist(),
    }

def plot_episode_figure(episode, group_name, num_bins, output_dir):
    min_val, max_val, alpha = 0.0, 13.8, 2.0
    rel = torch.linspace(0, 1, num_bins + 1)
    edges = (min_val + (max_val - min_val) * (rel ** alpha)).tolist()
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    widths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]

    states = episode["states"]; actions = episode["actions"]; rewards = episode["rewards"]
    num_phases = len(states)

    fig = plt.figure(figsize=(20, num_phases * 3 + 3))
    fig.patch.set_facecolor("#fffaf0")
    fig.suptitle("\U0001F389 Fun Episode Analysis \U0001F389", fontsize=18, fontweight="bold", y=0.995)

    gs_top = gridspec.GridSpec(nrows=num_phases, ncols=4, top=0.90, bottom=0.55, wspace=0.4, hspace=0.6)
    gs_bot = gridspec.GridSpec(nrows=1, ncols=4, top=0.50, bottom=0.05, wspace=0.5)

    for i in range(num_phases):
        s = states[i]; r = rewards[i]
        sb = breakdown_state(s, num_bins)

        ax0 = fig.add_subplot(gs_top[i, 0]); ax0.set_facecolor("#f5f5f5")
        ax0.bar(centers, sb["easy_correct_hist"], widths, align="center", color="green", hatch="//", alpha=0.7, label="Correct")
        ax0.bar(centers, sb["easy_incorrect_hist"], widths, align="center", color="red", hatch="xx", alpha=0.7, label="Incorrect")
        if i == 0: ax0.set_title("Easy Loss Hist", fontsize=10, fontweight="bold"); ax0.legend(fontsize=8)
        ax0.set_ylabel(f"P{i+1}\nR:{r:.2f}", fontsize=9); ax0.grid(True, linestyle="--", alpha=0.5)
        ax0.tick_params(axis="both", labelsize=8, rotation=45); ax0.set_xticks(edges)

        ax1 = fig.add_subplot(gs_top[i, 1]); ax1.set_facecolor("#f5f5f5")
        ax1.bar(centers, sb["medium_correct_hist"], widths, color="green", hatch="//", alpha=0.7)
        ax1.bar(centers, sb["medium_incorrect_hist"], widths, color="red", hatch="xx", alpha=0.7)
        if i == 0: ax1.set_title("Medium Loss Hist", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5); ax1.tick_params(axis="both", labelsize=8, rotation=45); ax1.set_xticks(edges)

        ax2 = fig.add_subplot(gs_top[i, 2]); ax2.set_facecolor("#f5f5f5")
        ax2.bar(centers, sb["hard_correct_hist"], widths, color="green", hatch="//", alpha=0.7)
        ax2.bar(centers, sb["hard_incorrect_hist"], widths, color="red", hatch="xx", alpha=0.7)
        if i == 0: ax2.set_title("Hard Loss Hist", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5); ax2.tick_params(axis="both", labelsize=8, rotation=45); ax2.set_xticks(edges)

        ax3 = fig.add_subplot(gs_top[i, 3]); ax3.set_facecolor("#f5f5f5")
        info = sb["relative_sizes"] + sb["extra"]
        ax3.bar(range(5), info, color=["blue", "orange", "purple", "cyan", "magenta"], alpha=0.8)
        if i == 0: ax3.set_title("State Info", fontsize=10, fontweight="bold")
        ax3.set_xticks(range(5)); ax3.set_xticklabels(["Easy", "Med", "Hard", "PhaseRatio", "AvailRatio"], rotation=45, fontsize=8)
        ax3.tick_params(axis="both", labelsize=8); ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.text(2, max(info)*1.05 if info else 0.0, f"R={r:.1f}", ha="center", fontsize=8, color="darkred")

    phases = list(range(1, num_phases + 1))
    lrs, usage, mixrs = [], [], []
    for a in actions:
        if torch.is_tensor(a): a = a.detach().cpu()
        lrs.append(float(a[0])); usage.append(float(a[4])); mixrs.append([float(x) for x in a[1:4]])
    rews = [float(r) for r in rewards]
    # if rews: rews[-1] = rews[-1] / 10.0

    ax_lr = fig.add_subplot(gs_bot[0, 0]); ax_lr.set_facecolor("#f5f5f5")
    ax_lr.plot(phases, lrs, marker="D", linestyle="-", color="blue", markersize=6)
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Phase"); ax_lr.set_ylabel("LR"); ax_lr.set_xticks(phases); ax_lr.grid(True, linestyle=":", alpha=0.6)

    ax_us = fig.add_subplot(gs_bot[0, 1]); ax_us.set_facecolor("#f5f5f5")
    ax_us.plot(phases, usage, marker="D", linestyle="-", color="orange", markersize=6)
    ax_us.set_title("Sample Usage"); ax_us.set_xlabel("Phase"); ax_us.set_ylabel("Usage"); ax_us.set_xticks(phases); ax_us.grid(True, linestyle=":", alpha=0.6)

    ax_mx = fig.add_subplot(gs_bot[0, 2]); ax_mx.set_facecolor("#f5f5f5")
    bar_w = 0.6
    for idx, mr in enumerate(mixrs):
        bottom = 0.0
        ax_mx.bar(idx, mr[0], bottom=bottom, width=bar_w, color="green", label="Easy" if idx == 0 else ""); bottom += mr[0]
        ax_mx.bar(idx, mr[1], bottom=bottom, width=bar_w, color="yellow", label="Med" if idx == 0 else ""); bottom += mr[1]
        ax_mx.bar(idx, mr[2], bottom=bottom, width=bar_w, color="red", label="Hard" if idx == 0 else "")
    ax_mx.set_xticks(range(num_phases)); ax_mx.set_xticklabels([f"P{p}" for p in phases])
    ax_mx.set_title("Mixing Ratios"); ax_mx.set_xlabel("Phase"); ax_mx.set_ylabel("Ratio"); ax_mx.legend(fontsize=8); ax_mx.grid(True, linestyle=":", alpha=0.6)

    ax_rw = fig.add_subplot(gs_bot[0, 3]); ax_rw.set_facecolor("#f5f5f5")
    ax_rw.plot(phases, rews, marker="D", linestyle="-", color="magenta", markersize=6)
    ax_rw.set_title("Reward"); ax_rw.set_xlabel("Phase"); ax_rw.set_ylabel("Reward"); ax_rw.set_xticks(phases); ax_rw.grid(True, linestyle=":", alpha=0.6)

    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.98, hspace=0.6, wspace=0.4)
    fname = os.path.join(output_dir, f"{group_name}_episode_{episode['index']}_detailed.png")
    plt.savefig(fname); plt.close(fig)
    print(f"Saved detailed figure for {group_name} episode {episode['index']} to {fname}")


# ----- Main Training and Evaluation -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="path to your config YAML")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. If unset, runs single-GPU."
    )
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))
    gpu_ids = _parse_gpu_list(args.gpus)
    visible = torch.cuda.device_count()
    if gpu_ids and max(gpu_ids) >= visible:
        raise ValueError(
            f"--gpus {gpu_ids} out of range for visible device count {visible}. "
            "Pass local ids (e.g., with CUDA_VISIBLE_DEVICES=2,3 use --gpus 0,1)."
        )

    if gpu_ids:
        # Use the first requested GPU as primary (no env var mucking; ids are local)
        primary = gpu_ids[0]
        config["device"] = f"cuda:{primary}"
        torch.cuda.set_device(primary)
    else:
        # Fallback: honor YAML unless CUDA is unavailable
        if str(config.get("device", "cuda:0")).startswith("cuda") and not torch.cuda.is_available():
            config["device"] = "cpu"

    torch.backends.cudnn.benchmark = True

    results_dir = os.path.join("results", "off_policy_v6_dp")
    os.makedirs(results_dir, exist_ok=True)

    if wandb is not None:
        wandb_kwargs = {"project": "off_policy_training", "config": config}
        run_name = config.get("run_name")
        if run_name:
            wandb_kwargs["name"] = run_name

        tags = []
        existing_tags = config.get("run_tags")
        if isinstance(existing_tags, (list, tuple)):
            tags.extend(str(t) for t in existing_tags)
        elif isinstance(existing_tags, str):
            tags.append(existing_tags)

        run_tag = config.get("run_tag")
        if run_tag:
            tags.append(str(run_tag))

        if tags:
            wandb_kwargs["tags"] = tags

        wandb.init(**wandb_kwargs)
    else:
        print("wandb not installed; proceeding without online logging")

    # Hourly checkpoint overwrites
    hourly_actor   = os.path.join(results_dir, "off_policy_actor_latest.pth")
    hourly_critic1 = os.path.join(results_dir, "off_policy_critic1_latest.pth")
    hourly_critic2 = os.path.join(results_dir, "off_policy_critic2_latest.pth")
    last_hour_save = time.time()

    # Load dataset to CPU
    dataset_path = config["paths"]["pretrain_path"]
    data = torch.load(dataset_path, map_location="cpu")
    states, actions = data["states"], data["actions"]
    rewards, next_states, dones = data["rewards"], data["next_states"], data["dones"]
    
    # --- Reward normalization: dataset has final phase reward ×10; scale terminals down ---
    if isinstance(dones, torch.Tensor) and dones.dtype == torch.bool:
        rewards = rewards.clone()
        # zero out all non-terminal rewards
        rewards[~dones] = 0.0
        # scale terminal rewards if you want (here ×10)
        rewards[dones] = rewards[dones] / 10.0
    else:
        # if dones is float 0/1
        dones_f = dones.to(torch.float32)
        rewards = rewards.clone()
        rewards = torch.where(dones_f > 0.5, rewards / 10.0, torch.zeros_like(rewards))

    # Project offline dataset actions onto the updated manifold so critics see reachable actions.
    with torch.no_grad():
        projected_actions = actions.clone()
        lr_min, lr_max = config["curriculum"]["learning_rate_range"]
        projected_actions[:, 0:1] = projected_actions[:, 0:1].clamp(lr_min, lr_max)
        projected_actions[:, 1:4] = _project_mixture(
            projected_actions[:, 1:4].clamp(min=0.0),
            eps=float(config["rl"].get("mix_floor", 0.05)),
        )
        projected_actions[:, 4:5] = projected_actions[:, 4:5].clamp(0.0, 1.0)
        actions = projected_actions
    
    dataset = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}
    print(f"Loaded dataset from {dataset_path} with {len(states)} transitions")

    # Environment & agent on user-specified device
    env = CurriculumEnv(config)
    obs_dim = len(env.reset()); action_dim = 5
    agent = DDPGAgent(obs_dim, action_dim, config)

    # Wrap models in DataParallel when multiple GPUs are requested
    def _sd(m):
        return m.module.state_dict() if isinstance(m, DataParallel) else m.state_dict()

    if gpu_ids and len(gpu_ids) > 1:
        agent.actor   = DataParallel(agent.actor,   device_ids=gpu_ids, output_device=gpu_ids[0])
        agent.critic1 = DataParallel(agent.critic1, device_ids=gpu_ids, output_device=gpu_ids[0])
        agent.critic2 = DataParallel(agent.critic2, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"[DP] Using GPUs {gpu_ids} (primary cuda:{gpu_ids[0]})")
    else:
        print(f"[DP] Single device: {config['device']}")

    # --- State normalization (mean/std over the offline set) ---
    with torch.no_grad():
        s_mean = states.mean(dim=0)
        s_std  = states.std(dim=0, unbiased=False).clamp_min(1e-6)
        states = (states - s_mean) / s_std
        next_states = (next_states - s_mean) / s_std
        
    def _norm_obs(x):
        return (x - s_mean.to(x.device)) / s_std.to(x.device)

    dataset = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}

    # --- Behavior Cloning ---
    try:
        expert = select_top_percent(dataset, percent=float(config["rl"].get("bc_top_percent", 20)))
        behavior_clone(agent.actor, expert, epochs=int(config["rl"].get("bc_epochs", 5)),
                        batch_size=int(config["rl"].get("bc_batch_size", 256)), device=agent.device)
        agent._hard_update(agent.actor_target, agent.actor)
        print("Behavior cloning warm-start complete.")
    except Exception as e:
        print(f"BC warm-start skipped due to error: {e}")

    # --------- Build a CPU-resident PER buffer via streaming ingestion ----------
    shard_size = int(config["rl"].get("ingest_shard_size", 200_000))
    cache_path = config.get("paths", {}).get("replay_cache_path")
    use_cache = bool(cache_path)
    replay_buffer = None

    # Normalize cache path to repo-relative path if provided
    if use_cache:
        cache_path = os.path.expanduser(cache_path)
        cache_path = os.path.abspath(cache_path)

    per_enabled = bool(config["rl"].get("per_enabled", True))
    per_alpha = float(config["rl"].get("per_alpha", 0.4))
    per_beta = float(config["rl"].get("per_beta", 0.6))
    per_epsilon = float(config["rl"].get("per_epsilon", 1e-6))
    per_type = str(config["rl"].get("per_type", "proportional"))
    buffer_capacity = int(config["rl"]["buffer_size"])

    def _cache_matches(meta: dict, expected: dict) -> bool:
        for key, value in expected.items():
            if meta.get(key) != value:
                return False
        return True

    dataset_mtime = None
    try:
        dataset_mtime = os.path.getmtime(dataset_path)
    except (OSError, FileNotFoundError):
        dataset_mtime = None

    expected_meta = {
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_mtime": dataset_mtime,
        "dataset_length": int(len(states)),
        "buffer_capacity": buffer_capacity,
        "per_enabled": per_enabled,
        "per_alpha": per_alpha,
        "per_beta": per_beta,
        "per_epsilon": per_epsilon,
        "per_type": per_type,
        "shard_size": shard_size,
    }

    if use_cache and os.path.exists(cache_path):
        try:
            cached_buffer, cached_meta = load_replay_buffer(cache_path)
            if _cache_matches(cached_meta, expected_meta):
                replay_buffer = cached_buffer
                print(f"Loaded replay buffer cache from {cache_path}")
            else:
                print("Replay buffer cache metadata mismatch; rebuilding cache.")
        except Exception as cache_err:
            print(f"Failed to load replay buffer cache ({cache_err}); rebuilding.")

    if replay_buffer is None:
        replay_buffer = build_replay_buffer_streaming(
            dataset,
            capacity=buffer_capacity,          # you can lower this in YAML if RAM is tight
            use_per=per_enabled,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            per_type=per_type,
            shard_size=shard_size,
            show_progress=True,
            progress_desc="Initializing replay buffer",
        )
        if use_cache:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            try:
                save_replay_buffer(replay_buffer, cache_path, expected_meta)
                print(f"Saved replay buffer cache to {cache_path}")
            except Exception as cache_save_err:
                print(f"Warning: failed to save replay buffer cache ({cache_save_err})")

    if per_enabled and hasattr(replay_buffer, "set_beta"):
        replay_buffer.set_beta(per_beta)

    print(f"Replay buffer initialized with {len(replay_buffer)} transitions (capacity {config['rl']['buffer_size']}).")

    # Probe states for action variance tracking (on agent device)
    probe_batch_size = 256
    perm = torch.randperm(len(states))
    idxs = perm[:probe_batch_size]
    variance_states_tensor = states[idxs].to(agent.device).float()

    # Training schedule
    num_updates = int(config["rl"].get("off_policy_updates", 1_000_000))
    default_eval_interval = max(1, num_updates // 200)
    evaluation_interval  = max(1, int(config["rl"].get("eval_every_updates", default_eval_interval)))
    checkpoint_interval  = max(1, num_updates // 200)   # save every 5%

    print(f"Starting off-policy training for {num_updates} updates")

    # Logs
    actor_losses, critic1_losses, critic2_losses = [], [], []
    actor_lrs, critic1_lrs, critic2_lrs = [], [], []
    reward_progress, reward_stds, eval_updates = [], [], []
    action_var_history, update_steps = [], []
    num_bins = int(config["observation"]["num_bins"])

    best_mean_reward = -float("inf")
    no_improve = 0
    early_stop = config["rl"].get("early_stop_patience")
    last_actor_loss = 0.0
    num_eval_eps = int(config["rl"].get("num_eval_episodes", 1))
    stop_training = False

    for update in trange(num_updates, desc="Off-policy Training"):
        if len(replay_buffer) >= int(config["rl"]["batch_size"]):
            metrics = agent.update(replay_buffer, int(config["rl"]["batch_size"]))
            if metrics is not None:
                if metrics["actor_loss"] is not None:
                    last_actor_loss = metrics["actor_loss"]
                actor_losses.append(last_actor_loss)
                critic1_losses.append(metrics["critic1_loss"])
                critic2_losses.append(metrics["critic2_loss"])
                actor_lrs.append(agent.actor_optimizer.param_groups[0]["lr"])
                critic1_lrs.append(agent.critic1_optimizer.param_groups[0]["lr"])
                critic2_lrs.append(agent.critic2_optimizer.param_groups[0]["lr"])
                if wandb is not None:
                    wandb.log({
                        "actor_loss": last_actor_loss,
                        "critic1_loss": metrics["critic1_loss"],
                        "critic2_loss": metrics["critic2_loss"],
                        "actor_lr": actor_lrs[-1],
                        "critic1_lr": critic1_lrs[-1],
                        "critic2_lr": critic2_lrs[-1],
                        "ood_l2": metrics.get("ood_l2", None),
                        "update": update,
                    })
                else:
                    if metrics.get("ood_l2") is not None:
                        print(f"Update {update}: ood_l2={metrics['ood_l2']:.4f}")

        with torch.no_grad():
            pred_actions = agent.actor(variance_states_tensor).cpu()
        var = torch.var(pred_actions, dim=0, unbiased=False)
        action_var_history.append(var.tolist()); update_steps.append(update)

        # Hourly rolling checkpoint
        if time.time() - last_hour_save >= 3600:
            torch.save(_sd(agent.actor), hourly_actor)
            torch.save(_sd(agent.critic1), hourly_critic1)
            torch.save(_sd(agent.critic2), hourly_critic2)
            print(f"Hourly checkpoint saved at update {update}")
            last_hour_save = time.time()

        # Periodic evaluation + plots
        if update % evaluation_interval == 0:
            rewards_this_ckpt = []
            first_episode = None
            for ep_i in range(num_eval_eps):
                eval_states, eval_actions, eval_rewards = [], [], []
                obs_eval = env.reset(); done = False
                obs_eval = _norm_obs(obs_eval)
                agent.ou_noise.reset() 
                with tqdm(total=env.max_phases, desc=f"Eval {update} Ep{ep_i}", leave=False) as pbar:
                    while not done:
                        action_eval = agent.select_action(obs_eval, noise_enable=False)
                        eval_states.append(obs_eval); eval_actions.append(action_eval)
                        obs_eval, reward, done = env.step(action_eval)
                        obs_eval = _norm_obs(obs_eval)
                        eval_rewards.append(reward); pbar.update(1)
                total_reward = float(sum(eval_rewards))
                rewards_this_ckpt.append(total_reward)
                if ep_i == 0:
                    first_episode = {
                        "index": update,
                        "states": eval_states,
                        "actions": eval_actions,
                        "rewards": eval_rewards,
                        "total_reward": total_reward,
                        "episode_length": len(eval_states),
                    }

            mean_reward = float(np.mean(rewards_this_ckpt))
            std_reward  = float(np.std(rewards_this_ckpt))
            reward_progress.append(mean_reward); reward_stds.append(std_reward); eval_updates.append(update)
            print(f"Checkpoint at update {update}/{num_updates} - Eval reward {mean_reward:.2f} ± {std_reward:.2f}")
            with torch.no_grad():
                a_eval = agent.actor(variance_states_tensor).cpu()
                mix = a_eval[:, 1:4].clamp(min=1e-8)
                mix_H = (-(mix * mix.log()).sum(dim=1)).mean().item()
                mix_min = mix.min().item()
                mix_max = mix.max().item()
                usage_mean = a_eval[:, 4].mean().item()
            if wandb is not None:
                wandb.log({"eval_mean_reward": mean_reward, "eval_std_reward": std_reward, "update": update})
                wandb.log({"mix_entropy": mix_H, "mix_min": mix_min, "mix_max": mix_max, "usage_mean": usage_mean})

            # Optional: compare vs GA elites if provided
            ga_paths = config.get("compare_models", {}).get("GA_Elites", None)
            if ga_paths:
                try:
                    ga = torch.load(ga_paths, map_location="cpu")
                    elites = ga.get("population")
                    num_phases = int(config["curriculum"].get("max_phases", 3))
                    scores = []
                    for i in range(min(3, elites.size(0))):
                        env_comp = CurriculumEnv(config)
                        scores.append(rollout_ga_candidate(env_comp, elites[i], num_phases))
                    ga_mean = float(np.mean(scores))
                    if wandb is not None:
                        wandb.log({"ga_elite_mean_reward": ga_mean, "update": update})
                    print(f"GA elite mean reward (sample): {ga_mean:.2f}")
                except Exception as e:
                    print(f"GA compare skipped: {e}")

            eval_episode = first_episode

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(_sd(agent.actor),  os.path.join(results_dir, "best_actor.pth"))
                torch.save(_sd(agent.critic1), os.path.join(results_dir, "best_critic1.pth"))
                torch.save(_sd(agent.critic2), os.path.join(results_dir, "best_critic2.pth"))
                no_improve = 0
            else:
                no_improve += 1

            if early_stop is not None and no_improve >= int(early_stop):
                print(f"Early stopping at update {update} due to no improvement")
                stop_training = True
                break

            # Plots
            plot_episode_figure(eval_episode, f"eval_{update}", num_bins, results_dir)

            fig_loss, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(8, 10))
            ax_actor.plot(actor_losses, color='blue'); ax_actor.set_title("Actor Loss Progression")
            ax_actor.set_xlabel("Update Steps"); ax_actor.set_ylabel("Loss")
            ax_critic.plot(critic1_losses, label="Critic1")
            ax_critic.plot(critic2_losses, label="Critic2"); ax_critic.legend()
            ax_critic.set_title("Critic Loss Progression"); ax_critic.set_xlabel("Update Steps"); ax_critic.set_ylabel("Loss")
            fig_loss.tight_layout(); fig_loss.savefig(os.path.join(results_dir, f"training_losses_{update}.png")); plt.close(fig_loss)

            plt.figure(); plt.errorbar(eval_updates, reward_progress, yerr=reward_stds, marker='o', capsize=3)
            plt.xlabel("Update Steps"); plt.ylabel("Reward"); plt.title("Periodic Reward Evaluation")
            plt.savefig(os.path.join(results_dir, f"reward_progress_{update}.png")); plt.close()

            fig_lr = plt.figure()
            plt.plot(actor_lrs, label='Actor'); plt.plot(critic1_lrs, label='Critic1'); plt.plot(critic2_lrs, label='Critic2')
            plt.xlabel('Update Steps'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Progression'); plt.legend()
            fig_lr.tight_layout(); fig_lr.savefig(os.path.join(results_dir, f"learning_rates_{update}.png")); plt.close(fig_lr)

            if update % checkpoint_interval == 0:
                torch.save(_sd(agent.actor),  os.path.join(results_dir, f"off_policy_actor_{update}.pth"))
                torch.save(_sd(agent.critic1), os.path.join(results_dir, f"off_policy_critic1_{update}.pth"))
                torch.save(_sd(agent.critic2), os.path.join(results_dir, f"off_policy_critic2_{update}.pth"))

            action_var_array = torch.tensor(action_var_history)
            names = ["learning_rate","mix_easy","mix_med","mix_hard","sample_usage"]
            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8,12), sharex=True)
            for idx, name in enumerate(names):
                axs[idx].plot(update_steps, action_var_array[:, idx].numpy())
                axs[idx].set_ylabel("Var"); axs[idx].set_title(name); axs[idx].grid(True)
            axs[-1].set_xlabel("Update Step")
            plt.tight_layout(); plt.savefig(os.path.join(results_dir, "action_variance.png")); plt.close()

        if stop_training:
            break

    # Final full evaluation
    eval_states, eval_actions, eval_rewards = [], [], []
    obs_eval = env.reset(); done = False
    obs_eval = _norm_obs(obs_eval)
    agent.ou_noise.reset() 
    with tqdm(total=env.max_phases, desc="Final Eval", leave=False) as pbar:
        while not done:
            action_eval = agent.select_action(obs_eval, noise_enable=False)
            eval_states.append(obs_eval); eval_actions.append(action_eval)
            obs_eval, reward, done = env.step(action_eval)
            obs_eval = _norm_obs(obs_eval)
            eval_rewards.append(reward); pbar.update(1)
    total_reward = float(sum(eval_rewards))
    print(f"Final evaluation episode total reward: {total_reward}")

    final_index = update if stop_training else num_updates
    eval_episode = {
        "index": final_index, "states": eval_states, "actions": eval_actions, "rewards": eval_rewards,
        "total_reward": total_reward, "episode_length": len(eval_states)
    }
    plot_episode_figure(eval_episode, "final_eval", num_bins, results_dir)

    # Save finals
    final_actor   = os.path.join(results_dir, "off_policy_actor_model_final.pth")
    final_critic1 = os.path.join(results_dir, "off_policy_critic1_model_final.pth")
    final_critic2 = os.path.join(results_dir, "off_policy_critic2_model_final.pth")
    torch.save(_sd(agent.actor), final_actor)
    torch.save(_sd(agent.critic1), final_critic1)
    torch.save(_sd(agent.critic2), final_critic2)
    print(f"Saved final actor → {final_actor}")
    print(f"Saved final critic 1 → {final_critic1}")
    print(f"Saved final critic 2 → {final_critic2}")

    # Final plots
    fig_loss, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(8, 10))
    ax_actor.plot(actor_losses, color='blue'); ax_actor.set_title("Actor Loss Progression")
    ax_actor.set_xlabel("Update Steps"); ax_actor.set_ylabel("Loss")
    ax_critic.plot(critic1_losses, color='red', label="Critic 1")
    ax_critic.plot(critic2_losses, color='green', label="Critic 2"); ax_critic.legend()
    ax_critic.set_title("Critic Loss Progression"); ax_critic.set_xlabel("Update Steps"); ax_critic.set_ylabel("Loss")
    fig_loss.tight_layout(); fig_loss.savefig(os.path.join(results_dir, "training_losses_final.png")); plt.close(fig_loss)

    plt.figure(); plt.errorbar(eval_updates, reward_progress, yerr=reward_stds, marker='o', capsize=3)
    plt.xlabel("Update Steps"); plt.ylabel("Reward"); plt.title("Periodic Reward Evaluation")
    plt.savefig(os.path.join(results_dir, "reward_progress_final.png")); plt.close()

    plt.figure()
    plt.plot(actor_lrs, label='Actor'); plt.plot(critic1_lrs, label='Critic1'); plt.plot(critic2_lrs, label='Critic2')
    plt.xlabel('Update Steps'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Progression'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(results_dir, 'learning_rates_final.png')); plt.close()

    if wandb is not None:
        wandb.log({"final_total_reward": total_reward})
        wandb.finish()


if __name__ == "__main__":
    main()
