#!/usr/bin/env python3
"""
analyze_history.py

This script loads a PyTorch .pt file (saved by generate_dataset.py) that contains training history 
(transitions from an evolutionary dataset), groups transitions into episodes using the 
done flag as an episode terminator, and then analyzes the episodes. The script:
  - Computes the total reward per episode.
  - Selects 5 episodes with the lowest total rewards, 5 episodes around the median, 
    and 5 episodes with the highest total rewards.
  - For each selected episode, it produces a text summary and also generates a detailed figure:
      * TOP BLOCK: For each phase (transition) in the episode, a row of 4 subplots:
            - Column 0: Combined "Easy" loss histogram (green for correct, red for incorrect).
            - Column 1: Combined "Medium" loss histogram.
            - Column 2: Combined "Hard" loss histogram.
            - Column 3: State Info bar chart (relative sizes and extra features).
         Each row is annotated with that phase’s reward.
      * BOTTOM BLOCK: Aggregated evolution across phases with 4 subplots:
            1. Learning Rate evolution.
            2. Sample Usage evolution.
            3. Mixing Ratios as a stacked bar plot (one bar per phase).
            4. Reward evolution (with the last phase reward divided by 10).
  - Also plots and saves a histogram of all episode total rewards.

Usage:
    python analyze_history.py \
        --pt_file evolutionary_dataset.pt \
        [--num_bins 64] \
        [--output_dir history]

If no arguments are given, it defaults to:
    pt_file: results/curriculum_rl/evolutionary_dataset.pt
    num_bins: 16
    output_dir: history
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_data(pt_file: str):
    """Load the saved transitions from a .pt file."""
    data = torch.load(pt_file, map_location='cpu')
    return data['states'], data['actions'], data['rewards'], data['dones']

def group_episodes_by_done(states, actions, rewards, dones):
    """
    Groups transitions into episodes by accumulating consecutive transitions
    until a transition with done == True is encountered.
    Each episode dictionary now includes an 'index' field.
    """
    episodes = []
    current_states = []
    current_actions = []
    current_rewards = []
    episode_counter = 0
    for s, a, r, d in zip(states, actions, rewards, dones):
        current_states.append(s)
        current_actions.append(a)
        current_rewards.append(r)
        if d:
            aggregated_reward = sum(float(x) for x in current_rewards)
            episodes.append({
                'index': episode_counter,
                'states': current_states.copy(),
                'actions': current_actions.copy(),
                'rewards': current_rewards.copy(),
                'total_reward': aggregated_reward,
                'episode_length': len(current_states)
            })
            episode_counter += 1
            current_states = []
            current_actions = []
            current_rewards = []
    # Group any remaining transitions as an incomplete episode.
    if current_states:
        aggregated_reward = sum(float(x) for x in current_rewards)
        episodes.append({
            'index': episode_counter,
            'states': current_states.copy(),
            'actions': current_actions.copy(),
            'rewards': current_rewards.copy(),
            'total_reward': aggregated_reward,
            'episode_length': len(current_states)
        })
    return episodes

def select_episode_groups(episodes):
    """
    Sorts episodes by total reward and selects:
       - 5 episodes with the lowest total rewards,
       - 5 episodes around the median,
       - 5 episodes with the highest total rewards.
    """
    sorted_eps = sorted(episodes, key=lambda ep: ep['total_reward'])
    num_eps = len(sorted_eps)
    low = sorted_eps[:5] if num_eps >= 5 else sorted_eps
    high = sorted_eps[-5:] if num_eps >= 5 else sorted_eps
    median_index = num_eps // 2
    start = max(0, median_index - 2)
    end = start + 5
    if end > num_eps:
        end = num_eps
        start = max(0, end - 5)
    median = sorted_eps[start:end]
    return low, median, high

def breakdown_state(state, num_bins):
    """
    Breaks down the state vector into its constituent parts.
    Assumes the following layout:
      - Indices 0:num_bins                  : Easy Correct Histogram
      - Indices num_bins:2*num_bins         : Easy Incorrect Histogram
      - Indices 2*num_bins:3*num_bins       : Medium Correct Histogram
      - Indices 3*num_bins:4*num_bins       : Medium Incorrect Histogram
      - Indices 4*num_bins:5*num_bins       : Hard Correct Histogram
      - Indices 5*num_bins:6*num_bins       : Hard Incorrect Histogram
      - Indices 6*num_bins:6*num_bins+3     : Relative dataset sizes (3 values)
      - Indices 6*num_bins+3:6*num_bins+5   : Extra state features (2 values)
    Total length = 6*num_bins + 5.
    """
    breakdown = {}
    breakdown['easy_correct_hist']    = state[0:num_bins]
    breakdown['easy_incorrect_hist']  = state[num_bins:2*num_bins]
    breakdown['medium_correct_hist']  = state[2*num_bins:3*num_bins]
    breakdown['medium_incorrect_hist'] = state[3*num_bins:4*num_bins]
    breakdown['hard_correct_hist']     = state[4*num_bins:5*num_bins]
    breakdown['hard_incorrect_hist']   = state[5*num_bins:6*num_bins]
    breakdown['relative_sizes']       = state[6*num_bins:6*num_bins+3]
    breakdown['extra']                = state[6*num_bins+3:6*num_bins+5]
    return breakdown

def breakdown_action(action):
    """
    Breaks down the 5-dimensional action vector into:
      - Learning rate: action[0]
      - Mixing ratios for (Easy, Medium, Hard): action[1:4]
      - Sample usage fraction: action[4]
    """
    return {
        'learning_rate': action[0],
        'mixing_ratios': action[1:4],
        'sample_usage_fraction': action[4],
    }

def save_episode_details(episodes, group_name, num_bins, output_dir):
    """
    Saves a text file summarizing each episode’s detailed breakdown for a specified group.
    """
    filename = os.path.join(output_dir, f"{group_name}_episodes.txt")
    with open(filename, "w") as f:
        f.write(f"--- {group_name.capitalize()} Episodes Analysis ---\n\n")
        for ep in episodes:
            f.write(f"Episode {ep['index']} - Total Reward: {ep['total_reward']}, Length: {ep['episode_length']}\n")
            for i, (state, action, reward) in enumerate(zip(ep['states'], ep['actions'], ep['rewards'])):
                f.write(f"  Phase {i+1} (Reward: {reward}):\n")
                sb = breakdown_state(state, num_bins)
                ab = breakdown_action(action)
                f.write("    State breakdown:\n")
                for key, value in sb.items():
                    f.write(f"      {key}: {value}\n")
                f.write("    Action breakdown:\n")
                f.write(f"      Learning Rate: {ab['learning_rate']:.4f}\n")
                f.write(f"      Mixing Ratios: {ab['mixing_ratios']}\n")
                f.write(f"      Sample Usage Fraction: {ab['sample_usage_fraction']:.4f}\n")
            f.write("\n")
    print(f"Saved {group_name} episodes analysis to {filename}")

def plot_reward_distribution(episodes, output_dir):
    """Plots and saves a histogram of total rewards for all episodes."""
    rewards = [ep['total_reward'] for ep in episodes]
    plt.figure()
    plt.hist(rewards, bins=20, edgecolor='black', color='skyblue')
    plt.title("Episode Total Reward Distribution")
    plt.xlabel("Total Reward")
    plt.ylabel("Number of Episodes")
    plot_path = os.path.join(output_dir, "episode_reward_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved reward distribution plot to {plot_path}")

def plot_episode_figure(episode, group_name, num_bins, output_dir):
    """
    For a given episode, creates a detailed figure with playful styling:
      - Adds a fun emoji title and light background.
      - TOP BLOCK: one row per phase:
          * Columns 0–2: loss histograms with original colors (green/red), plus hatch patterns.
          * Column 3: state info bar chart with original 'colors_info'.
          * Grids, light axis background, and phase annotations keep it lively.
      - BOTTOM BLOCK: evolution plots with:
          1. LR (blue, diamond markers) +
             rocket 🚀 annotation at the max point.
          2. Sample usage (orange, diamond markers).
          3. Mixing ratios (stacked bars in original green/yellow/red).
          4. Reward (magenta, diamond markers) +
             star ★ annotation at the final point.
    """
    import torch
    # Recompute bin edges
    min_val, max_val, alpha = 0.0, 13.8, 2.0
    rel = torch.linspace(0, 1, num_bins + 1)
    edges = (min_val + (max_val - min_val) * (rel ** alpha)).tolist()
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    widths  = [edges[i+1] - edges[i]       for i in range(len(edges)-1)]

    num_phases = len(episode['states'])
    fig = plt.figure(figsize=(20, num_phases * 3 + 3))
    fig.patch.set_facecolor('#fffaf0')
    fig.suptitle("🎉 Fun Episode Analysis 🎉", fontsize=18, fontweight='bold', y=0.995)

    gs_top = gridspec.GridSpec(
        nrows=num_phases, ncols=4,
        top=0.90, bottom=0.55, wspace=0.4, hspace=0.6,
        height_ratios=[1] * num_phases
    )
    gs_bot = gridspec.GridSpec(
        nrows=1, ncols=4,
        top=0.50, bottom=0.05, wspace=0.5
    )

    # --- TOP BLOCK ---
    for i in range(num_phases):
        s  = episode['states'][i]
        r  = episode['rewards'][i]
        sb = breakdown_state(s, num_bins)

        # Easy losses
        ax0 = fig.add_subplot(gs_top[i, 0])
        ax0.set_facecolor('#f5f5f5')
        ax0.bar(centers, sb['easy_correct_hist'],   widths,
                align='center', color='green', hatch='//', alpha=0.7, label='Correct')
        ax0.bar(centers, sb['easy_incorrect_hist'], widths,
                align='center', color='red',   hatch='xx', alpha=0.7, label='Incorrect')
        if i == 0:
            ax0.set_title("Easy Loss Hist", fontsize=10, fontweight='bold')
            ax0.legend(fontsize=8)
        ax0.set_ylabel(f"P{i+1}\nR:{r:.2f}", fontsize=9)
        ax0.grid(True, linestyle='--', alpha=0.5)
        ax0.tick_params(axis='both', labelsize=8, rotation=45)
        ax0.set_xticks(edges)

        # Medium losses
        ax1 = fig.add_subplot(gs_top[i, 1])
        ax1.set_facecolor('#f5f5f5')
        ax1.bar(centers, sb['medium_correct_hist'],   widths, color='green', hatch='//', alpha=0.7)
        ax1.bar(centers, sb['medium_incorrect_hist'], widths, color='red',   hatch='xx', alpha=0.7)
        if i == 0: ax1.set_title("Medium Loss Hist", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.tick_params(axis='both', labelsize=8, rotation=45)
        ax1.set_xticks(edges)

        # Hard losses
        ax2 = fig.add_subplot(gs_top[i, 2])
        ax2.set_facecolor('#f5f5f5')
        ax2.bar(centers, sb['hard_correct_hist'],   widths, color='green', hatch='//', alpha=0.7)
        ax2.bar(centers, sb['hard_incorrect_hist'], widths, color='red',   hatch='xx', alpha=0.7)
        if i == 0: ax2.set_title("Hard Loss Hist", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.tick_params(axis='both', labelsize=8, rotation=45)
        ax2.set_xticks(edges)

        # State Info
        ax3 = fig.add_subplot(gs_top[i, 3])
        ax3.set_facecolor('#f5f5f5')
        info = sb['relative_sizes'].tolist() + sb['extra'].tolist()
        colors_info = ['blue','orange','purple','cyan','magenta']
        ax3.bar(range(5), info, color=colors_info, alpha=0.8)
        if i == 0: ax3.set_title("State Info", fontsize=10, fontweight='bold')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Easy','Med','Hard','PhaseRatio','AvailRatio'], rotation=45, fontsize=8)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.text(2, max(info)*1.05, f"R={r:.1f}", ha='center', fontsize=8, color='darkred')

    # --- BOTTOM BLOCK ---
    phases = list(range(1, num_phases + 1))
    lrs    = [episode['actions'][i][0] for i in range(num_phases)]
    usage  = [episode['actions'][i][4] for i in range(num_phases)]
    mixrs  = [episode['actions'][i][1:4] for i in range(num_phases)]
    rews   = [episode['rewards'][i]        for i in range(num_phases)]
    if rews: rews[-1] /= 10.0

    # Learning rate
    ax_lr = fig.add_subplot(gs_bot[0, 0])
    ax_lr.set_facecolor('#f5f5f5')
    ax_lr.plot(phases, lrs, marker='D', linestyle='-', color='blue', markersize=6)
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Phase"); ax_lr.set_ylabel("LR")
    ax_lr.set_xticks(phases); ax_lr.grid(True, linestyle=':', alpha=0.6)

    # Sample usage
    ax_us = fig.add_subplot(gs_bot[0, 1])
    ax_us.set_facecolor('#f5f5f5')
    ax_us.plot(phases, usage, marker='D', linestyle='-', color='orange', markersize=6)
    ax_us.set_title("Sample Usage"); ax_us.set_xlabel("Phase"); ax_us.set_ylabel("Usage")
    ax_us.set_xticks(phases); ax_us.grid(True, linestyle=':', alpha=0.6)

    # Mixing ratios
    ax_mx = fig.add_subplot(gs_bot[0, 2])
    ax_mx.set_facecolor('#f5f5f5')
    bar_w = 0.6
    for idx, mr in enumerate(mixrs):
        bottom = 0.0
        ax_mx.bar(idx, mr[0], bottom=bottom, width=bar_w, color='green',  label='Easy'  if idx==0 else "")
        bottom += mr[0]
        ax_mx.bar(idx, mr[1], bottom=bottom, width=bar_w, color='yellow', label='Med'   if idx==0 else "")
        bottom += mr[1]
        ax_mx.bar(idx, mr[2], bottom=bottom, width=bar_w, color='red',    label='Hard'  if idx==0 else "")
    ax_mx.set_xticks(range(num_phases))
    ax_mx.set_xticklabels([f"P{p}" for p in phases])
    ax_mx.set_title("Mixing Ratios"); ax_mx.set_xlabel("Phase"); ax_mx.set_ylabel("Ratio")
    ax_mx.legend(fontsize=8); ax_mx.grid(True, linestyle=':', alpha=0.6)

    # Reward evolution
    ax_rw = fig.add_subplot(gs_bot[0, 3])
    ax_rw.set_facecolor('#f5f5f5')
    ax_rw.plot(phases, rews, marker='D', linestyle='-', color='magenta', markersize=6)
    ax_rw.set_title("Reward"); ax_rw.set_xlabel("Phase"); ax_rw.set_ylabel("Reward")
    ax_rw.set_xticks(phases); ax_rw.grid(True, linestyle=':', alpha=0.6)

    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.98, hspace=0.6, wspace=0.4)

    out_f = os.path.join(output_dir, f"{group_name}_episode_{episode['index']}_detailed_fun.png")
    plt.savefig(out_f)
    plt.close(fig)
    print(f"Saved fun detailed figure for {group_name} episode {episode['index']} to {out_f}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze training history from pt file with detailed figures."
    )
    parser.add_argument(
        "--pt_file", type=str,
        default="results/curriculum_rl/evolutionary_dataset.pt",
        help="Path to the pt file saved by generate_dataset.py"
    )
    parser.add_argument(
        "--num_bins", type=int, default=16,
        help="Number of bins used in the loss histograms in the state vector"
    )
    parser.add_argument(
        "--output_dir", type=str, default="history",
        help="Directory to save analysis outputs (default: history)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data.
    states, actions, rewards, dones = load_data(args.pt_file)
    print(f"Loaded {states.shape[0]} transitions from {args.pt_file}")

    # Group transitions into episodes using the done flag.
    episodes = group_episodes_by_done(states, actions, rewards, dones)
    print(f"Grouped into {len(episodes)} episodes based on done flags.")

    # Select low, median, and high groups.
    low_eps, median_eps, high_eps = select_episode_groups(episodes)

    # Save text summaries.
    save_episode_details(low_eps, "low", args.num_bins, output_dir)
    save_episode_details(median_eps, "median", args.num_bins, output_dir)
    save_episode_details(high_eps, "high", args.num_bins, output_dir)

    # Plot reward distribution for all episodes.
    plot_reward_distribution(episodes, output_dir)

    # Generate detailed figures for each chosen episode.
    for group_name, group_eps in zip(["low", "median", "high"], [low_eps, median_eps, high_eps]):
        for ep in group_eps:
            plot_episode_figure(ep, group_name, args.num_bins, output_dir)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
