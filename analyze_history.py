#!/usr/bin/env python3
"""
analyze_history.py

This script loads an NPZ file (saved by generate_dataset.py) that contains training history 
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
    python analyze_history.py --npz_file evolutionary_dataset.npz [--num_bins 64]

If no arguments are given, it defaults to:
    npz_file: evolutionary_dataset.npz
    num_bins: 64
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_data(npz_file):
    data = np.load(npz_file)
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    return states, actions, rewards, dones

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
            aggregated_reward = np.sum(current_rewards)
            episodes.append({
                'index': episode_counter,
                'states': np.array(current_states),
                'actions': np.array(current_actions),
                'rewards': np.array(current_rewards),
                'total_reward': aggregated_reward,
                'episode_length': len(current_states)
            })
            episode_counter += 1
            current_states = []
            current_actions = []
            current_rewards = []
    # Group any remaining transitions as an incomplete episode.
    if current_states:
        aggregated_reward = np.sum(current_rewards)
        episodes.append({
            'index': episode_counter,
            'states': np.array(current_states),
            'actions': np.array(current_actions),
            'rewards': np.array(current_rewards),
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
      - Indices 6*num_bins+3:6*num_bins+5     : Extra state features (2 values)
    Total length = 6*num_bins + 5.
    """
    breakdown = {}
    breakdown['easy_correct_hist'] = state[0:num_bins]
    breakdown['easy_incorrect_hist'] = state[num_bins:2*num_bins]
    breakdown['medium_correct_hist'] = state[2*num_bins:3*num_bins]
    breakdown['medium_incorrect_hist'] = state[3*num_bins:4*num_bins]
    breakdown['hard_correct_hist'] = state[4*num_bins:5*num_bins]
    breakdown['hard_incorrect_hist'] = state[5*num_bins:6*num_bins]
    breakdown['relative_sizes'] = state[6*num_bins:6*num_bins+3]
    breakdown['extra'] = state[6*num_bins+3:6*num_bins+5]
    return breakdown

def breakdown_action(action):
    """
    Breaks down the 5-dimensional action vector into:
      - Learning rate: action[0]
      - Mixing ratios for (Easy, Medium, Hard): action[1:4]
      - Sample usage fraction: action[4]
    """
    breakdown = {}
    breakdown['learning_rate'] = action[0]
    breakdown['mixing_ratios'] = action[1:4]
    breakdown['sample_usage_fraction'] = action[4]
    return breakdown

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
                state_breakdown = breakdown_state(state, num_bins)
                action_breakdown = breakdown_action(action)
                f.write("    State breakdown:\n")
                for key, value in state_breakdown.items():
                    f.write(f"      {key}: {np.array2string(value, precision=4, separator=',')}\n")
                f.write("    Action breakdown:\n")
                f.write(f"      Learning Rate: {action_breakdown['learning_rate']:.4f}\n")
                f.write(f"      Mixing Ratios: {np.array2string(action_breakdown['mixing_ratios'], precision=4, separator=',')}\n")
                f.write(f"      Sample Usage Fraction: {action_breakdown['sample_usage_fraction']:.4f}\n")
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
    For a given episode, creates a detailed figure with:
      - TOP BLOCK: For each phase (transition), a row of 4 subplots:
            * Column 0: Combined "Easy" loss histogram (green for correct, red for incorrect).
            * Column 1: Combined "Medium" loss histogram.
            * Column 2: Combined "Hard" loss histogram.
            * Column 3: State Info bar chart (relative sizes and extra features).
         Each row is annotated with the phase reward.
      - BOTTOM BLOCK: Aggregated evolution across phases with 4 subplots:
            1. Learning Rate evolution.
            2. Sample Usage evolution.
            3. Mixing Ratios as a stacked bar plot (one bar per phase).
            4. Reward evolution (with the last phase reward divided by 10).
    The resulting figure is saved to the output directory.
    """
    num_phases = len(episode['states'])
    
    # Create the figure with uniform row height for the top block.
    fig = plt.figure(figsize=(20, num_phases * 3 + 3))
    
    # TOP BLOCK: create a grid with uniform height ratios.
    gs_top = gridspec.GridSpec(nrows=num_phases, ncols=4, top=0.95, bottom=0.55, 
                               wspace=0.4, hspace=0.6, height_ratios=[1] * num_phases)
    
    # BOTTOM BLOCK: 1 row, 4 columns.
    gs_bot = gridspec.GridSpec(nrows=1, ncols=4, top=0.5, bottom=0.05, wspace=0.5)
    
    # TOP BLOCK: For each phase, plot the state breakdown.
    for i in range(num_phases):
        state = episode['states'][i]
        reward_phase = episode['rewards'][i]
        s_break = breakdown_state(state, num_bins)
        x = np.arange(num_bins)
        width = 0.4
        
        # Column 0: Easy losses.
        ax_easy = fig.add_subplot(gs_top[i, 0])
        ax_easy.bar(x - width/2, s_break['easy_correct_hist'], width, color='green', label='Correct')
        ax_easy.bar(x + width/2, s_break['easy_incorrect_hist'], width, color='red', label='Incorrect')
        if i == 0:
            ax_easy.set_title("Easy Loss Hist", fontsize=10)
        ax_easy.set_ylabel(f"P{i+1}\nR: {reward_phase:.2f}", fontsize=9)
        ax_easy.tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_easy.legend(fontsize=8)
        
        # Column 1: Medium losses.
        ax_med = fig.add_subplot(gs_top[i, 1])
        ax_med.bar(x - width/2, s_break['medium_correct_hist'], width, color='green')
        ax_med.bar(x + width/2, s_break['medium_incorrect_hist'], width, color='red')
        if i == 0:
            ax_med.set_title("Medium Loss Hist", fontsize=10)
        ax_med.tick_params(axis='both', labelsize=8)
        
        # Column 2: Hard losses.
        ax_hard = fig.add_subplot(gs_top[i, 2])
        ax_hard.bar(x - width/2, s_break['hard_correct_hist'], width, color='green')
        ax_hard.bar(x + width/2, s_break['hard_incorrect_hist'], width, color='red')
        if i == 0:
            ax_hard.set_title("Hard Loss Hist", fontsize=10)
        ax_hard.tick_params(axis='both', labelsize=8)
        
        # Column 3: State Info.
        ax_info = fig.add_subplot(gs_top[i, 3])
        state_info = np.concatenate([s_break['relative_sizes'], s_break['extra']])
        colors_info = ['blue', 'orange', 'purple', 'cyan', 'magenta']
        ax_info.bar(np.arange(5), state_info, color=colors_info)
        ax_info.set_xticks(np.arange(5))
        ax_info.set_xticklabels(['Easy', 'Med', 'Hard', 'PhaseRatio', 'AvailRatio'], 
                                rotation=45, fontsize=8)
        if i == 0:
            ax_info.set_title("State Info", fontsize=10)
        ax_info.tick_params(axis='both', labelsize=8)
    
    # BOTTOM BLOCK: Aggregated evolution across phases.
    phases = np.arange(1, num_phases + 1)
    learning_rates = [episode['actions'][i][0] for i in range(num_phases)]
    sample_usage = [episode['actions'][i][4] for i in range(num_phases)]
    mixing_ratios = np.array([episode['actions'][i][1:4] for i in range(num_phases)])
    rewards_phase = [episode['rewards'][i] for i in range(num_phases)]
    # Adjust last phase reward by dividing by 10.
    adjusted_rewards = np.array(rewards_phase, dtype=float)
    if len(adjusted_rewards) > 0:
        adjusted_rewards[-1] = adjusted_rewards[-1] / 10.0

    # Aggregated Plot 1: Learning Rate Evolution.
    ax_lr = fig.add_subplot(gs_bot[0, 0])
    ax_lr.plot(phases, learning_rates, marker='o', linestyle='-', color='blue')
    ax_lr.set_title("Learning Rate", fontsize=10)
    ax_lr.set_xlabel("Phase", fontsize=9)
    ax_lr.set_ylabel("LR", fontsize=9)
    ax_lr.set_xticks(phases)
    
    # Aggregated Plot 2: Sample Usage Evolution.
    ax_usage = fig.add_subplot(gs_bot[0, 1])
    ax_usage.plot(phases, sample_usage, marker='o', linestyle='-', color='orange')
    ax_usage.set_title("Sample Usage", fontsize=10)
    ax_usage.set_xlabel("Phase", fontsize=9)
    ax_usage.set_ylabel("Usage", fontsize=9)
    ax_usage.set_xticks(phases)
    
    # Aggregated Plot 3: Mixing Ratios as a stacked bar plot.
    ax_mix = fig.add_subplot(gs_bot[0, 2])
    bar_width = 0.6
    for i in range(num_phases):
        ratio = mixing_ratios[i]
        ax_mix.bar(i, ratio[0], color='green', width=bar_width, label='Easy' if i==0 else "")
        ax_mix.bar(i, ratio[1], bottom=ratio[0], color='orange', width=bar_width, label='Med' if i==0 else "")
        ax_mix.bar(i, ratio[2], bottom=ratio[0]+ratio[1], color='red', width=bar_width, label='Hard' if i==0 else "")
    ax_mix.set_xticks(np.arange(num_phases))
    ax_mix.set_xticklabels([f"P{p}" for p in phases])
    ax_mix.set_title("Mixing Ratios", fontsize=10)
    ax_mix.set_xlabel("Phase", fontsize=9)
    ax_mix.set_ylabel("Ratio", fontsize=9)
    ax_mix.legend(fontsize=8)
    
    # Aggregated Plot 4: Reward Evolution.
    ax_reward = fig.add_subplot(gs_bot[0, 3])
    ax_reward.plot(phases, adjusted_rewards, marker='o', linestyle='-', color='black')
    ax_reward.set_title("Reward", fontsize=10)
    ax_reward.set_xlabel("Phase", fontsize=9)
    ax_reward.set_ylabel("Reward", fontsize=9)
    ax_reward.set_xticks(phases)
    
    plt.tight_layout()
    fig_filename = os.path.join(output_dir, f"{group_name}_episode_{episode['index']}_detailed.png")
    plt.savefig(fig_filename)
    plt.close()
    print(f"Saved detailed figure for {group_name} episode {episode['index']} to {fig_filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze training history from npz file with detailed figures.")
    parser.add_argument("--npz_file", type=str, default="results/curriculum_rl/evolutionary_dataset.npz",
                        help="Path to the npz file saved by generate_dataset.py")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Number of bins used in the loss histograms in the state vector (default 64)")
    args = parser.parse_args()
    
    # Create output directory "history"
    output_dir = "history2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data.
    states, actions, rewards, dones = load_data(args.npz_file)
    print(f"Loaded {states.shape[0]} transitions from {args.npz_file}")
    
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
