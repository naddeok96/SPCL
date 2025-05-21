#!/usr/bin/env python3
import argparse
import os
import glob
import torch
import random

from population_utils import load_config, mutate, crossover

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True)
    p.add_argument("--pop_file",          required=True)
    p.add_argument("--eval_dir",          required=True)
    p.add_argument("--gen",      type=int,required=True)
    p.add_argument("--output_population",required=True)
    p.add_argument("--history_dir",       required=True)
    args = p.parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    num_phases = cfg["curriculum"].get("max_phases", 3)
    lr_range   = tuple(cfg["curriculum"]["learning_rate_range"])
    top_k      = cfg["rl"].get("ea_top_k", 10)
    mut_rate   = cfg["rl"].get("ea_mutation_rate", 0.1)

    # Load current population
    pop_data   = torch.load(args.pop_file, map_location="cpu")
    population = pop_data["population"]                    # Tensor [pop_size, cand_dim]
    pop_size, cand_dim = population.shape

    # Prepare to collect rewards and history
    reward_map       = torch.zeros(pop_size, dtype=torch.float32)
    states_list      = []
    actions_list     = []
    rewards_list     = []
    next_states_list = []
    dones_list       = []

    # Gather per-candidate eval files
    pattern = os.path.join(args.eval_dir, f"eval_gen{args.gen}_part*.pt")
    for fn in sorted(glob.glob(pattern)):
        data = torch.load(fn, map_location="cpu")
        idxs = data["candidate_indices"]              # Tensor of indices
        rwd  = data["aggregated_rewards"]            # Tensor of rewards
        reward_map[idxs] = rwd

        # Accumulate history segments
        states_list.append(data["states"])
        actions_list.append(data["actions"])
        rewards_list.append(data["rewards"])
        next_states_list.append(data["next_states"])
        dones_list.append(data["dones"])

    # Concatenate all phases of this generation
    hs = torch.cat(states_list,      dim=0)
    ha = torch.cat(actions_list,     dim=0)
    hr = torch.cat(rewards_list,     dim=0)
    hn = torch.cat(next_states_list, dim=0)
    hd = torch.cat(dones_list,       dim=0)

    # Save this generation's history
    os.makedirs(args.history_dir, exist_ok=True)
    hist_fn = os.path.join(args.history_dir, f"history_gen{args.gen}.pt")
    torch.save({
        "states":      hs,
        "actions":     ha,
        "rewards":     hr,
        "next_states": hn,
        "dones":       hd
    }, hist_fn)
    print(f"Saved gen {args.gen} history to {hist_fn}")

    # Select top_k candidates by reward
    sel_idxs = torch.topk(reward_map, k=top_k, largest=True).indices
    selected = population[sel_idxs]  # Tensor [top_k, cand_dim]

    # Build next generation
    new_pop = []
    while len(new_pop) < pop_size:
        p1, p2 = random.sample(list(selected), 2)
        child  = crossover(p1, p2, cand_dim, num_phases, lr_range)
        child  = mutate(child, mut_rate, cand_dim, num_phases, lr_range)
        new_pop.append(child)
    new_pop = torch.stack(new_pop, dim=0)  # Tensor [pop_size, cand_dim]

    # Save the new population
    os.makedirs(os.path.dirname(args.output_population), exist_ok=True)
    torch.save({"population": new_pop}, args.output_population)
    print(f"Produced generation {args.gen+1} population → {args.output_population}")

if __name__ == "__main__":
    main()


