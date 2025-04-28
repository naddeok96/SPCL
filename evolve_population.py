# evolve_population.py
#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import yaml
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

    # load config
    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)

    pop_data = np.load(args.pop_file)
    population = pop_data["population"]
    pop_size, cand_dim = population.shape
    num_phases = cfg["curriculum"].get("max_phases", 3)
    lr_range   = tuple(cfg["curriculum"]["learning_rate_range"])
    top_k      = cfg["rl"].get("ea_top_k", 10)
    mut_rate   = cfg["rl"].get("ea_mutation_rate", 0.1)

    # gather eval parts
    pattern = os.path.join(args.eval_dir, f"eval_gen{args.gen}_part*.npz")
    reward_map = np.zeros(pop_size, dtype=float)
    history_segments = []

    for fn in sorted(glob.glob(pattern)):
        data = np.load(fn)
        idxs = data["candidate_indices"]
        rwd  = data["aggregated_rewards"]
        reward_map[idxs] = rwd
        # collect transitions
        history_segments.append((
            data["states"],
            data["actions"],
            data["rewards"],
            data["next_states"],
            data["dones"],
        ))

    # save this generation's history
    hs = np.concatenate([h[0] for h in history_segments])
    ha = np.concatenate([h[1] for h in history_segments])
    hr = np.concatenate([h[2] for h in history_segments])
    hn = np.concatenate([h[3] for h in history_segments])
    hd = np.concatenate([h[4] for h in history_segments])

    os.makedirs(args.history_dir, exist_ok=True)
    hist_fn = os.path.join(args.history_dir, f"history_gen{args.gen}.npz")
    np.savez(hist_fn, states=hs, actions=ha, rewards=hr, next_states=hn, dones=hd)
    print(f"Saved gen {args.gen} history to {hist_fn}")

    # select top_k and produce next population
    sel_idxs = np.argsort(-reward_map)[:top_k]
    selected = population[sel_idxs]

    new_pop = []
    while len(new_pop) < pop_size:
        p1, p2 = random.sample(list(selected), 2)
        child = crossover(p1, p2, cand_dim, num_phases, lr_range)
        child = mutate(child, mut_rate, cand_dim, num_phases, lr_range)
        new_pop.append(child)
    new_pop = np.stack(new_pop)

    os.makedirs(os.path.dirname(args.output_population), exist_ok=True)
    np.savez(args.output_population, population=new_pop)
    print(f"Produced generation {args.gen+1} population → {args.output_population}")

if __name__ == "__main__":
    main()
