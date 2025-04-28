# eval_population.py
#!/usr/bin/env python3
import argparse
import numpy as np
import copy
from tqdm import trange

from population_utils import set_seed, load_config, evaluate_candidate
from curriculum_env import CurriculumEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True, help="path to your config YAML")
    p.add_argument("--pop_file",       required=True, help=".npz containing ‘population’")
    p.add_argument("--start_idx",    type=int,  required=True, help="first candidate index")
    p.add_argument("--num_candidates", type=int, required=True, help="how many candidates")
    p.add_argument("--out_file",       required=True, help="where to write this slice’s .npz")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    cfg["device"]    = "cuda:0"
    cfg["device_id"] = 0
    env = CurriculumEnv(cfg)

    data       = np.load(args.pop_file)
    population = data["population"]
    cand_dim   = population.shape[1]
    num_phases = int(cfg["curriculum"].get("max_phases", 3))

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    agg_rewards, indices = [], []

    # tqdm over the slice
    for local_idx in trange(args.num_candidates,
                            desc=f"Eval gen slice {args.start_idx}-{args.start_idx+args.num_candidates-1}",
                            unit="cand"):
        idx = args.start_idx + local_idx
        transitions, total_r = evaluate_candidate(
            copy.deepcopy(env),
            population[idx],
            cand_dim,
            num_phases
        )
        for (s, a, r, ns, d) in transitions:
            all_s .append(s)
            all_a .append(a)
            all_r .append(r)
            all_ns.append(ns)
            all_d .append(d)
        agg_rewards.append(total_r)
        indices    .append(idx)

    # save
    np.savez(
        args.out_file,
        candidate_indices   = np.array(indices,    dtype=int),
        aggregated_rewards  = np.array(agg_rewards, dtype=float),
        states              = np.stack(all_s),
        actions             = np.stack(all_a),
        rewards             = np.array(all_r),
        next_states         = np.stack(all_ns),
        dones               = np.array(all_d,      dtype=bool),
    )
    print(f"Wrote eval slice → {args.out_file}")

if __name__ == "__main__":
    main()
