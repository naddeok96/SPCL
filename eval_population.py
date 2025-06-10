# eval_population.py
#!/usr/bin/env python3
import argparse
import torch
from tqdm import trange

from population_utils import (
    set_seed,
    load_config,
    evaluate_candidate,
    evaluate_candidate_parallel,
)
from curriculum_env import CurriculumEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True, help="path to your config YAML")
    p.add_argument("--pop_file",       required=True, help=".pt c`ntaining ‘population’")
    p.add_argument("--start_idx",    type=int,  required=True, help="first candidate index")
    p.add_argument("--num_candidates", type=int, required=True, help="how many candidates")
    p.add_argument("--out_file",       required=True, help="where to write this slice’s .pt")
    p.add_argument("--num_models", type=int, default=1, help="number of models to evaluate per candidate")
    p.add_argument("--model_type", choices=["cnn", "mlp"], default=None,
                   help="override model type from config")
    p.add_argument("--parallel", action="store_true",
                   help="use evaluate_candidate_parallel (for multi-model eval)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model_type:
        cfg["model_type"] = args.model_type
    set_seed(cfg.get("seed", 42))
    cfg["device"]    = "cuda:0"
    cfg["device_id"] = 0

    data       = torch.load(args.pop_file, map_location="cpu")
    population = data["population"]
    cand_dim   = population.size(1)
    num_phases = int(cfg["curriculum"].get("max_phases", 3))

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    agg_rewards, indices = [], []

    env = None
    if not args.parallel:
        env = CurriculumEnv(cfg)

    # tqdm over the slice
    for local_idx in trange(args.num_candidates,
                            desc=f"Eval gen slice {args.start_idx}-{args.start_idx+args.num_candidates-1}",
                            unit="cand"):
        idx = args.start_idx + local_idx
        if args.parallel:
            transitions, total_r = evaluate_candidate_parallel(
                cfg,
                population[idx],
                cand_dim,
                num_phases,
                args.num_models,
            )
        else:
            transitions, total_r = evaluate_candidate(
                env,
                population[idx],
                cand_dim,
                num_phases,
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
    torch.save({
        'candidate_indices'  : torch.tensor(indices,    dtype=torch.int64),
        'aggregated_rewards' : torch.tensor(agg_rewards, dtype=torch.float32),
        'states'             : torch.stack(all_s),
        'actions'            : torch.stack(all_a),
        'rewards'            : torch.tensor(all_r,       dtype=torch.float32),
        'next_states'        : torch.stack(all_ns),
        'dones'              : torch.tensor(all_d,       dtype=torch.bool),
    }, args.out_file)
    print(f"Wrote eval slice → {args.out_file}")

if __name__ == "__main__":
    main()




