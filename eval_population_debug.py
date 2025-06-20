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
    # ==== Hard-coded configuration ====
    config_path    = "config_parallel.yaml"    # path to your config YAML
    pop_file       = "vec_evo_results_parallel/populations/pop_gen_0.pt"  # .pt containing 'population'
    start_idx      = 0                # first candidate index
    num_candidates = 4              # how many candidates to evaluate
    out_file       = "temp4/eval_slice.pt"  # where to write this slice's .pt
    num_models     = 10                # number of models to evaluate per candidate
    model_type     = "mlp"             # override model type from config ("cnn" or "mlp")
    parallel       = True            # use parallel evaluation?

    cfg = load_config(config_path)
    if model_type:
        cfg["model_type"] = model_type
    set_seed(cfg.get("seed", 42))
    cfg["device"]    = "cuda:0"
    cfg["device_id"] = 0

    data       = torch.load(pop_file, map_location="cpu")
    population = data["population"]
    cand_dim   = population.size(1)
    num_phases = int(cfg["curriculum"].get("max_phases", 3))

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    agg_rewards, indices = [], []

    env = None
    if not parallel:
        env = CurriculumEnv(cfg)

    # tqdm over the slice
    for local_idx in trange(num_candidates,
                            desc=f"Eval gen slice {start_idx}-{start_idx+num_candidates-1}",
                            unit="cand"):
        idx = start_idx + local_idx
        if parallel:
            transitions, total_r = evaluate_candidate_parallel(
                cfg,
                population[idx],
                cand_dim,
                num_phases,
                num_models,
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
    }, out_file)
    print(f"Wrote eval slice → {out_file}")

if __name__ == "__main__":
    main()




