# init_population.py
#!/usr/bin/env python3
import argparse
import os
import numpy as np

from population_utils import set_seed, load_config, initialize_population

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--output",    required=True,
                   help=".npz file to write generation‑0 population")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    num_phases    = cfg["curriculum"].get("max_phases", 3)
    unit          = 5
    candidate_dim = num_phases * unit
    pop_size      = cfg["rl"].get("ea_pop_size", 100)
    lr_range      = tuple(cfg["curriculum"]["learning_rate_range"])
    macro_actions = cfg["rl"].get("macro_actions", None)

    pop = initialize_population(
        pop_size, candidate_dim, num_phases, macro_actions, lr_range
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, population=pop)
    print(f"Saved initial population ({pop_size} × {candidate_dim}) to {args.output}")

if __name__ == "__main__":
    main()
