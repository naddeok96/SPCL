#!/usr/bin/env python3
"""Reorganize parallel evaluation history into per-episode sequences.

``run_evolution_parallel.sh`` evaluates ``NUM_MODELS`` candidates at once.  For
each environment step, the transitions of all models are appended in a single
batch, and a block of ``NUM_MODELS`` ``done`` flags marks the end of the batch.
For example, with ``NUM_MODELS == 1000`` and three steps per episode the raw
``dones`` tensor looks like::

    D[0]       == False
    D[0+1000]  == False
    D[0+2000]  == True

``analyze_history.py`` expects transitions for each episode to be contiguous, so
this script reorganizes the batched history into the expected order.  All paths
and constants are defined below; no command-line arguments are required.
"""
import glob
import os
import torch

# Directory containing ``history_gen*.pt`` files produced during parallel
# evolution.
HISTORY_DIR = "vec_evo_results_parallel/history"

# Output file that will contain the reorganized dataset.
OUTPUT_FILE = "vec_evo_results_parallel/fixed_history.pt"

# Number of models evaluated in parallel.  This must match the ``NUM_MODELS``
# parameter used in ``run_evolution_parallel.sh`` / ``config_parallel.yaml``.
NUM_MODELS = 1000


def load_history_files(history_dir: str):
    files = sorted(glob.glob(os.path.join(history_dir, "history_gen*.pt")))
    if not files:
        raise FileNotFoundError(f"No history_gen*.pt files found in {history_dir}")
    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    for fn in files:
        d = torch.load(fn, map_location="cpu")
        all_s.append(d["states"])
        all_a.append(d["actions"])
        all_r.append(d["rewards"])
        all_ns.append(d["next_states"])
        all_d.append(d["dones"])
    return {
        "states": torch.cat(all_s, dim=0),
        "actions": torch.cat(all_a, dim=0),
        "rewards": torch.cat(all_r, dim=0),
        "next_states": torch.cat(all_ns, dim=0),
        "dones": torch.cat(all_d, dim=0),
    }


def reorganize_parallel_history(data: dict, num_models: int) -> dict:
    """Reorder transitions assuming ``num_models`` were evaluated in parallel.

    Episodes are reconstructed by following the pattern::

        D[start + p * num_models + m]

    where ``start`` is the first transition index of the batch and ``m`` is the
    model index.  ``D[start]`` must be ``False`` and ``D[start + k*num_models]``
    must become ``True`` simultaneously for all ``m``.  The next episode begins
    at ``start + k*num_models``.
    """
    S, A, R, NS, D = (
        data["states"],
        data["actions"],
        data["rewards"],
        data["next_states"],
        data["dones"],
    )
    n = D.size(0)
    if n == 0:
        return data

    new_s, new_a, new_r, new_ns, new_d = [], [], [], [], []
    start = 0
    while start < n:
        if bool(D[start]) and A[start,4]!=1.0:
            raise RuntimeError(f"Expected first done flag at {start} to be False")

        steps = 0
        while True:
            idx = start + steps * num_models
            if idx >= n:
                raise RuntimeError(
                    f"Reached end of history while searching for episode end "
                    f"starting at {start}"
                )
            if bool(D[idx]):
                steps += 1
                break
            steps += 1

        if steps > 3:
            raise RuntimeError(
                f"Episode starting at {start} has {steps} steps; expected at most 3"
            )

        end_idx = start + steps * num_models

        # Validate pattern for all models
        for m in range(num_models):
            for p in range(steps):
                idx = start + p * num_models + m
                dflag = bool(D[idx])
                if p < steps - 1 and dflag:
                    raise RuntimeError(
                        f"Done flag early at index {idx} (model {m}, step {p})"
                    )
                if p == steps - 1 and not dflag:
                    raise RuntimeError(
                        f"Missing final done flag at index {idx} (model {m})"
                    )

                new_s.append(S[idx])
                new_a.append(A[idx])
                new_r.append(R[idx])
                new_ns.append(NS[idx])
                new_d.append(p == steps - 1)

        start = end_idx

    return {
        "states": torch.stack(new_s),
        "actions": torch.stack(new_a),
        "rewards": torch.stack(new_r),
        "next_states": torch.stack(new_ns),
        "dones": torch.tensor(new_d, dtype=torch.bool),
    }


def main():
    data = load_history_files(HISTORY_DIR)
    fixed = reorganize_parallel_history(data, NUM_MODELS)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(fixed, OUTPUT_FILE)
    print(f"Saved reorganized history to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()