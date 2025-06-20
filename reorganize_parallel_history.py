#!/usr/bin/env python3
"""Reorganize parallel evaluation history into per-episode sequences.

This utility loads all ``history_gen*.pt`` files from the specified directory,
concatenates them, reorders the transitions produced by
``evaluate_candidate_parallel`` so that each episode corresponds to a single
model run, and saves the fixed dataset.
"""
import argparse
import glob
import os
import torch


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


def reorganize_parallel_history(data: dict) -> dict:
    S, A, R, NS, D = data["states"], data["actions"], data["rewards"], data["next_states"], data["dones"]
    n = D.size(0)
    if n == 0:
        return data
    # determine num_models from the first run of consecutive dones
    done_idx = (D.nonzero(as_tuple=False).flatten()).tolist()
    if not done_idx:
        # no done flags, nothing to fix
        return data
    first_done = done_idx[0]
    num_models = 1
    idx = first_done + 1
    while idx < n and D[idx]:
        num_models += 1
        idx += 1

    new_s, new_a, new_r, new_ns, new_d = [], [], [], [], []
    i = 0
    while i < n:
        # locate first done within this batch
        j = i
        while j < n and not D[j]:
            j += 1
        if j == n:
            raise RuntimeError("Malformed history: reached end without done flag")
        # length of consecutive dones tells us how many models were evaluated
        k = j
        while k < n and D[k]:
            k += 1
        num_models_here = k - j
        if num_models_here != num_models:
            raise RuntimeError("Inconsistent number of models within history")
        total_trans = k - i
        steps_per_model = total_trans // num_models
        for m in range(num_models):
            for p in range(steps_per_model):
                idx = i + p * num_models + m
                new_s.append(S[idx])
                new_a.append(A[idx])
                new_r.append(R[idx])
                new_ns.append(NS[idx])
                new_d.append(p == steps_per_model - 1)
        i = k

    return {
        "states": torch.stack(new_s),
        "actions": torch.stack(new_a),
        "rewards": torch.stack(new_r),
        "next_states": torch.stack(new_ns),
        "dones": torch.tensor(new_d, dtype=torch.bool),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_dir", required=True, help="Directory containing history_gen*.pt files")
    p.add_argument("--output", required=True, help="Output .pt file")
    args = p.parse_args()

    data = load_history_files(args.history_dir)
    fixed = reorganize_parallel_history(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(fixed, args.output)
    print(f"Saved reorganized history to {args.output}")


if __name__ == "__main__":
    main()
