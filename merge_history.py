# merge_history.py
#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import os
import re
import sys

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--history_dir", required=True)
    p.add_argument("--output",      required=True,
                help="final .npz with full states/actions/rewards/…")
    args = p.parse_args()

    output = args.output
    history_dir= args.history_dir

    # output = "temp/temp.npz"
    # history_dir = "eval_parts/"

    basename = os.path.basename(output)
    m = re.match(r".*gen(\d+)\.npz$", basename)
    if m:
        pattern = f"eval_gen{m.group(1)}_part*.npz"
    else:
        pattern = "eval_gen*_part*.npz"

    # collect all matching parts
    files = sorted(glob.glob(os.path.join(history_dir, pattern)))
    if not files:
        print(f"Error: no files matching '{pattern}' in '{history_dir}'", file=sys.stderr)
        sys.exit(1)

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []

    for fn in files:
        d = np.load(fn)
        all_s.append(d["states"])
        all_a.append(d["actions"])
        all_r.append(d["rewards"])
        all_ns.append(d["next_states"])
        all_d.append(d["dones"])

    S = np.concatenate(all_s)
    A = np.concatenate(all_a)
    R = np.concatenate(all_r)
    NS= np.concatenate(all_ns)
    D = np.concatenate(all_d)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(output, states=S, actions=A, rewards=R, next_states=NS, dones=D)
    print(f"Final evolutionary dataset saved to {output}")

if __name__ == "__main__":
    main()
