# merge_history.py
#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_dir", required=True)
    p.add_argument("--output",      required=True,
                   help="final .npz with full states/actions/rewards/…")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.history_dir, "history_gen*.npz")))
    all_s, all_a, all_r, all_ns, all_d = [],[],[],[],[]

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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, states=S, actions=A, rewards=R, next_states=NS, dones=D)
    print(f"Final evolutionary dataset saved to {args.output}")

if __name__ == "__main__":
    main()
