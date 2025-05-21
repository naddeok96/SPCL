# merge_history.py
#!/usr/bin/env python3
import argparse
import glob
import torch
import os
import re
import sys

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--history_dir", required=True)
    p.add_argument("--output",      required=True,
                help="final .pt with full states/actions/rewards/…")
    args = p.parse_args()

    output = args.output
    history_dir= args.history_dir

    # output = "temp/temp.pt"
    # history_dir = "eval_parts/"

    basename = os.path.basename(output)
    m = re.match(r".*gen(\d+)\.pt$", basename)
    if m:
        pattern = f"eval_gen{m.group(1)}_part*.pt"
    else:
        pattern = "eval_gen*_part*.pt"

    # collect all matching parts
    files = sorted(glob.glob(os.path.join(history_dir, pattern)))
    if not files:
        print(f"Error: no files matching '{pattern}' in '{history_dir}'", file=sys.stderr)
        sys.exit(1)

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []

    for fn in files:
        d = torch.load(fn, map_location='cpu')
        all_s.append(d["states"])
        all_a.append(d["actions"])
        all_r.append(d["rewards"])
        all_ns.append(d["next_states"])
        all_d.append(d["dones"])

    S  = torch.cat(all_s,  dim=0)
    A  = torch.cat(all_a,  dim=0)
    R  = torch.cat(all_r,  dim=0)
    NS = torch.cat(all_ns, dim=0)
    D  = torch.cat(all_d,  dim=0)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save({
        'states':      S,
        'actions':     A,
        'rewards':     R,
        'next_states': NS,
        'dones':       D
    }, output)
    print(f"Final evolutionary dataset saved to {output}")

if __name__ == "__main__":
    main()




