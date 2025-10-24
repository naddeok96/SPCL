# quick_audit.py
import torch
d = torch.load("/data/naddeok/spcl/merged_history.pt", map_location="cpu")
a = d["actions"]; r = d["rewards"]; done = d["dones"]

def edge_frac(x, lo, hi, eps=0.01):
    return ((x<=lo+eps) | (x>=hi-eps)).float().mean().item()

print("edge% lr:", edge_frac(a[:,0], 0.001, 0.01))
print("edge% usage:", ((a[:,4]<=0.01)|(a[:,4]>=0.99)).float().mean().item())
mix = a[:,1:4].clamp(1e-8,1); ent = (-(mix*mix.log()).sum(dim=1))
print("mix entropy mean:", ent.mean().item(), "min:", ent.min().item(), "max:", ent.max().item())

# terminal-only reward ratio
print("terminal rewards %:", (done.float().mean().item()))

# reward vs. edge-ness correlation
import numpy as np
edge_score = ((a[:,0]-0.001)/(0.01-0.001)).abs() + (1 - ent/ent.max()) + (a[:,4].clamp(0,1)-0.5).abs()
ep_idx = []; ep_ret=[]; s=0; ret=0.
for i, rr in enumerate(r):
    ret += float(rr)
    if bool(done[i]): ep_idx.append((s,i+1)); ep_ret.append(ret); s=i+1; ret=0.
from scipy.stats import spearmanr
print("Kendall/Spearman vs edge-score:",
      spearmanr(edge_score[[j for s,e in ep_idx for j in range(e-1,e)]].numpy(), np.array(ep_ret)).correlation)
