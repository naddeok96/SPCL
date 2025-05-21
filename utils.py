"""
Utility functions for computing loss histograms, constructing observation vectors,
and other helper methods.
"""

import torch
import matplotlib.pyplot as plt

# --- tweakable max clip and alpha ---
_MAX_LOSS = 13.8
_ALPHA    = 2.0

# cache edges per (num_bins, device)
_edges_cache = {}

def _get_bin_edges(num_bins: int, device: torch.device):
    key = (num_bins, device)
    if key not in _edges_cache:
        rel = torch.linspace(0.0, 1.0, steps=num_bins+1, device=device)
        _edges_cache[key] = (rel ** _ALPHA) * _MAX_LOSS
    return _edges_cache[key]

def compute_loss_histogram(losses, num_bins: int, device="cuda:0"):
    """
    Vectorized 1D histogram on GPU via torch.bincount.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # to GPU tensor
    if not torch.is_tensor(losses):
        losses = torch.tensor(losses, device=device, dtype=torch.float32)
    else:
        losses = losses.to(device).float()
    losses = losses.flatten().clamp(0.0, _MAX_LOSS)

    # bucket boundaries
    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    # bin indices
    bins = torch.bucketize(losses, boundaries)

    # fast count
    hist = torch.bincount(bins, minlength=num_bins).to(device, torch.float32)
    tot  = hist.sum()
    if tot > 0:
        hist /= tot

    return hist, edges

def compute_dual_loss_histograms(losses_correct, losses_incorrect, num_bins: int, device="cuda:0"):
    """
    Joint correct/incorrect histograms, vectorized.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # pack into GPU tensors
    lc = torch.as_tensor(losses_correct or [], device=device, dtype=torch.float32).flatten()
    li = torch.as_tensor(losses_incorrect or [], device=device, dtype=torch.float32).flatten()
    lc = lc.clamp(0.0, _MAX_LOSS)
    li = li.clamp(0.0, _MAX_LOSS)

    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    bins_c = torch.bucketize(lc, boundaries)
    bins_i = torch.bucketize(li, boundaries)

    hist_c = torch.bincount(bins_c, minlength=num_bins).to(device, torch.float32)
    hist_i = torch.bincount(bins_i, minlength=num_bins).to(device, torch.float32)

    total = hist_c + hist_i
    tot_sum = total.sum()
    if tot_sum > 0:
        hist_c /= tot_sum
        hist_i /= tot_sum

    return hist_c, hist_i, edges

def plot_histogram(hist, edges=None, title=None, filename=None):
    plt.figure()
    vals = hist.cpu().tolist()
    if edges is not None:
        e = edges.cpu().tolist()
        centers = [(e[i]+e[i+1])/2 for i in range(len(e)-1)]
        widths  = [(e[i+1]-e[i])    for i in range(len(e)-1)]
        plt.bar(centers, vals, width=widths, align="center")
        plt.xticks(e, rotation=45)
    else:
        plt.bar(range(len(vals)), vals)
    if title:    plt.title(title)
    plt.xlabel("Loss bin"); plt.ylabel("Freq")
    plt.tight_layout()
    if filename: plt.savefig(filename)
    plt.close()
def plot_histogram(hist, edges=None, title=None, filename=None):
    """
    Plot a normalized histogram (Tensor) with optional variable-width bins.

    Args:
        hist (torch.Tensor): Length-N tensor of frequencies.
        edges (torch.Tensor or None): Length-(N+1) tensor of bin edges.
        title (str or None): Plot title.
        filename (str or None): If given, save to this path.
    """
    plt.figure()
    hist_vals = hist.cpu().tolist()
    
    if edges is not None:
        edges_cpu = edges.cpu()
        centers = ((edges_cpu[:-1] + edges_cpu[1:]) / 2).tolist()
        widths  = (edges_cpu[1:] - edges_cpu[:-1]).tolist()
        plt.bar(centers, hist_vals, width=widths, align='center')
        plt.xticks(edges_cpu.tolist(), rotation=45)
    else:
        positions = list(range(len(hist_vals)))
        plt.bar(positions, hist_vals)

    if title:
        plt.title(title)
    plt.xlabel("Loss bins")
    plt.ylabel("Normalized frequency")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()







