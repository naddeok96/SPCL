"""
Utility functions for computing loss histograms, constructing observation vectors,
and other helper methods.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_loss_histogram(losses, num_bins):
    """
    Compute a normalized histogram for a 1D array of losses,
    using variable-width bins denser near the low end.

    Args:
        losses (np.array or torch.Tensor): Array of loss values.
        num_bins (int): Number of bins to use for the histogram.

    Returns:
        hist (np.ndarray): A normalized histogram whose bins sum to 1.
        edges (np.ndarray): Bin-edge locations for plotting.
    """
    # Convert to NumPy array
    if torch.is_tensor(losses):
        losses = losses.cpu().numpy().flatten()
    else:
        losses = np.array(losses).flatten()

    # Clip into [0, 13.8] → 13.8 ≈ -ln(1e-6) practical upper bound
    min_val, max_val = 0.0, 13.8
    losses_clipped = np.clip(losses, min_val, max_val)

    # Build variable-width bin edges:
    # alpha > 1: denser bins at low losses; alpha < 1: denser at high losses
    alpha = 2.0
    rel = np.linspace(0, 1, num_bins + 1)
    edges = min_val + (max_val - min_val) * rel**alpha

    # Compute histogram on those edges
    hist, _ = np.histogram(losses_clipped, bins=edges)

    # Normalize so bins sum to 1
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist, edges


def compute_dual_loss_histograms(losses_correct, losses_incorrect, num_bins):
    """
    Compute histograms for correct and incorrect losses over the same
    variable-width bins [0,13.8], and normalize so their combined sum is 1.

    Args:
        losses_correct (np.array or torch.Tensor): Loss values for correct samples.
        losses_incorrect (np.array or torch.Tensor): Loss values for incorrect samples.
        num_bins (int): Number of bins to use.

    Returns:
        norm_correct (np.ndarray): Normalized histogram for correct losses.
        norm_incorrect (np.ndarray): Normalized histogram for incorrect losses.
        edges (np.ndarray): Bin-edge locations for plotting.
    """
    # Convert to NumPy arrays
    if torch.is_tensor(losses_correct[0]):
        lc = np.array([l.cpu().numpy() for l in losses_correct]).flatten()
    else:
        lc = np.array(losses_correct).flatten()
    if torch.is_tensor(losses_incorrect[0]):
        li =  np.array([l.cpu().numpy() for l in losses_incorrect]).flatten()
    else:
        li = np.array(losses_incorrect).flatten()

    # Clip into [0, 13.8]
    min_val, max_val = 0.0, 13.8
    lc = np.clip(lc, min_val, max_val)
    li = np.clip(li, min_val, max_val)

    # Build variable-width bin edges (denser at low end)
    alpha = 2.0
    rel = np.linspace(0, 1, num_bins + 1)
    edges = min_val + (max_val - min_val) * rel**alpha

    # Compute histograms
    hist_correct, _   = np.histogram(lc, bins=edges)
    hist_incorrect, _ = np.histogram(li, bins=edges)

    # Normalize combined
    combined = hist_correct + hist_incorrect
    total = combined.sum()
    if total > 0:
        norm_correct   = hist_correct   / total
        norm_incorrect = hist_incorrect / total
    else:
        norm_correct   = np.zeros_like(hist_correct,   dtype=float)
        norm_incorrect = np.zeros_like(hist_incorrect, dtype=float)

    return norm_correct, norm_incorrect

def construct_observation(easy_correct, easy_incorrect,
                          medium_correct, medium_incorrect,
                          hard_correct, hard_incorrect,
                          easy_count, medium_count, hard_count, num_bins):
    """
    Construct the observation vector by concatenating six binned normalized loss vectors
    (for correct and incorrect samples for the easy, medium, and hard datasets) and the
    relative dataset sizes.
    
    Args:
        easy_correct, easy_incorrect, medium_correct, medium_incorrect,
        hard_correct, hard_incorrect (list or np.array): Lists/arrays of loss values.
        
        easy_count, medium_count, hard_count (int): The sample counts for each dataset.
        
        num_bins (int): Number of bins for each loss histogram.
        
    Returns:
        observation (np.ndarray): A fixed-length vector (6 * num_bins + 3 features).
    """
    ec_hist, ei_hist = compute_dual_loss_histograms(easy_correct, easy_incorrect, num_bins)
    mc_hist, mi_hist = compute_dual_loss_histograms(medium_correct, medium_incorrect, num_bins)
    hc_hist, hi_hist = compute_dual_loss_histograms(hard_correct, hard_incorrect, num_bins)

    
    # Normalize the dataset sizes so they sum to 1.
    total_count = easy_count + medium_count + hard_count
    rel_sizes = np.array([easy_count, medium_count, hard_count], dtype=float) / total_count
    
    observation = np.concatenate([ec_hist, ei_hist, mc_hist, mi_hist, hc_hist, hi_hist, rel_sizes])
    return observation

def plot_histogram(hist, edges=None, title=None, filename=None):
    """
    Plot a normalized histogram with optional variable-width bins and fixed ticks.

    Args:
        hist (np.ndarray): Histogram bin counts or frequencies.
        edges (np.ndarray or None): If provided, use these as bin edges; otherwise assume uniform bins.
        title (str or None): Plot title.
        filename (str or None): If provided, save figure to this path.
    """
    plt.figure()
    if edges is not None:
        # Compute bin centers and widths
        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]
        plt.bar(centers, hist, width=widths, align='center')
        # Set ticks at every bin edge
        plt.xticks(edges, rotation=45)
    else:
        plt.bar(np.arange(len(hist)), hist)
    if title:
        plt.title(title)
    plt.xlabel("Loss bins")
    plt.ylabel("Normalized frequency")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()



