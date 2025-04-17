"""
Utility functions for computing loss histograms, constructing observation vectors,
and other helper methods.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_loss_histogram(losses, num_bins):
    """
    Compute a normalized histogram for a 1D array of losses.
    
    Args:
        losses (np.array or torch.Tensor): Array of loss values.
        num_bins (int): Number of bins to use for the histogram.
        
    Returns:
        hist (np.ndarray): A normalized histogram whose bins sum to 1.
    """
    # Convert to NumPy array if necessary.
    if torch.is_tensor(losses):
        losses = losses.cpu().numpy().flatten()
    else:
        losses = np.array(losses).flatten()
    
    hist, _ = np.histogram(losses, bins=num_bins, range=(losses.min(), losses.max()), density=False)
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist

def compute_dual_loss_histograms(losses_correct, losses_incorrect, num_bins):
    """
    Compute histograms for correct and incorrect losses over the same bin range,
    and normalize them together so that their combined total equals 1.
    
    Args:
        losses_correct (np.array or torch.Tensor): Loss values for correct samples.
        losses_incorrect (np.array or torch.Tensor): Loss values for incorrect samples.
        num_bins (int): Number of bins to use.
        
    Returns:
        norm_correct, norm_incorrect (tuple of np.ndarray): Histograms for correct and incorrect losses,
            normalized such that (norm_correct + norm_incorrect).sum() == 1.
    """
    # Convert to numpy arrays.
    if torch.is_tensor(losses_correct):
        losses_correct = losses_correct.cpu().numpy().flatten()
    else:
        losses_correct = np.array(losses_correct).flatten()
    if torch.is_tensor(losses_incorrect):
        losses_incorrect = losses_incorrect.cpu().numpy().flatten()
    else:
        losses_incorrect = np.array(losses_incorrect).flatten()
    
    # Use combined losses to define common histogram bins.
    all_losses = np.concatenate([losses_correct, losses_incorrect])
    if all_losses.size == 0:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = all_losses.min(), all_losses.max()
    
    hist_correct, _ = np.histogram(losses_correct, bins=num_bins, range=(min_val, max_val), density=False)
    hist_incorrect, _ = np.histogram(losses_incorrect, bins=num_bins, range=(min_val, max_val), density=False)
    
    combined = hist_correct + hist_incorrect
    total = combined.sum()
    if total > 0:
        norm_correct = hist_correct / total
        norm_incorrect = hist_incorrect / total
    else:
        norm_correct = np.zeros_like(hist_correct)
        norm_incorrect = np.zeros_like(hist_incorrect)
    
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

def plot_histogram(hist, title, filename):
    """
    Plot a normalized histogram and save the figure.
    
    Args:
        hist (np.ndarray): The histogram bins.
        title (str): The title of the plot.
        filename (str): Path to save the figure.
    """
    plt.figure()
    plt.bar(np.arange(len(hist)), hist)
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Normalized Frequency")
    plt.savefig(filename)
    plt.close()
