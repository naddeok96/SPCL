"""
Module that implements the curriculum training pipeline functions.
This module contains a simple burn-in phase for loss collection and a curriculum training routine 
that uses hyperparameters provided by the RL agent.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import _get_bin_edges, _MAX_LOSS

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

def eval_loader(model, loader, device, num_bins):
    """
    Compute per-sample correct/incorrect loss histograms on GPU.
    Returns (hist_correct, hist_incorrect), each length num_bins.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    hist_c = torch.zeros(num_bins, device=device)
    hist_i = torch.zeros(num_bins, device=device)

    # shared binning
    edges      = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    max_num_batches = int(len(loader) * 0.5)
    with torch.no_grad():
        for batch_num, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            losses  = ce_loss(outputs, labels)
            preds   = outputs.argmax(dim=1)

            # split correct vs incorrect
            lc = losses[preds == labels].clamp(0.0, _MAX_LOSS).flatten()
            li = losses[preds != labels].clamp(0.0, _MAX_LOSS).flatten()

            # bucketize + count
            bc = torch.bucketize(lc, boundaries)
            bi = torch.bucketize(li, boundaries)

            hist_c += torch.bincount(bc, minlength=num_bins).to(device, torch.float32)
            hist_i += torch.bincount(bi, minlength=num_bins).to(device, torch.float32)
            
            if batch_num > max_num_batches:
                break
            

    total = hist_c + hist_i
    S = total.sum()
    if S > 0:
        hist_c /= S
        hist_i /= S

    return hist_c, hist_i

def run_phase_training(model, easy_loader, medium_loader, hard_loader, hyperparams, device):
    """
    Runs a single phase of curriculum training using the provided hyperparameters.
    
    Hyperparameters dictionary (hyperparams) should contain:
        - 'training_samples': int, number of samples to use in this phase.
        - 'learning_rate': float, the learning rate for this phase.
        - 'mixture_ratio': list of 3 floats, the probability-like mixing ratios for [easy, medium, hard].
        - 'phase_batch_size': int, batch size for this phase.
    
    Args:
        model (nn.Module): The model to train.
        easy_loader, medium_loader, hard_loader (DataLoader): Data loaders for the easy, medium, and hard datasets.
        hyperparams (dict): Hyperparameters for this phase.
        device (torch.device): Computation device.
        
    Returns:
        reward (float): The macro accuracy achieved after phase training,
                        computed as the average of accuracies on the easy, medium, and hard datasets.
    """
    phase_batch_size = hyperparams.get("phase_batch_size", 1024)
    criterion = torch.nn.CrossEntropyLoss()

    # Create a mixed training loader based on the provided mixing ratios.
    current_mixture = hyperparams["mixture_ratio"]
    mixed_loader = get_mixed_loader(
        easy_loader.dataset,
        medium_loader.dataset,
        hard_loader.dataset,
        current_mixture,
        num_samples=hyperparams["training_samples"],
        batch_size=phase_batch_size
    )
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model.train()
    phase_samples = 0
    pbar = tqdm(mixed_loader, desc="Phase Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        phase_samples += imgs.size(0)
        pbar.set_postfix(loss=loss.item())
        if phase_samples >= hyperparams["training_samples"]:
            break

    # Evaluate accuracy on each curriculum subset and compute macro accuracy.
    easy_acc = evaluate_accuracy(model, easy_loader, device)
    med_acc = evaluate_accuracy(model, medium_loader, device)
    hard_acc = evaluate_accuracy(model, hard_loader, device)
    macro_acc = (easy_acc + med_acc + hard_acc) / 3.0

    return macro_acc

def run_curriculum_training(model, easy_loader, medium_loader, hard_loader, hyperparams, val_loader, device):
    """
    Runs a simplified curriculum training process using the hyperparameters
    output by the RL agent.
    
    Args:
        model (nn.Module): The model to train.
        easy_loader, medium_loader, hard_loader (DataLoader): Data loaders for each curriculum subset.
        hyperparams (dict): Dictionary with keys:
            - 'training_samples': list of training sample counts (one per phase)
            - 'learning_rates': list of learning rates (one per phase)
            - 'mixture_ratio': list of three lists (each with 3 values) for easy, medium, and hard mixing per phase.
            - 'phase_batch_size': optional batch size (default 512)
        val_loader (DataLoader): Validation loader.
        device (torch.device): Device to perform training.
        
    Returns:
        reward (float): The true macro accuracy achieved on the validation set.
    """
    phase_batch_size = hyperparams.get("phase_batch_size", 512)
    criterion = nn.CrossEntropyLoss()
    
    # For simplicity, run training for a fixed number of phases.
    for phase in range(3):
        # For the mixture ratios, use the set corresponding to the current phase.
        current_mixture = hyperparams["mixture_ratio"][phase]
        mixed_loader = get_mixed_loader(easy_loader.dataset,
                                        medium_loader.dataset,
                                        hard_loader.dataset,
                                        current_mixture,
                                        num_samples=hyperparams["training_samples"][phase],
                                        batch_size=phase_batch_size)
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rates"][phase])
        model.train()
        phase_samples = 0
        pbar = tqdm(mixed_loader, desc=f"Curriculum Phase {phase+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            phase_samples += imgs.size(0)
            pbar.set_postfix(loss=loss.item())
            if phase_samples >= hyperparams["training_samples"][phase]:
                break
            
    # After training, evaluate the model on the validation set.
    acc = evaluate_accuracy(model, val_loader, device)
    return acc

def get_mixed_loader(easy_ds, medium_ds, hard_ds, mixture, num_samples, batch_size=64):
    """
    Create a DataLoader that samples from the concatenation of the easy, medium, and hard datasets
    with weights given by the mixture ratios.
    
    Args:
        easy_ds, medium_ds, hard_ds (Dataset): The three datasets.
        mixture (list): List of three mixture ratios for [easy, medium, hard].
        num_samples (int): Number of samples to draw.
        batch_size (int): Batch size.
        
    Returns:
        DataLoader: The mixed DataLoader.
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
    concat_ds = ConcatDataset([easy_ds, medium_ds, hard_ds])
    weights = [mixture[0]] * len(easy_ds) + [mixture[1]] * len(medium_ds) + [mixture[2]] * len(hard_ds)
    sampler = WeightedRandomSampler(weights, num_samples=max(num_samples, 100), replacement=True)
    loader = DataLoader(concat_ds, batch_size=batch_size, sampler=sampler)
    return loader

def evaluate_accuracy(model, loader, device):
    """
    Evaluate model accuracy on the provided loader.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader.
        device (torch.device): Device to perform evaluation.
        
    Returns:
        accuracy (float): The classification accuracy in percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


















