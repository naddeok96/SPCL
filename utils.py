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


def select_top_percent(dataset, percent):
    """Select transitions from the top ``percent``% highest-reward trajectories.

    Args:
        dataset (dict): Mapping with keys ``states``, ``actions``, ``rewards``,
            ``next_states`` and ``dones`` storing tensors for a set of episodes.
        percent (float): Value in ``[0, 100]`` determining how many of the best
            trajectories to keep based on summed episode reward.

    Returns:
        dict: Subset of ``dataset`` containing only the transitions belonging to
        the selected top trajectories.
    """
    rewards = dataset["rewards"].cpu()
    dones = dataset["dones"].cpu()

    ep_indices = []
    ep_returns = []
    start = 0
    ret = 0.0
    for i, r in enumerate(rewards):
        ret += float(r)
        if dones[i]:
            ep_indices.append((start, i + 1))
            ep_returns.append(ret)
            start = i + 1
            ret = 0.0
    if start < len(rewards):
        ep_indices.append((start, len(rewards)))
        ep_returns.append(ret)

    k = max(1, int(len(ep_returns) * percent / 100.0))
    top_idx = sorted(range(len(ep_returns)), key=lambda i: ep_returns[i], reverse=True)[:k]

    selected = []
    for idx in top_idx:
        s, e = ep_indices[idx]
        selected.extend(range(s, e))

    selected = torch.tensor(selected, dtype=torch.long)
    return {
        "states": dataset["states"][selected],
        "actions": dataset["actions"][selected],
        "rewards": dataset["rewards"][selected],
        "next_states": dataset["next_states"][selected],
        "dones": dataset["dones"][selected],
    }


def behavior_clone(policy, expert_data, epochs=5, batch_size=64, weight_decay=1e-4, device=None):
    """Fit ``policy`` to ``expert_data`` using supervised learning.

    The policy is trained with dropout active and L2 weight decay.  A simple
    mean-squared-error loss is minimized over ``epochs`` passes through the
    dataset.

    Args:
        policy (nn.Module): Policy network mapping states to actions.
        expert_data (dict): Dataset containing ``states`` and ``actions`` of the
            expert trajectories.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        weight_decay (float): Weight decay coefficient for the optimizer.
        device (torch.device or str, optional): Device for computation.  If not
            given, inferred from ``policy`` parameters.
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = device or next(policy.parameters()).device
    device = torch.device(device)

    ds = TensorDataset(expert_data["states"].to(device), expert_data["actions"].to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    policy.train()
    optimiser = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for s, a in loader:
            pred = policy(s)
            loss = loss_fn(pred, a)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()


def tune_value_function(policy, value_net, data_loader, config):
    """Train ``value_net`` off-policy with a CQL regularizer.

    The policy is frozen during this procedure.  Targets are computed using the
    current policy and discount factor ``gamma`` from ``config``.  A simplified
    conservative-Q (CQL) penalty discourages overestimation of unseen actions.

    Args:
        policy (nn.Module): Actor network.
        value_net (nn.Module): Critic/value function network.
        data_loader (DataLoader): Yields batches of ``(s, a, r, ns, d)``.
        config (dict): Configuration with keys ``gamma``, ``critic_lr`` and
            optional ``cql_alpha``.
    """
    import torch.nn.functional as F
    import torch.optim as optim

    device = next(value_net.parameters()).device
    gamma = config.get("gamma", 0.99)
    alpha = config.get("cql_alpha", 1.0)

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()

    optimiser = optim.Adam(value_net.parameters(), lr=config.get("critic_lr", 3e-4))

    for states, actions, rewards, next_states, dones in data_loader:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

        with torch.no_grad():
            next_a = policy(next_states)
            target_q = rewards + gamma * (1 - dones) * value_net(next_states, next_a)

        current_q = value_net(states, actions)
        td_loss = F.mse_loss(current_q, target_q)

        rand_a = torch.rand_like(actions)
        rand_a[:, 0:1] = rand_a[:, 0:1].clamp(*config.get("learning_rate_range", (0.0, 1.0)))
        rand_a[:, 1:4] = torch.softmax(rand_a[:, 1:4], dim=1)
        rand_a[:, 4:5] = rand_a[:, 4:5].clamp(0.0, 1.0)
        q_rand = value_net(states, rand_a)
        q_policy = value_net(states, policy(states))
        cql_loss = (torch.logsumexp(q_rand, dim=0).mean() - q_policy.mean()) * alpha

        loss = td_loss + cql_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()








