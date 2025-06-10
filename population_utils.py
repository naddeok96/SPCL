# population_utils.py
#!/usr/bin/env python3

import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

from curriculum import get_mixed_loader
from curriculum_env import CurriculumEnv
from utils import _get_bin_edges, _MAX_LOSS

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def safe_normalize(arr: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a non-negative tensor so it sums to 1. If sum==0, returns uniform.
    """
    arr = torch.clamp(arr, min=0.0)
    total = arr.sum()
    if total.item() > 0.0:
        return arr / total
    return torch.full_like(arr, 1.0 / arr.numel())


def initialize_population(pop_size: int,
                          candidate_dim: int,
                          num_phases: int,
                          macro_actions: dict,
                          lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    population = []
    if macro_actions:
        macro_vals = [torch.tensor(v, dtype=torch.float32).flatten()
                      for v in macro_actions.values()]
        while len(population) < pop_size:
            base = random.choice(macro_vals).clone()
            if base.numel() != candidate_dim:
                base = base.repeat(num_phases)[:candidate_dim]
            cand = base + torch.randn(candidate_dim) * 0.1
            for p in range(num_phases):
                i = p * unit
                cand[i] = cand[i].clamp(lr_range[0], lr_range[1])
                cand[i+1:i+4] = safe_normalize(cand[i+1:i+4])
                cand[i+4] = cand[i+4].clamp(0.0, 1.0)
            population.append(cand)
    else:
        for _ in range(pop_size):
            cand = torch.zeros(candidate_dim)
            for p in range(num_phases):
                i = p * unit
                cand[i] = random.uniform(*lr_range)
                cand[i+1:i+4] = safe_normalize(torch.rand(3))
                cand[i+4] = random.random()
            population.append(cand)
    return torch.stack(population)

def mutate(child: torch.Tensor,
           mutation_rate: float,
           candidate_dim: int,
           num_phases: int,
           lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    m = child + torch.randn_like(child) * mutation_rate
    for p in range(num_phases):
        i = p * unit
        m[i] = m[i].clamp(lr_range[0], lr_range[1])
        m[i+1:i+4] = safe_normalize(m[i+1:i+4])
        m[i+4] = m[i+4].clamp(0.0, 1.0)
    return m

def crossover(p1: torch.Tensor,
              p2: torch.Tensor,
              candidate_dim: int,
              num_phases: int,
              lr_range: tuple) -> torch.Tensor:
    unit = candidate_dim // num_phases
    mask = torch.rand(candidate_dim) < 0.5
    
    child = p1.clone()
    child[mask] = p2[mask]
    for p in range(num_phases):
        i = p * unit
        child[i] = child[i].clamp(lr_range[0], lr_range[1])
        child[i+1:i+4] = safe_normalize(child[i+1:i+4])
        child[i+4] = child[i+4].clamp(0.0, 1.0)
    return child

def evaluate_candidate(env: CurriculumEnv,
                       candidate: torch.Tensor,
                       candidate_dim: int,
                       num_phases: int):
    """
    Evaluate a multi-phase candidate, with a tqdm over phases.
    Returns transitions list and total reward.
    """
    transitions = []
    total_reward = 0.0
    print("Initializing State")
    state = env.reset()
    print("State Initialized")
    unit = candidate_dim // num_phases
    done = False
    for p in trange(num_phases, desc="Eval phases", unit="phase", leave=False):
        action = candidate[state.new_zeros(())] if False else None  # placeholder
        
        idx = len(transitions) * unit
        action = candidate[idx:idx+unit]
        
        nxt, r, done = env.step(action)
        transitions.append((state, action, r, nxt, done))
        total_reward += r
        state = nxt
        if done:
            break

    return transitions, total_reward


class PaddedBatchedMLP(nn.Module):
    """Minimal padded MLP supporting a batch of differently-sized networks."""

    def __init__(self, configs, init_models=None, input_dim=28*28, output_dim=10):
        super().__init__()
        self.configs = configs
        self.num_models = len(configs)
        self.depths = [len(c) for c in configs]
        self.max_depth = max(self.depths)
        self.max_width = max(max(c) for c in configs)

        dims = [input_dim] + [self.max_width]*self.max_depth + [output_dim]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1], dims[i]))
            for i in range(len(dims)-1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1]))
            for i in range(len(dims)-1)
        ])

        # either copy from provided models or random init
        if init_models:
            for mi, model in enumerate(init_models):
                li = 0
                for m in model:
                    if isinstance(m, nn.Linear):
                        out, inp = m.weight.size()
                        self.weights[li].data[mi, :out, :inp].copy_(m.weight.data)
                        self.biases[li].data[mi, :out].copy_(m.bias.data)
                        li += 1
        else:
            for mi, cfg in enumerate(configs):
                full = [input_dim] + cfg + [output_dim]
                for li in range(len(full)-1):
                    out, inp = full[li+1], full[li]
                    self.weights[li].data[mi, :out, :inp].normal_(0.0, 0.1)
                    self.biases[li].data[mi, :out].normal_(0.0, 0.1)

        for li in range(self.max_depth):
            mask = torch.tensor([li < d for d in self.depths], dtype=torch.bool)
            self.register_buffer(f"mask_{li}", mask.view(-1,1,1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0).expand(self.num_models, -1, -1)
        for li, (W, B) in enumerate(zip(self.weights, self.biases)):
            y = torch.bmm(W.to(x.device), x.transpose(1,2)).transpose(1,2) + B.to(x.device).unsqueeze(1)
            if li == self.max_depth:
                x = y
            else:
                if li == 0:
                    x = F.relu(y)
                else:
                    mask = getattr(self, f"mask_{li}")
                    x = torch.where(mask, F.relu(y), x)
        return x


def _run_phase_training_batched(model, easy_loader, medium_loader, hard_loader, hp, device):
    """Train a PaddedBatchedMLP for one phase and return per-model rewards."""
    phase_batch_size = hp.get("phase_batch_size", 1024)
    criterion = nn.CrossEntropyLoss(reduction="none")

    mixed = get_mixed_loader(
        easy_loader.dataset,
        medium_loader.dataset,
        hard_loader.dataset,
        hp["mixture_ratio"],
        num_samples=hp["training_samples"],
        batch_size=phase_batch_size,
    )

    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    model.train()
    phase_samples = 0
    for imgs, labels in mixed:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        outs = model(imgs)
        lbl = labels.expand(model.num_models, -1)
        losses = criterion(outs.view(-1, outs.size(-1)), lbl.reshape(-1))
        loss_pm = losses.view(model.num_models, -1).mean(dim=1)
        loss_pm.sum().backward()
        opt.step()
        phase_samples += imgs.size(0)
        if phase_samples >= hp["training_samples"]:
            break

    def acc(loader):
        correct = torch.zeros(model.num_models, device=device)
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).argmax(dim=2)
                correct += (preds == yb).sum(dim=1).to(correct.dtype)
                total += yb.size(0)
        return correct / total * 100.0

    ea = acc(easy_loader)
    ma = acc(medium_loader)
    ha = acc(hard_loader)
    return ((ea + ma + ha) / 3.0).tolist()


def eval_loader_batched(model, loader, device, num_bins):
    """Vectorized ``eval_loader`` for ``PaddedBatchedMLP``.

    The normalization step divides each histogram row by the total count for
    that row.  ``totals.sum`` returns a ``(N, 1)`` tensor, so we squeeze the
    trailing dimension before using it as a boolean mask to select the rows to
    normalize.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    hist_c = torch.zeros(model.num_models, num_bins, device=device)
    hist_i = torch.zeros(model.num_models, num_bins, device=device)

    edges = _get_bin_edges(num_bins, device)
    boundaries = edges[1:-1]
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    max_batches = int(len(loader) * 0.5)
    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outs = model(imgs)  # [num_models, batch, C]
            lbl = labels.expand(model.num_models, -1)
            losses = ce_loss(outs.reshape(-1, outs.size(-1)), lbl.reshape(-1))
            losses = losses.reshape(model.num_models, -1)
            preds = outs.argmax(dim=2)
            correct = preds.eq(labels)
            for mi in range(model.num_models):
                lc = losses[mi][correct[mi]].clamp(0.0, _MAX_LOSS)
                li = losses[mi][~correct[mi]].clamp(0.0, _MAX_LOSS)
                bc = torch.bucketize(lc, boundaries)
                bi_idx = torch.bucketize(li, boundaries)
                hist_c[mi] += torch.bincount(bc, minlength=num_bins).float()
                hist_i[mi] += torch.bincount(bi_idx, minlength=num_bins).float()
            if bi > max_batches:
                break

    totals = hist_c + hist_i
    S = totals.sum(dim=1, keepdim=True).clamp(min=1.0)
    hist_c /= S
    hist_i /= S
    return hist_c, hist_i


def evaluate_candidate_parallel(cfg: dict,
                                candidate: torch.Tensor,
                                candidate_dim: int,
                                num_phases: int,
                                num_models: int):
    """Evaluate a candidate using multiple random MLP models in parallel."""
    base_env = CurriculumEnv(cfg)
    base_env.reset()

    model_cfgs = []
    for _ in range(num_models):
        depth = random.randint(1, 3)
        widths = [random.randint(32, 128) for _ in range(depth)]
        model_cfgs.append(widths)

    depth_groups = {}
    for idx, cfgs in enumerate(model_cfgs):
        depth_groups.setdefault(len(cfgs), []).append(idx)

    group_models = {
        d: PaddedBatchedMLP([model_cfgs[i] for i in idxs]).to(base_env.device)
        for d, idxs in depth_groups.items()
    }

    def get_obs_group(vec_model):
        ec, ei = eval_loader_batched(vec_model, base_env.easy_loader, base_env.device, base_env.num_bins)
        mc, mi = eval_loader_batched(vec_model, base_env.medium_loader, base_env.device, base_env.num_bins)
        hc, hi = eval_loader_batched(vec_model, base_env.hard_loader, base_env.device, base_env.num_bins)
        counts = [len(base_env.easy_subset), len(base_env.medium_subset), len(base_env.hard_subset)]
        total = sum(counts)
        rel = torch.tensor([c/total for c in counts], device=base_env.device).view(1,-1)
        rel = rel.expand(vec_model.num_models, -1)
        obs = torch.cat([ec, ei, mc, mi, hc, hi, rel], dim=1)
        phase = torch.full((vec_model.num_models,1), base_env.current_phase/base_env.max_phases, device=base_env.device)
        avail = torch.full((vec_model.num_models,1), base_env.remaining_samples/base_env.train_samples_max, device=base_env.device)
        return torch.cat([obs, phase, avail], dim=1)

    states = [None for _ in range(num_models)]
    for d, idxs in depth_groups.items():
        obs = get_obs_group(group_models[d])
        for li, gi in enumerate(idxs):
            states[gi] = obs[li]

    unit = candidate_dim // num_phases
    remaining = base_env.train_samples_max
    total_rewards = [0.0 for _ in range(num_models)]
    transitions = []

    done = False
    for p in range(num_phases):
        idx = p * unit
        action = candidate[idx:idx+unit]
        lr, mix, frac = float(action[0]), action[1:4], float(action[4])
        num = int(frac * remaining)
        hp = {"training_samples": num, "learning_rate": lr,
              "mixture_ratio": mix.tolist(), "phase_batch_size": base_env.batch_size}

        rewards = [0.0 for _ in range(num_models)]
        for d, idxs in depth_groups.items():
            vec = group_models[d]
            r = _run_phase_training_batched(vec, base_env.easy_loader,
                                            base_env.medium_loader,
                                            base_env.hard_loader,
                                            hp, base_env.device)
            for li, gi in enumerate(idxs):
                rewards[gi] = r[li]

        remaining -= num
        base_env.remaining_samples = remaining
        base_env.current_phase += 1
        done = (base_env.current_phase >= base_env.max_phases) or remaining <= 0 or frac <= 0

        next_states = [None for _ in range(num_models)]
        for d, idxs in depth_groups.items():
            obs = get_obs_group(group_models[d])
            for li, gi in enumerate(idxs):
                next_states[gi] = obs[li]

        for j in range(num_models):
            transitions.append((states[j], action, rewards[j], next_states[j], done))
            total_rewards[j] += rewards[j]

        states = next_states
        if done:
            break

    return transitions, sum(total_rewards) / num_models




