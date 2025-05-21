import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
GPU             = 4
NUM_MODELS      = 500
MIN_DEPTH       = 1
MAX_DEPTH       = 3
MIN_WIDTH       = 16
MAX_WIDTH       = 512
OUTPUT_DIM      = 10
EPOCHS          = 1
BATCH_SIZE      = 2048
SEED_SHAPE      = 0
SEED_INIT_BASE  = 1000
OUT_DIR         = "vec_seq_full_experiment"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Device & Seeding
# -----------------------------
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(SEED_SHAPE)
random.seed(SEED_SHAPE)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# -----------------------------
# Transform pools
# -----------------------------
pil_transform_factories = [
    lambda: transforms.RandomRotation(degrees=30),
    lambda: transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    lambda: transforms.RandomHorizontalFlip(),
    lambda: transforms.RandomVerticalFlip(),
    lambda: transforms.ColorJitter(brightness=0.5, contrast=0.5),
    lambda: transforms.RandomResizedCrop(size=28, scale=(0.8,1.0)),
    lambda: transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
]
tensor_transform_factories = [
    lambda: transforms.RandomErasing(scale=(0.02,0.33), ratio=(0.3,3.3))
]

# -----------------------------
# Prepare per-model datasets & loaders
# -----------------------------
ds_list     = []
loader_list = []
for i in range(NUM_MODELS):
    random.seed(SEED_SHAPE + i)
    n_pil = random.randint(1, len(pil_transform_factories))
    selected_pil = random.sample(pil_transform_factories, k=n_pil)
    pil_transforms = [fn() for fn in selected_pil]

    n_tensor = random.randint(0, len(tensor_transform_factories))
    selected_tensor = random.sample(tensor_transform_factories, k=n_tensor)
    tensor_transforms = [fn() for fn in selected_tensor]

    pipeline = transforms.Compose(
        pil_transforms +
        [transforms.ToTensor()] +
        tensor_transforms +
        [transforms.Lambda(lambda x: x.view(-1))]
    )

    ds = datasets.MNIST(root='.', train=True, download=True, transform=pipeline)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    ds_list.append(ds)
    loader_list.append(dl)

INPUT_DIM = 784

# -----------------------------
# Generate & save model architectures
# -----------------------------
model_configs = []
for i in range(NUM_MODELS):
    depth  = random.randint(MIN_DEPTH, MAX_DEPTH)
    widths = [random.randint(MIN_WIDTH, MAX_WIDTH) for _ in range(depth)]
    model_configs.append(widths)
with open(os.path.join(OUT_DIR, "model_configs.json"), "w") as f:
    json.dump(model_configs, f, indent=2)

# -----------------------------
# Build depth_groups
# -----------------------------
depth_groups = defaultdict(list)
for idx, cfg in enumerate(model_configs):
    depth_groups[len(cfg)].append(idx)

# -----------------------------
# Helper: single MLP builder
# -----------------------------
def make_mlp(widths, model_idx):
    dims   = [INPUT_DIM] + widths + [OUTPUT_DIM]
    layers = []
    for j in range(len(dims)-1):
        layers.append(nn.Linear(dims[j], dims[j+1]))
        if j < len(dims)-2:
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers)

    cnt = 0
    for m in mlp:
        if isinstance(m, nn.Linear):
            base = SEED_INIT_BASE + model_idx*100 + cnt*2
            torch.manual_seed(base)
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.manual_seed(base+1)
            nn.init.normal_(m.bias,   mean=0.0, std=0.1)
            cnt += 1
    return mlp

# -----------------------------
# CombinedDataset for vectorized loading
# -----------------------------
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length   = min(len(ds) for ds in datasets)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        xs, ys = [], []
        for ds in self.datasets:
            x, y = ds[idx]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.tensor(ys)

# -----------------------------
# PaddedBatchedMLP for vectorized models
# -----------------------------
class PaddedBatchedMLP(nn.Module):
    def __init__(self, configs, global_idxs):
        super().__init__()
        self.configs     = configs
        self.global_idxs = global_idxs
        self.num_models  = len(configs)
        self.depths      = [len(c) for c in configs]
        self.max_depth   = max(self.depths)
        self.max_width   = max(max(c) for c in configs)

        dims = [INPUT_DIM] + [self.max_width]*self.max_depth + [OUTPUT_DIM]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1], dims[i]))
            for i in range(len(dims)-1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1]))
            for i in range(len(dims)-1)
        ])

        # initialize each sub-model slice
        for local_i, global_i in enumerate(global_idxs):
            full = [INPUT_DIM] + configs[local_i] + [OUTPUT_DIM]
            cnt  = 0
            for li in range(len(full)-1):
                out, inp = full[li+1], full[li]
                base = SEED_INIT_BASE + global_i*100 + cnt*2
                torch.manual_seed(base)
                w = torch.randn(out, inp)*0.1
                torch.manual_seed(base+1)
                b = torch.randn(out)*0.1
                self.weights[li].data[local_i, :out, :inp] = w
                self.biases[li].data[local_i, :out]        = b
                cnt += 1

        # register masks as buffers (will move to GPU with module)
        for li in range(self.max_depth):
            mask = torch.tensor([li < d for d in self.depths], dtype=torch.bool)
            mask = mask.view(-1,1,1)
            self.register_buffer(f"mask_{li}", mask)

    def forward(self, x):
        # x: [num_models, batch, INPUT_DIM]
        for li, (W, B) in enumerate(zip(self.weights, self.biases)):
            # W, B already on x.device
            y = torch.bmm(W, x.transpose(1,2)).transpose(1,2) + B.unsqueeze(1)

            if li == self.max_depth:
                x = y
            else:
                if li == 0:
                    x = F.relu(y)
                else:
                    # fetch buffer mask_{li} (now on correct device)
                    mask = getattr(self, f"mask_{li}")
                    x = torch.where(mask, F.relu(y), x)
        return x  # [num_models, batch, OUTPUT_DIM]


# -----------------------------
# Sample 5 models for tracking
# -----------------------------
random.seed(SEED_SHAPE)
sample_ids = random.sample(range(NUM_MODELS), k=5)
print(f"Tracking models: {sample_ids}")

# -----------------------------
# Metric containers for sampled models
# -----------------------------
seq_losses = {i: [] for i in sample_ids}
seq_accs   = {i: [] for i in sample_ids}
vec_losses = {i: [] for i in sample_ids}
vec_accs   = {i: [] for i in sample_ids}

# -----------------------------
# Precompute combined datasets and loaders for each depth group
# -----------------------------
combined_ds_dict = {}
loader_vec_dict = {}
for depth, group_idxs in depth_groups.items():
    combined_ds = CombinedDataset([ds_list[i] for i in group_idxs])
    loader_vec = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    combined_ds_dict[depth] = combined_ds
    loader_vec_dict[depth] = loader_vec
    
# -----------------------------------------
# Precompute, per depth, the sample‐positions
# -----------------------------------------
# sample_ids = your list of 5 global model‐indices
sample_pos = {}
for depth, group_idxs in depth_groups.items():
    # local indices within this group where the global id ∈ sample_ids
    local_idxs  = [i for i, gi in enumerate(group_idxs) if gi in sample_ids]
    # corresponding global ids
    global_ids  = [gi for gi in group_idxs if gi in sample_ids]
    sample_pos[depth] = (local_idxs, global_ids)

# -----------------------------
# Vectorized training
# -----------------------------

timings = {
    "load+transfer": [],
    "forward": [],
    "backward": [],
    "optimizer": [],
    "metrics": [],
}

print("=== Vectorized training ===")
vec_start = time.time()

for depth, group_idxs in depth_groups.items():
    print(f"Depth {depth} → models {group_idxs}")
    cfgs       = [model_configs[i] for i in group_idxs]
    model_vec  = PaddedBatchedMLP(cfgs, group_idxs).to(device)
    opt_vec    = torch.optim.Adam(model_vec.parameters(), lr=1e-3)
    loader_vec = loader_vec_dict[depth]
    local_idxs, global_ids = sample_pos[depth]

    # per‐batch timers
    for xbs, ybs in tqdm(loader_vec, desc=f"Depth {depth}"):
        t0 = time.perf_counter()
        # ─── load & transfer ─────────────────────────────────────────────
        xbs = xbs.permute(1,0,2).to(device, non_blocking=True)
        ybs = ybs.permute(1,0).to(device, non_blocking=True)
        t1 = time.perf_counter()

        # ─── forward ──────────────────────────────────────────────────────
        yoh    = F.one_hot(ybs, num_classes=OUTPUT_DIM).float()
        outs   = model_vec(xbs)
        loss_pm = (
            F.mse_loss(outs, yoh, reduction='none')
             .mean(dim=2)
             .mean(dim=1)
        )
        t2 = time.perf_counter()

        # ─── backward ─────────────────────────────────────────────────────
        loss_pm.sum().backward()
        t3 = time.perf_counter()

        # ─── optimizer step ───────────────────────────────────────────────
        opt_vec.step()
        opt_vec.zero_grad()
        t4 = time.perf_counter()

        # ─── metrics ──────────────────────────────────────────────────────
        preds = outs.argmax(dim=2)
        if local_idxs:
            # slice out only the k models you track:
            outs_s = outs[local_idxs]               # [k, batch, OUTPUT_DIM]
            ybs_s  = ybs[local_idxs]                # [k, batch]

            # argmax & accuracy just on those k models:
            preds_s       = outs_s.argmax(dim=2)    # [k, batch]
            sample_losses = loss_pm[local_idxs].cpu().tolist()
            sample_accs   = ((preds_s == ybs_s)
                                .float()
                                .mean(dim=1)
                                .cpu()
                                .tolist())

            for glb, lss, acc in zip(global_ids, sample_losses, sample_accs):
                vec_losses[glb].append(lss)
                vec_accs[glb].append(acc)
                
        t5 = time.perf_counter()

        # record
        timings["load+transfer"].append(t1 - t0)
        timings["forward"].append(t2 - t1)
        timings["backward"].append(t3 - t2)
        timings["optimizer"].append(t4 - t3)
        timings["metrics"].append(t5 - t4)
        


vec_time = time.time() - vec_start
print(f"Vectorized training time: {vec_time:.2f}s")
with open(os.path.join(OUT_DIR, "time_vectorized.txt"), "w") as f:
    f.write(f"{vec_time:.6f}\n")

print("\nAverage times per batch:")
for k, v in timings.items():
    print(f"  {k:15s}: {sum(v)/len(v):8.4f}s  (n={len(v)})")
    
# -----------------------------
# Sequential training
# -----------------------------
print("=== Sequential training ===")
seq_start = time.time()

for idx, cfg in enumerate(model_configs):
    print(f"Model {idx}")
    mlp_seq = make_mlp(cfg, idx).to(device)
    opt_seq = torch.optim.Adam(mlp_seq.parameters(), lr=1e-3)

    for xb, yb in tqdm(loader_list[idx], desc=f"Model {idx}"):
        xb  = xb.to(device)
        yb  = yb.to(device)
        yoh = F.one_hot(yb, OUTPUT_DIM).float()

        logits = mlp_seq(xb)
        loss   = F.mse_loss(logits, yoh)
        loss.backward()
        opt_seq.step()
        opt_seq.zero_grad()

        pred = logits.argmax(dim=1)
        if idx in sample_ids:
            seq_losses[idx].append(loss.item())
            seq_accs[idx].append((pred == yb).float().mean().item())

seq_time = time.time() - seq_start
print(f"Sequential training time: {seq_time:.2f}s")
with open(os.path.join(OUT_DIR, "time_sequential.txt"), "w") as f:
    f.write(f"{seq_time:.6f}\n")

# -----------------------------
# Speedup factor
# -----------------------------
print(f"Speedup factor: {seq_time/vec_time:.2f}×")

# -----------------------------
# Plot comparisons for sampled models
# -----------------------------
for i in sample_ids:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), sharex=True)
    ax1.plot(seq_losses[i], label="Sequential Loss", marker='o')
    ax1.plot(vec_losses[i], label="Vectorized Loss", marker='s')
    ax1.set_ylabel("MSE Loss")
    ax1.legend()

    ax2.plot(seq_accs[i], label="Sequential Acc", marker='o')
    ax2.plot(vec_accs[i], label="Vectorized Acc", marker='s')
    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"model_{i}_comparison.png"))
    plt.close(fig)

print(f"\n✅ All timing logs and comparison plots for models {sample_ids} are in: {os.path.abspath(OUT_DIR)}")
