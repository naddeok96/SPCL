import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import csv
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
GPU = 5
NUM_MODELS = 8
MIN_DEPTH = 1
MAX_DEPTH = 4
MIN_WIDTH = 16
MAX_WIDTH = 128
OUTPUT_DIM = 10
EPOCHS = 1
BATCH_SIZE = 128
SEED_SHAPE = 0
SEED_INIT_BASE = 1000
OUT_DIR = "vec_exp_w_acc_temp"

# -----------------------------
# Setup
# -----------------------------
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED_SHAPE)
random.seed(SEED_SHAPE)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

os.makedirs(OUT_DIR, exist_ok=True)
seq_dir = os.path.join(OUT_DIR, "sequential")
vec_dir = os.path.join(OUT_DIR, "vectorized")
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(vec_dir, exist_ok=True)

# -----------------------------
# Load MNIST
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28×28 → 784
])
train_loader = DataLoader(
    datasets.MNIST(root='.', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)
INPUT_DIM = 784

# -----------------------------
# Generate model architectures
# -----------------------------
model_configs = []
for i in range(NUM_MODELS):
    depth = random.randint(MIN_DEPTH, MAX_DEPTH)
    widths = [random.randint(MIN_WIDTH, MAX_WIDTH) for _ in range(depth)]
    model_configs.append(widths)
with open(os.path.join(OUT_DIR, "model_configs.json"), "w") as f:
    json.dump(model_configs, f, indent=2)

# -----------------------------
# Helper: build & init a single MLP
# -----------------------------
def make_mlp(widths, model_idx):
    dims = [INPUT_DIM] + widths + [OUTPUT_DIM]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers)
    counter = 0
    for module in mlp:
        if isinstance(module, nn.Linear):
            base = SEED_INIT_BASE + model_idx * 100 + counter * 2
            torch.manual_seed(base)
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            torch.manual_seed(base + 1)
            nn.init.normal_(module.bias, mean=0.0, std=0.1)
            counter += 1
    return mlp

# -----------------------------
# Group models by their depth
# -----------------------------
depth_groups = defaultdict(list)
for idx, cfg in enumerate(model_configs):
    depth_groups[len(cfg)].append(idx)

# prepare containers for loss & accuracy
batched_losses     = [[] for _ in range(NUM_MODELS)]
batched_accuracies = [[] for _ in range(NUM_MODELS)]

# -----------------------------
# Padded, masked MLP for a depth group
# -----------------------------
class PaddedBatchedMLP(nn.Module):
    def __init__(self, configs, global_indices):
        super().__init__()
        self.configs = configs
        self.global_indices = global_indices
        self.num_models = len(configs)
        self.depths = [len(c) for c in configs]
        self.max_depth = max(self.depths)
        self.max_width = max(max(c) for c in configs)

        dims = [INPUT_DIM] + [self.max_width] * self.max_depth + [OUTPUT_DIM]

        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1], dims[i]))
            for i in range(len(dims)-1)
        ])
        self.biases  = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1]))
            for i in range(len(dims)-1)
        ])

        # initialize each sub-model exactly as in make_mlp
        for local_i, global_i in enumerate(global_indices):
            full = [INPUT_DIM] + configs[local_i] + [OUTPUT_DIM]
            cnt = 0
            for layer_i in range(len(full)-1):
                out, inp = full[layer_i+1], full[layer_i]
                base = SEED_INIT_BASE + global_i*100 + cnt*2
                torch.manual_seed(base)
                w = torch.randn(out, inp) * 0.1
                torch.manual_seed(base+1)
                b = torch.randn(out) * 0.1
                self.weights[layer_i].data[local_i, :out, :inp] = w
                self.biases[layer_i].data[local_i, :out]     = b
                cnt += 1

    def forward(self, x):
        x = x.unsqueeze(0).expand(self.num_models, -1, -1)
        for layer_i, (W, B) in enumerate(zip(self.weights, self.biases)):
            y = torch.bmm(W.to(x.device), x.transpose(1,2)).transpose(1,2) \
                + B.to(x.device).unsqueeze(1)
            is_output = (layer_i == self.max_depth)
            if is_output:
                x = y
            else:
                if layer_i == 0:
                    x = F.relu(y)
                else:
                    mask = torch.tensor([layer_i < d for d in self.depths],
                                        device=x.device).view(-1,1,1)
                    x = torch.where(mask, F.relu(y), x)
        return x  # [num_models, batch, OUTPUT_DIM]

# -----------------------------
# Grouped Vectorized Training
# -----------------------------
print("Starting grouped vectorized training...")
vec_start = time.time()
for depth, group_idxs in depth_groups.items():
    group_cfgs = [model_configs[i] for i in group_idxs]
    print(f"  → Depth={depth}, models={group_idxs}")
    vec_model = PaddedBatchedMLP(group_cfgs, group_idxs).to(device)
    opt = torch.optim.Adam(vec_model.parameters(), lr=1e-3)

    for xb, yb in tqdm(train_loader, desc=f"Depth {depth}"):
        xb = xb.to(device)
        # one-hot for loss
        y_onehot = F.one_hot(yb, num_classes=OUTPUT_DIM).float().to(device)
        # forward
        outs = vec_model(xb)  # [group_size, batch, OUTPUT_DIM]
        # compute per-model MSE loss
        loss_per_model = F.mse_loss(
            outs,
            y_onehot.unsqueeze(0).expand(len(group_idxs), -1, -1),
            reduction='none'
        ).mean(dim=2).mean(dim=1)
        total_loss = loss_per_model.sum()
        # backward
        total_loss.backward()
        opt.step()
        opt.zero_grad()

        # compute accuracy
        preds = outs.argmax(dim=2).cpu()  # [group_size, batch]
        for local_j, global_j in enumerate(group_idxs):
            # loss
            batched_losses[global_j].append(loss_per_model[local_j].item())
            # accuracy
            acc = (preds[local_j] == yb).float().mean().item()
            batched_accuracies[global_j].append(acc)

    # extract & save each individual model
    for local_j, global_j in enumerate(group_idxs):
        single = make_mlp(model_configs[global_j], global_j).to(device)
        li = 0
        for module in single:
            if isinstance(module, nn.Linear):
                w = vec_model.weights[li][local_j,
                                          :module.out_features,
                                          :module.in_features]
                b = vec_model.biases[li][local_j, :module.out_features]
                module.weight.data.copy_(w)
                module.bias.data.copy_(b)
                li += 1
        torch.save(single.state_dict(),
                   os.path.join(vec_dir, f"model_{global_j}.pt"))

vec_end = time.time()
print(f"Total grouped vectorized training time: {vec_end - vec_start:.3f} s")
with open(os.path.join(OUT_DIR, "total_vectorized_time.txt"), "w") as f:
    f.write(f"{vec_end - vec_start:.6f}\n")

# -----------------------------
# Sequential Training
# -----------------------------
print("Starting sequential training...")
sequential_losses     = [[] for _ in range(NUM_MODELS)]
sequential_accuracies = [[] for _ in range(NUM_MODELS)]
seq_start = time.time()
for idx, cfg in enumerate(tqdm(model_configs, desc="Sequential")):
    model = make_mlp(cfg, idx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_onehot = F.one_hot(y_batch, num_classes=OUTPUT_DIM).float().to(device)

        # forward & loss
        logits = model(x_batch)
        loss = F.mse_loss(logits, y_onehot, reduction='mean')
        loss.backward()
        opt.step()
        opt.zero_grad()

        # accuracy
        preds_seq = logits.argmax(dim=1)
        acc_seq = (preds_seq == y_batch).float().mean().item()

        sequential_losses[idx].append(loss.item())
        sequential_accuracies[idx].append(acc_seq)

    torch.save(model.state_dict(), os.path.join(seq_dir, f"model_{idx}.pt"))
    with open(os.path.join(seq_dir, f"loss_model_{idx}.csv"), "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["batch_idx", "loss"])
        writer.writerows(enumerate(sequential_losses[idx]))
    with open(os.path.join(seq_dir, f"acc_model_{idx}.csv"), "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["batch_idx", "accuracy"])
        writer.writerows(enumerate(sequential_accuracies[idx]))

seq_end = time.time()
print(f"Total sequential training time: {seq_end - seq_start:.3f} s")
with open(os.path.join(OUT_DIR, "total_sequential_time.txt"), "w") as f:
    f.write(f"{seq_end - seq_start:.6f}\n")

# -----------------------------
# Save and plot comparisons
# -----------------------------
for i in range(NUM_MODELS):
    # save vectorized metrics
    with open(os.path.join(vec_dir, f"loss_model_{i}.csv"), "w", newline="") as cf:
        csv.writer(cf).writerows(enumerate(batched_losses[i]))
    with open(os.path.join(vec_dir, f"acc_model_{i}.csv"), "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["batch_idx", "accuracy"])
        writer.writerows(enumerate(batched_accuracies[i]))

    # plot loss + accuracy in subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # Loss subplot
    axes[0].plot(sequential_losses[i],    label='Sequential Loss', marker='o')
    axes[0].plot(batched_losses[i],       label='Vectorized Loss',  marker='s')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    # Accuracy subplot
    axes[1].plot(sequential_accuracies[i],    label='Sequential Accuracy', marker='o')
    axes[1].plot(batched_accuracies[i],       label='Vectorized Accuracy',  marker='s')
    axes[1].set_xlabel('Batch Index')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"comparison_model_{i}.png"))
    plt.close(fig)

print(f"\n✅ All logs, weights, CSVs, and comparison plots saved under: {os.path.abspath(OUT_DIR)}")
