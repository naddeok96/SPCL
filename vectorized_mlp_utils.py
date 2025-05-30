import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PaddedBatchedMLP(nn.Module):
    """Vectorized collection of MLPs with varying depths/widths."""
    def __init__(self, configs, global_idxs, input_dim, output_dim):
        super().__init__()
        self.configs = configs
        self.global_idxs = global_idxs
        self.num_models = len(configs)
        self.depths = [len(c) for c in configs]
        self.max_depth = max(self.depths)
        self.max_width = max(max(c) for c in configs) if configs else 1
        dims = [input_dim] + [self.max_width] * self.max_depth + [output_dim]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1], dims[i]))
            for i in range(len(dims)-1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_models, dims[i+1]))
            for i in range(len(dims)-1)
        ])
        seed_base = 1000
        for local_i, global_i in enumerate(global_idxs):
            full = [input_dim] + configs[local_i] + [output_dim]
            cnt = 0
            for li in range(len(full)-1):
                out, inp = full[li+1], full[li]
                base = seed_base + global_i*100 + cnt*2
                torch.manual_seed(base)
                w = torch.randn(out, inp) * 0.1
                torch.manual_seed(base + 1)
                b = torch.randn(out) * 0.1
                self.weights[li].data[local_i, :out, :inp] = w
                self.biases[li].data[local_i, :out] = b
                cnt += 1
        for li in range(self.max_depth):
            mask = torch.tensor([li < d for d in self.depths], dtype=torch.bool)
            self.register_buffer(f"mask_{li}", mask.view(-1,1,1))

    def forward(self, x):
        # x: [num_models, batch, input_dim]
        for li, (W, B) in enumerate(zip(self.weights, self.biases)):
            y = torch.bmm(W, x.transpose(1,2)).transpose(1,2) + B.unsqueeze(1)
            if li == self.max_depth:
                x = y
            else:
                if li == 0:
                    x = F.relu(y)
                else:
                    mask = getattr(self, f"mask_{li}")
                    x = torch.where(mask, F.relu(y), x)
        return x

class CombinedDataset(Dataset):
    """Stacks samples from multiple datasets along a new axis."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = min(len(ds) for ds in datasets)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        xs, ys = [], []
        for ds in self.datasets:
            x, y = ds[idx]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.tensor(ys)
