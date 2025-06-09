import torch
import random
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn

from curriculum import eval_loader, run_phase_training


def build_cnn_model(n_convs, conv_ch, n_fcs, fc_units, activation_cls, dropout_rate,
                    input_channels=1, input_size=28, num_classes=10):
    """
    Dynamically build a CNN with the given hyperparameters.
    """
    layers = []
    in_ch = input_channels
    out_size = input_size
    for _ in range(n_convs):
        layers.append(nn.Conv2d(in_ch, conv_ch, kernel_size=3, padding=1))
        layers.append(activation_cls())
        layers.append(nn.MaxPool2d(2))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        in_ch = conv_ch
        out_size //= 2
    layers.append(nn.Flatten())
    in_features = in_ch * out_size * out_size
    for _ in range(n_fcs):
        layers.append(nn.Linear(in_features, fc_units))
        layers.append(activation_cls())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_features = fc_units
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def build_mlp_model(hidden_layers, activation_cls, input_dim=28*28, num_classes=10):
    """Dynamically build an MLP with the given hidden layer sizes."""
    layers = [nn.Flatten()]
    in_features = input_dim
    for units in hidden_layers:
        layers.append(nn.Linear(in_features, units))
        layers.append(activation_cls())
        in_features = units
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


class CurriculumEnv:
    """
    Custom environment with cached DataLoaders and models,
    avoiding deep copies by resetting subsets in place.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.batch_size = config["curriculum"]["student_batch_size"]
        self.num_bins = config["observation"]["num_bins"]

        # Fraction bounds
        fr = config["fractions"]
        self.easy_lower = fr["easy_lower"]
        self.easy_upper = fr["easy_upper"]
        self.medium_lower = fr.get("medium_lower", 0.05)
        self.hard_min = fr.get("hard_min", 0.02)

        # Model search space
        ms = config["model_space"]
        self.n_convs_choices = ms["n_convs_choices"]
        self.conv_channels_choices = ms["conv_channels_choices"]
        self.n_fcs_choices = ms["n_fcs_choices"]
        self.fc_units_choices = ms["fc_units_choices"]
        self.activation_names = ms["activations"]
        self.dropout_rates = ms["dropout_rates"]

        # Transforms
        mean, std = (0.1307,), (0.3081,)
        self.easy_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.medium_transform = T.Compose([
            T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2,0.2,0.2),
            T.ToTensor(), T.Normalize(mean, std)
        ])
        self.hard_transform = T.Compose([
            T.RandomHorizontalFlip(0.5), T.ColorJitter(0.3,0.3,0.3),
            T.RandomRotation(15), T.GaussianBlur(3),
            T.ToTensor(), T.Normalize(mean, std)
        ])

        # Load base datasets
        data_path = config["paths"]["data_path"]
        self.full_easy_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.easy_transform)
        self.full_medium_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.medium_transform)
        self.full_hard_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=self.hard_transform)

        # Create empty Subsets
        self.easy_subset = Subset(self.full_easy_ds, [])
        self.medium_subset = Subset(self.full_medium_ds, [])
        self.hard_subset = Subset(self.full_hard_ds, [])

        # Hyperparams
        self.train_samples_max = config["curriculum"]["train_samples_max"]
        self.lr_range = config["curriculum"]["learning_rate_range"]
        self.max_phases = config["curriculum"]["max_phases"]

        # Initialize model cache
        self._init_model()

        # Now perform first reset to build loaders and warm-up
        self.reset()

    def _generate_splits(self):
        total = len(self.full_easy_ds)
        idxs = list(range(total))
        random.shuffle(idxs)
        n_easy = int(self.easy_frac * total)
        n_medium = int(self.medium_frac * total)
        return idxs[:n_easy], idxs[n_easy:n_easy+n_medium], idxs[n_easy+n_medium:]

    def _init_model(self):
        model_type = self.config.get("model_type", "cnn")
        if model_type == "mlp":
            if hasattr(self, "model_config"):
                cfg = self.model_config
                act = cfg.get("activation", nn.ReLU)
                self.model = build_mlp_model(cfg["hidden_layers"], act).to(self.device)
            else:
                self.model = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU(), nn.Linear(128,10)).to(self.device)
        else:
            if hasattr(self, "model_config"):
                cfg = self.model_config
                self.model = build_cnn_model(cfg["n_convs"], cfg["conv_ch"], cfg["n_fcs"],
                                             cfg["fc_units"], cfg["activation"], cfg["dropout"]).to(self.device)
            else:
                self.model = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU(), nn.Linear(128,10)).to(self.device)

    def get_observation(self):
        ec, ei = eval_loader(self.model, self.easy_loader,   self.device, self.num_bins)
        mc, mi = eval_loader(self.model, self.medium_loader, self.device, self.num_bins)
        hc, hi = eval_loader(self.model, self.hard_loader,   self.device, self.num_bins)

        counts = [len(self.easy_subset), len(self.medium_subset), len(self.hard_subset)]
        total = sum(counts)
        rel = torch.tensor([c/total for c in counts], device=self.device)

        obs = torch.cat([ec, ei, mc, mi, hc, hi, rel], dim=0)
        phase = torch.tensor(self.current_phase/self.max_phases, device=self.device).unsqueeze(0)
        avail = torch.tensor(self.remaining_samples/self.train_samples_max, device=self.device).unsqueeze(0)
        return torch.cat([obs, phase, avail], dim=0)

    def reset(self):
        # Sample fractions
        easy = random.uniform(self.easy_lower, self.easy_upper)
        max_med = min(easy, 1.0-easy-self.hard_min)
        min_med = max(self.medium_lower, (1.0-easy)/2)
        self.easy_frac = easy
        self.medium_frac = (min_med+max_med)/2 if max_med<=min_med else random.uniform(min_med, max_med)

        # Update subset indices
        e_idx, m_idx, h_idx = self._generate_splits()
        self.easy_subset.indices = e_idx
        self.medium_subset.indices = m_idx
        self.hard_subset.indices = h_idx

        # Build DataLoaders now that subsets are non-empty
        dl_args = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        self.easy_loader = DataLoader(self.easy_subset, **dl_args)
        self.medium_loader = DataLoader(self.medium_subset, **dl_args)
        self.hard_loader = DataLoader(self.hard_subset, **dl_args)
        self.warmup_loader = DataLoader(ConcatDataset([self.easy_subset, self.medium_subset, self.hard_subset]), **dl_args)

        model_type = self.config.get("model_type", "cnn")
        if model_type == "mlp":
            depth = random.randint(1, 3)
            widths = [random.randint(32, 128) for _ in range(depth)]
            self.model_config = {
                "hidden_layers": widths,
                "activation": getattr(nn, random.choice(self.activation_names))
            }
        else:
            self.model_config = {
                "n_convs": random.choice(self.n_convs_choices),
                "conv_ch": random.choice(self.conv_channels_choices),
                "n_fcs": random.choice(self.n_fcs_choices),
                "fc_units": random.choice(self.fc_units_choices),
                "activation": getattr(nn, random.choice(self.activation_names)),
                "dropout": random.choice(self.dropout_rates)
            }
        self._init_model()

        # Reset counters
        self.current_phase = 0
        self.remaining_samples = self.train_samples_max

        # Warm-up
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=sum(self.lr_range)/2)
        criterion = nn.CrossEntropyLoss()
        max_batches = max(1, int(0.25 * len(self.warmup_loader)))
        for i, (x,y) in enumerate(self.warmup_loader):
            if i>=max_batches: break
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad(); loss = criterion(self.model(x), y); loss.backward(); opt.step()

        return self.get_observation()

    def step(self, action):
        a = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.float32)
        lr, mix, frac = float(a[0]), a[1:4], float(a[4])
        num = int(frac * self.remaining_samples)
        hp = {"training_samples": num, "learning_rate": lr,
              "mixture_ratio": mix.tolist(), "phase_batch_size": self.batch_size}
        reward = run_phase_training(self.model, self.easy_loader, self.medium_loader, self.hard_loader, hp, self.device)
        self.remaining_samples -= num; self.current_phase += 1
        if self.current_phase>=self.max_phases or self.remaining_samples<=0 or frac<=0:
            reward *= 10; done = True
        else:
            done = False
        return self.get_observation(), reward, done
