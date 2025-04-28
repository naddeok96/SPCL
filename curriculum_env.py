"""
Custom environment implementing the curriculum learning process with RL-based
hyperparameter control, now with dynamic data splits and model architectures.
"""

import torch
import random
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import os
import torch.nn as nn

from utils import construct_observation
from curriculum import eval_loader, run_phase_training, evaluate_accuracy, get_mixed_loader


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


class CurriculumEnv:
    """
    Custom environment implementing the curriculum learning process with RL-based
    hyperparameter control, now with dynamic data splits and model architectures.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.batch_size = config["batch_size"]
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
        self.easy_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.medium_transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.hard_transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.RandomRotation(15),
            T.GaussianBlur(3),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        # Placeholder fractions (updated at reset)
        self.easy_frac = (self.easy_lower + self.easy_upper) / 2.0
        self.medium_frac = self.medium_lower

        # Prepare base dataset and initial splits
        self.mnist_train = torchvision.datasets.MNIST(
            root=config["paths"]["data_path"], train=True, download=True, transform=None
        )
        self._create_curriculum_datasets()

        # Validation loader
        self.mnist_test = torchvision.datasets.MNIST(
            root=config["paths"]["data_path"], train=False, download=True, transform=T.ToTensor()
        )
        self.val_loader = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

        # Hyperparams
        self.train_samples_max = config["curriculum"]["train_samples_max"]
        self.lr_range = config["curriculum"]["learning_rate_range"]
        self.max_phases = config["curriculum"]["max_phases"]
        self.current_phase = 0
        self.remaining_samples = self.train_samples_max

        # Initialize placeholder model (will be overridden)
        self._init_model()

        # Initial warm‑up loader (rebuilt each reset)
        full_ds = ConcatDataset([self.easy_ds, self.medium_ds, self.hard_ds])
        self._warmup_loader = DataLoader(
            full_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def _create_curriculum_datasets(self):
        """
        Split MNIST train into easy/medium/hard according to current fractions.
        """
        full_ds = torchvision.datasets.MNIST(
            root=self.config["paths"]["data_path"], train=True, download=True, transform=None
        )
        indices = list(range(len(full_ds)))
        random.shuffle(indices)
        n_total = len(indices)
        n_easy = int(self.easy_frac * n_total)
        n_medium = int(self.medium_frac * n_total)
        n_hard = n_total - n_easy - n_medium

        easy_idx = indices[:n_easy]
        medium_idx = indices[n_easy:n_easy+n_medium]
        hard_idx = indices[n_easy+n_medium:]

        easy_full = torchvision.datasets.MNIST(
            root=self.config["paths"]["data_path"], train=True, download=False, transform=self.easy_transform
        )
        medium_full = torchvision.datasets.MNIST(
            root=self.config["paths"]["data_path"], train=True, download=False, transform=self.medium_transform
        )
        hard_full = torchvision.datasets.MNIST(
            root=self.config["paths"]["data_path"], train=True, download=False, transform=self.hard_transform
        )

        self.easy_ds = Subset(easy_full, easy_idx)
        self.medium_ds = Subset(medium_full, medium_idx)
        self.hard_ds = Subset(hard_full, hard_idx)

        self.easy_loader = DataLoader(self.easy_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.medium_loader = DataLoader(self.medium_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.hard_loader = DataLoader(self.hard_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def _init_model(self):
        """
        Initialize or reinitialize the classification model based on sampled architecture.
        """
        if hasattr(self, "model_config"):
            cfg = self.model_config
            self.model = build_cnn_model(
                cfg["n_convs"], cfg["conv_ch"], cfg["n_fcs"],
                cfg["fc_units"], cfg["activation"], cfg["dropout"]
            ).to(self.device)
        else:
            # Default MLP fallback
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ).to(self.device)

    def get_observation(self):
        """
        Compute observation: six loss histograms + relative sizes + extra features.
        """
        ec_easy, ei_easy = eval_loader(self.model, self.easy_loader, self.device)
        ec_med, ei_med = eval_loader(self.model, self.medium_loader, self.device)
        ec_hard, ei_hard = eval_loader(self.model, self.hard_loader, self.device)

        easy_count = len(self.easy_ds)
        medium_count = len(self.medium_ds)
        hard_count = len(self.hard_ds)

        base_obs = construct_observation(
            ec_easy, ei_easy,
            ec_med, ei_med,
            ec_hard, ei_hard,
            easy_count, medium_count, hard_count,
            self.num_bins
        )
        phase_ratio = self.current_phase / self.max_phases
        available_ratio = self.remaining_samples / self.train_samples_max
        return np.concatenate([base_obs, np.array([phase_ratio, available_ratio])])

    def reset(self):
        """
        Reinitialize model, data splits, and warm‑up with new random fractions
        and architecture, then return initial observation.
        """
        # Sample new data fractions
        easy = random.uniform(self.easy_lower, self.easy_upper)
        max_med = min(easy, 1.0 - easy - self.hard_min)
        min_med = max(self.medium_lower, (1.0 - easy) / 2.0)
        if max_med <= min_med:
            medium = (min_med + max_med) / 2.0
        else:
            medium = random.uniform(min_med, max_med)
        self.easy_frac = easy
        self.medium_frac = medium

        # Rebuild datasets and warm‑up loader
        self._create_curriculum_datasets()
        full_ds = ConcatDataset([self.easy_ds, self.medium_ds, self.hard_ds])
        self._warmup_loader = DataLoader(full_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Sample new model architecture
        n_convs = random.choice(self.n_convs_choices)
        conv_ch = random.choice(self.conv_channels_choices)
        n_fcs = random.choice(self.n_fcs_choices)
        fc_units = random.choice(self.fc_units_choices)
        act_name = random.choice(self.activation_names)
        activation_cls = getattr(nn, act_name)
        dropout_rate = random.choice(self.dropout_rates)
        self.model_config = {
            "n_convs": n_convs,
            "conv_ch": conv_ch,
            "n_fcs": n_fcs,
            "fc_units": fc_units,
            "activation": activation_cls,
            "dropout": dropout_rate
        }

        # Initialize the new model
        self._init_model()

        # Reset counters
        self.current_phase = 0
        self.remaining_samples = self.train_samples_max

        # Warm‑up epoch
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=(sum(self.lr_range) / 2.0))
        criterion = torch.nn.CrossEntropyLoss()
        for imgs, labels in self._warmup_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return self.get_observation()

    def step(self, action):
        """
        Execute one curriculum phase with the given action.
        Returns: next_obs, reward, done.
        """
        min_lr, max_lr = self.lr_range
        lr = float(np.clip(action[0], min_lr, max_lr))
        mixing_ratio = action[1:4]
        sample_frac = action[4]

        samples = int(sample_frac * self.remaining_samples)

        hyperparams = {
            "training_samples": samples,
            "learning_rate": lr,
            "mixture_ratio": mixing_ratio.tolist(),
            "phase_batch_size": self.config["curriculum"].get("student_batch_size", 1024)
        }

        reward = run_phase_training(
            self.model,
            self.easy_loader,
            self.medium_loader,
            self.hard_loader,
            hyperparams,
            self.device
        )

        self.remaining_samples -= samples
        self.current_phase += 1
        done = (
            self.current_phase >= self.max_phases or
            self.remaining_samples <= 0 or
            sample_frac <= 0.0
        )
        if done:
            reward *= 10
        next_obs = self.get_observation()
        return next_obs, reward, done




