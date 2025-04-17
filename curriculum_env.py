#==========================================
# File: curriculum_env.py
#==========================================

"""
Custom environment implementing the curriculum learning process with RL-based hyperparameter control.
Each episode follows these stages:
 1. Burn-In Phase: Collect loss data on the current model.
 2. Observation Construction: Compute six binned loss vectors and relative dataset sizes.
 3. RL Decision: The agent proposes hyperparameter ratios.
 4. Parameter Mapping & Curriculum Training: Map ratios to concrete hyperparameters and perform training.
 5. Reward Evaluation: Compute validation accuracy as the reward.
 
For simplicity, we assume each episode is terminal (i.e. one step per episode).
"""

import torch
import random
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from curriculum import burn_in_phase, run_phase_training, SimpleMLP, evaluate_accuracy, get_mixed_loader
from utils import construct_observation

class CurriculumEnv:
    def __init__(self, config):
        """
        Initialize the environment.
        
        Args:
            config (dict): Parsed configuration from the YAML file.
        """
        self.config = config
        self.device = torch.device(config["device"])
        self.batch_size = config["batch_size"]
        self.num_bins = config["observation"]["num_bins"]
        
        # Define transforms.
        mean = (0.1307,)
        std = (0.3081,)
        self.easy_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.medium_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.hard_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        
        # Prepare the base dataset (MNIST) and create curriculum splits.
        self.mnist_train = torchvision.datasets.MNIST(root=config["paths"]["data_path"], train=True, download=True, transform=T.ToTensor())
        self._create_curriculum_datasets()
        
        # Validation dataset.
        self.mnist_test = torchvision.datasets.MNIST(root=config["paths"]["data_path"], train=False, download=True, transform=T.ToTensor())
        self.val_loader = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)
        
        # Maximum total training samples (for scaling sample ratios).
        self.train_samples_max = config["curriculum"]["train_samples_max"]
        self.lr_range = config["curriculum"]["learning_rate_range"]  # [min_lr, max_lr]
        self.max_phases = config["curriculum"]["max_phases"]
        self.max_phases = config["curriculum"]["max_phases"]
        self.current_phase = 0
        self.remaining_samples = config["curriculum"]["train_samples_max"]

        
        # Initialize a fresh model.
        self._init_model()
        

    def _create_curriculum_datasets(self):
        """
        Splits the MNIST training set into easy, medium, and hard subsets based on provided fractions.
        """
        indices = list(range(len(self.mnist_train)))
        random.shuffle(indices)
        n_total = len(indices)
        n_easy = int(self.config["fractions"]["easy"] * n_total)
        n_medium = int(self.config["fractions"]["medium"] * n_total)
        n_hard = n_total - n_easy - n_medium
        
        self.easy_indices = indices[:n_easy]
        self.medium_indices = indices[n_easy:n_easy+n_medium]
        self.hard_indices = indices[n_easy+n_medium:]
        
        self.easy_ds = Subset(self.mnist_train, self.easy_indices)
        self.medium_ds = Subset(self.mnist_train, self.medium_indices)
        self.hard_ds = Subset(self.mnist_train, self.hard_indices)
        
        # Create DataLoaders for burn-in phase.
        self.easy_loader = DataLoader(self.easy_ds, batch_size=self.batch_size, shuffle=True)
        self.medium_loader = DataLoader(self.medium_ds, batch_size=self.batch_size, shuffle=True)
        self.hard_loader = DataLoader(self.hard_ds, batch_size=self.batch_size, shuffle=True)
    

    def _init_model(self):
        """
        Initialize (or reinitialize) the model.
        """
        self.model = SimpleMLP().to(self.device)
        

    def get_observation(self):
        """
        Compute the current observation from a burn-in phase and append extra state features.
        
        Extra features:
         - Phase completion ratio: current_phase / max_phases.
         - Available data ratio: remaining_samples / total initial samples.
        
        Returns:
            observation (np.ndarray): Extended observation vector.
        """
        ec_easy, ei_easy = burn_in_phase(self.model, self.easy_loader, self.device)
        ec_med, ei_med = burn_in_phase(self.model, self.medium_loader, self.device)
        ec_hard, ei_hard = burn_in_phase(self.model, self.hard_loader, self.device)
        
        easy_count = len(self.easy_ds)
        medium_count = len(self.medium_ds)
        hard_count = len(self.hard_ds)
        
        base_obs = construct_observation(ec_easy, ei_easy,
                                         ec_med, ei_med,
                                         ec_hard, ei_hard,
                                         easy_count, medium_count, hard_count,
                                         self.num_bins)
        phase_ratio = self.current_phase / self.max_phases
        available_ratio = self.remaining_samples / self.config["curriculum"]["train_samples_max"]
        extra = np.array([phase_ratio, available_ratio])
        return np.concatenate([base_obs, extra])


    def reset(self):
        """
        Resets the environment for a new overall episode.
        Reinitializes the model and resets phase counters and the remaining samples.
        
        Returns:
            observation (np.ndarray): The constructed observation vector including extra state features.
        """
        self._init_model()
        self.current_phase = 0
        self.remaining_samples = self.config["curriculum"]["train_samples_max"]
        return self.get_observation()


    def step(self, action):
        """
        Executes one training phase using the agent’s final action output.
        
        The action is a 5-dimensional vector:
        - [0]: learning rate (already in the proper range).
        - [1:4]: mixing ratio values that sum to 1.
        - [4]: sample usage fraction (between 0 and 1).
        
        This function maps the sample usage fraction to an actual sample count, runs one phase of training,
        updates internal counters, and returns the new observation, reward, and done flag.
        
        Args:
            action (np.ndarray): 5-dimensional action vector.
            
        Returns:
            next_obs (np.ndarray): New observation after phase training.
            reward (float): Validation accuracy from this phase.
            done (bool): True if all phases are completed or if no data remains.
        """
        # Clip learning rate to valid range before using it
        min_lr, max_lr = self.lr_range
        lr = float(np.clip(action[0], min_lr, max_lr))
        mixing_ratio = action[1:4]

        sample_frac = action[4]
        
        # Map the sample usage fraction to an actual sample count using the remaining samples.
        samples = int(sample_frac * self.remaining_samples)
        # Enforce a minimum sample count (defaulting to 100 or from config, if available)
        samples = max(samples, self.config["curriculum"].get("min_sample_usage", 100))
        
        hyperparams = {
            "training_samples": samples,
            "learning_rate": lr,
            "mixture_ratio": mixing_ratio.tolist(),
            "phase_batch_size": 512
        }
        
        # Run a single phase training routine.
        reward = run_phase_training(self.model,
                                    self.easy_loader,
                                    self.medium_loader,
                                    self.hard_loader,
                                    hyperparams,
                                    self.device)
        
        # Update internal counters.
        self.remaining_samples -= samples
        self.current_phase += 1
        done = (self.current_phase >= self.max_phases) or (self.remaining_samples <= 0)
        if done == True:
            reward = 10*reward
        next_obs = self.get_observation()
        return next_obs, reward, done
