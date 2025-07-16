import os
import argparse
import yaml
import numpy as np
import torch
import random
from tqdm import trange

from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler

from curriculum_env import CurriculumEnv
from curriculum_env import build_cnn_model, build_mlp_model
from curriculum import evaluate_accuracy, run_phase_training
from rl_agent import DDPGAgent


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rollout_episode(agent, env, gamma, reset_env=True):
    """Run one episode and return total reward and (Q, return) pairs."""
    state = env.reset() if reset_env else env.get_observation()
    done = False
    states, actions, rewards = [], [], []
    while not done:
        action = agent.select_action(state, noise_enable=False)
        states.append(state)
        actions.append(action)
        state, reward, done = env.step(action)
        if done:
            # print("Just looking at final reward")
            rewards.append(reward/10)
    total_reward = sum(rewards)
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    s_t = torch.stack([s.to(agent.device) if torch.is_tensor(s) else torch.tensor(s, dtype=torch.float32, device=agent.device) for s in states])
    a_t = torch.stack(actions).to(agent.device)
    with torch.no_grad():
        q_pred = agent.critic1(s_t, a_t).squeeze(1).cpu().numpy()
    pairs = list(zip(q_pred, returns))
    return total_reward, pairs


def evaluate_agent(agent, env, episodes=100, gamma=0.99):
    rewards = []
    preds, targets = [], []
    for _ in trange(episodes, desc="Eval episodes", leave=False):
        r, pairs = rollout_episode(agent, env, gamma)
        rewards.append(r)
        for q, ret in pairs:
            preds.append(q)
            targets.append(ret)
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    mse = float(np.mean((np.array(preds) - np.array(targets)) ** 2)) if preds else 0.0
    return avg_reward, mse


def evaluate_single_episode(agent, env, gamma=0.99):
    r, pairs = rollout_episode(agent, env, gamma, reset_env=False)
    preds, targets = zip(*pairs) if pairs else ([], [])
    mse = float(np.mean((np.array(preds) - np.array(targets)) ** 2)) if preds else 0.0
    return r, mse


def train_standard_model(env, total_samples, lr):
    """Train a model on MNIST without curriculum for a fixed number of samples."""
    from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
    import torch.nn as nn

    device = env.device
    model_cfg = env.model_config
    model_type = env.config.get("model_type", "cnn")
    if model_type == "mlp":
        model = build_mlp_model(model_cfg["hidden_layers"], model_cfg["activation"]).to(device)
    else:
        model = build_cnn_model(
            model_cfg["n_convs"],
            model_cfg["conv_ch"],
            model_cfg["n_fcs"],
            model_cfg["fc_units"],
            model_cfg["activation"],
            model_cfg["dropout"],
        ).to(device)

    dataset = ConcatDataset([
        env.full_easy_ds,
        env.full_medium_ds,
        env.full_hard_ds,
    ])
    weights = [1 / 3] * len(env.full_easy_ds) + [1 / 3] * len(env.full_medium_ds) + [1 / 3] * len(env.full_hard_ds)
    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=env.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    samples = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimiser.step()
        samples += imgs.size(0)
        if samples >= total_samples:
            break

    easy = evaluate_accuracy(model, env.easy_loader, device)
    med = evaluate_accuracy(model, env.medium_loader, device)
    hard = evaluate_accuracy(model, env.hard_loader, device)
    return (easy + med + hard) / 3.0


def prepare_dataset(cfg, easy_frac=0.9, med_frac=0.075, seed=0):
    """Create a fixed dataset split for all experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    env = CurriculumEnv(cfg)
    env.reset(easy_frac, med_frac)

    dl_args = dict(
        batch_size=env.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    easy_ds = Subset(env.full_easy_ds, list(env.easy_subset.indices))
    med_ds = Subset(env.full_medium_ds, list(env.medium_subset.indices))
    hard_ds = Subset(env.full_hard_ds, list(env.hard_subset.indices))

    easy_loader = DataLoader(easy_ds, **dl_args)
    med_loader = DataLoader(med_ds, **dl_args)
    hard_loader = DataLoader(hard_ds, **dl_args)

    return {
        "device": env.device,
        "batch_size": env.batch_size,
        "model_cfg": env.model_config,
        "easy_loader": easy_loader,
        "med_loader": med_loader,
        "hard_loader": hard_loader,
        "easy_ds": easy_ds,
        "med_ds": med_ds,
        "hard_ds": hard_ds,
    }


def build_model(cfg, model_cfg, device, seed=0):
    """Construct a fresh model with controlled initialization."""
    torch.manual_seed(seed)
    model_type = cfg.get("model_type", "cnn")
    if model_type == "mlp":
        return build_mlp_model(model_cfg["hidden_layers"], model_cfg["activation"]).to(device)
    return build_cnn_model(
        model_cfg["n_convs"],
        model_cfg["conv_ch"],
        model_cfg["n_fcs"],
        model_cfg["fc_units"],
        model_cfg["activation"],
        model_cfg["dropout"],
    ).to(device)


def run_curriculum(model, loaders, lr, mixtures, samples_per_phase, device):
    for mix in mixtures:
        hp = {
            "training_samples": samples_per_phase,
            "learning_rate": lr,
            "mixture_ratio": mix,
            "phase_batch_size": loaders["easy_loader"].batch_size,
        }
        run_phase_training(
            model,
            loaders["easy_loader"],
            loaders["med_loader"],
            loaders["hard_loader"],
            hp,
            device,
        )
    ea = evaluate_accuracy(model, loaders["easy_loader"], device)
    ma = evaluate_accuracy(model, loaders["med_loader"], device)
    ha = evaluate_accuracy(model, loaders["hard_loader"], device)
    return (ea + ma + ha) / 3.0


def train_uniform(model, loaders, lr, total_samples, device):
    dataset = ConcatDataset([
        loaders["easy_ds"],
        loaders["med_ds"],
        loaders["hard_ds"],
    ])
    # Use uniform weights so sampling follows the dataset's natural
    # composition (e.g. ~90% easy, 7.5% medium, 2.5% hard).
    weights = [1.0] * len(dataset)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=total_samples,
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=loaders["easy_loader"].batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        seen += x.size(0)
        if seen >= total_samples:
            break
    ea = evaluate_accuracy(model, loaders["easy_loader"], device)
    ma = evaluate_accuracy(model, loaders["med_loader"], device)
    ha = evaluate_accuracy(model, loaders["hard_loader"], device)
    return (ea + ma + ha) / 3.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")

    data = prepare_dataset(cfg, easy_frac=0.9, med_frac=0.075, seed=0)
    device = data["device"]
    lr = sum(cfg["curriculum"]["learning_rate_range"]) / 2
    total = cfg["curriculum"]["train_samples_max"]

    # Determine observation dimension for agents
    tmp_env = CurriculumEnv(cfg)
    obs_dim = len(tmp_env.reset())
    action_dim = 5

    agent = DDPGAgent(obs_dim, action_dim, cfg)
    actor_pth = os.path.join("results", "behavior_cloning", "actor.pth")
    if os.path.exists(actor_pth):
        agent.actor.load_state_dict(torch.load(actor_pth, map_location=agent.device))
        print(f"Loaded actor checkpoint from {actor_pth}")
    agent.actor.eval()

    rand_agent = DDPGAgent(obs_dim, action_dim, cfg)
    rand_agent.actor.eval()

    results = {}

    # 1) Easy -> Medium -> Hard
    m1 = build_model(cfg, data["model_cfg"], device, seed=0)
    mixtures = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    acc = run_curriculum(m1, data, lr, mixtures, total // 3, device)
    results["easy_med_hard"] = acc

    # 2) 80E10M10H -> 10E80M10H -> 10E10M80H
    m2 = build_model(cfg, data["model_cfg"], device, seed=0)
    mixtures = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    acc = run_curriculum(m2, data, lr, mixtures, total // 3, device)
    results["progressive"] = acc

    # 3) Curriculum from pretrained agent
    m3 = build_model(cfg, data["model_cfg"], device, seed=0)
    env_a = CurriculumEnv(cfg)
    env_a.reset(0.9, 0.075)
    env_a.easy_subset.indices = list(data["easy_ds"].indices)
    env_a.medium_subset.indices = list(data["med_ds"].indices)
    env_a.hard_subset.indices = list(data["hard_ds"].indices)
    state = env_a.get_observation()
    remaining = total
    for _ in range(env_a.max_phases):
        act = agent.select_action(state, noise_enable=False)
        lr_a = float(act[0])
        mix = act[1:4].cpu().numpy()
        mix = np.clip(mix, 0.0, None)
        mix = mix / mix.sum() if mix.sum() > 0 else np.array([1/3,1/3,1/3])
        frac = float(act[4])
        num = int(frac * remaining)
        if num <= 0:
            num = remaining
        run_phase_training(m3, data["easy_loader"], data["med_loader"], data["hard_loader"],
                           {"training_samples": num, "learning_rate": lr_a, "mixture_ratio": mix.tolist(), "phase_batch_size": data["batch_size"]}, device)
        remaining -= num
        state, _, done = env_a.step(act)
        if done or remaining <= 0:
            break
    ea = evaluate_accuracy(m3, data["easy_loader"], device)
    ma = evaluate_accuracy(m3, data["med_loader"], device)
    ha = evaluate_accuracy(m3, data["hard_loader"], device)
    results["pretrained_agent"] = (ea + ma + ha) / 3.0

    # 4) Curriculum from random agent
    m4 = build_model(cfg, data["model_cfg"], device, seed=0)
    env_b = CurriculumEnv(cfg)
    env_b.reset(0.9, 0.075)
    env_b.easy_subset.indices = list(data["easy_ds"].indices)
    env_b.medium_subset.indices = list(data["med_ds"].indices)
    env_b.hard_subset.indices = list(data["hard_ds"].indices)
    state = env_b.get_observation()
    remaining = total
    for _ in range(env_b.max_phases):
        act = rand_agent.select_action(state, noise_enable=False)
        lr_a = float(act[0])
        mix = act[1:4].cpu().numpy()
        mix = np.clip(mix, 0.0, None)
        mix = mix / mix.sum() if mix.sum() > 0 else np.array([1/3,1/3,1/3])
        frac = float(act[4])
        num = int(frac * remaining)
        if num <= 0:
            num = remaining
        run_phase_training(m4, data["easy_loader"], data["med_loader"], data["hard_loader"],
                           {"training_samples": num, "learning_rate": lr_a, "mixture_ratio": mix.tolist(), "phase_batch_size": data["batch_size"]}, device)
        remaining -= num
        state, _, done = env_b.step(act)
        if done or remaining <= 0:
            break
    ea = evaluate_accuracy(m4, data["easy_loader"], device)
    ma = evaluate_accuracy(m4, data["med_loader"], device)
    ha = evaluate_accuracy(m4, data["hard_loader"], device)
    results["random_agent"] = (ea + ma + ha) / 3.0

    # 5) No curriculum
    m5 = build_model(cfg, data["model_cfg"], device, seed=0)
    acc = train_uniform(m5, data, lr, total, device)
    results["no_curriculum"] = acc

    for k, v in results.items():
        print(f"{k}: {v:.2f}%")


if __name__ == "__main__":
    main()
