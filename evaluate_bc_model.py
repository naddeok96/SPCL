import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import trange

from curriculum_env import CurriculumEnv
from curriculum_env import build_cnn_model, build_mlp_model
from curriculum import evaluate_accuracy
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
        rewards.append(reward)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--episodes", "-n", type=int, default=100, help="Number of evaluation episodes"
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Temporary env to determine observation dimension
    tmp_env = CurriculumEnv(cfg)
    obs_dim = len(tmp_env.reset())
    action_dim = 5

    # Initialize agents
    agent = DDPGAgent(obs_dim, action_dim, cfg)
    bc_dir = os.path.join("results", "behavior_cloning")
    actor_pth = os.path.join(bc_dir, "actor.pth")
    critic_pth = os.path.join(bc_dir, "critic1.pth")
    if os.path.exists(actor_pth):
        agent.actor.load_state_dict(torch.load(actor_pth, map_location=agent.device))
        print(f"Loaded actor checkpoint from {actor_pth}")
    if os.path.exists(critic_pth):
        agent.critic1.load_state_dict(torch.load(critic_pth, map_location=agent.device))
        print(f"Loaded critic checkpoint from {critic_pth}")
    agent.actor.eval()
    agent.critic1.eval()

    base_agent = DDPGAgent(obs_dim, action_dim, cfg)
    base_agent.actor.eval()
    base_agent.critic1.eval()

    gamma = cfg["rl"].get("gamma", 0.99)
    lr_default = sum(cfg["curriculum"]["learning_rate_range"]) / 2

    bc_rewards, bc_mses = [], []
    base_rewards, base_mses = [], []
    std_accs = []

    for _ in trange(args.episodes, desc="Evaluation Episodes", leave=False):
        proto_env = CurriculumEnv(cfg)
        proto_env.reset()
        easy, med = proto_env.easy_frac, proto_env.medium_frac

        env_bc = CurriculumEnv(cfg)
        env_bc.reset(easy, med)
        r_bc, mse_bc = evaluate_single_episode(agent, env_bc, gamma)
        bc_rewards.append(r_bc)
        bc_mses.append(mse_bc)

        env_base = CurriculumEnv(cfg)
        env_base.reset(easy, med)
        r_base, mse_base = evaluate_single_episode(base_agent, env_base, gamma)
        base_rewards.append(r_base)
        base_mses.append(mse_base)

        env_std = CurriculumEnv(cfg)
        env_std.reset(easy, med)
        acc = train_standard_model(env_std, cfg["curriculum"]["train_samples_max"], lr_default)
        std_accs.append(acc)

    print(
        f"Behavior Cloned Model - Avg Reward: {np.mean(bc_rewards):.3f}, Critic MSE: {np.mean(bc_mses):.3f}"
    )
    print(
        f"Random Baseline - Avg Reward: {np.mean(base_rewards):.3f}, Critic MSE: {np.mean(base_mses):.3f}"
    )
    print(f"Standard Training Macro Accuracy: {np.mean(std_accs):.2f}%")


if __name__ == "__main__":
    main()
