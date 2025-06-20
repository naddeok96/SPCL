import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import select_top_percent, behavior_clone, tune_value_function
from replay_buffer import build_augmented_replay_buffer

class SimplePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x)

class SimpleValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.lin(x)


def make_dataset():
    states = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    actions = states.clone()
    rewards = torch.tensor([1.,1.,1.,0.,0.,0.])
    next_states = states + 1
    dones = torch.tensor([0,0,1,0,0,1], dtype=torch.float32)
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
    }


def test_select_top_percent():
    data = make_dataset()
    top = select_top_percent(data, 50)
    assert len(top['states']) == 3
    assert torch.all(top['rewards'] == 1)


def test_behavior_clone():
    torch.manual_seed(0)
    data = make_dataset()
    policy = SimplePolicy()
    # held-out last sample
    expert = {
        'states': data['states'][:5],
        'actions': data['actions'][:5]
    }
    hold = data['states'][5:]
    target = data['actions'][5:]
    before = torch.nn.functional.mse_loss(policy(hold), target).item()
    behavior_clone(policy, expert, epochs=200, batch_size=2)
    after = torch.nn.functional.mse_loss(policy(hold), target).item()
    assert after < before


def test_tune_value_function():
    torch.manual_seed(0)
    data = make_dataset()
    policy = SimplePolicy()
    value = SimpleValue()
    ds = TensorDataset(data['states'], data['actions'], data['rewards'], data['next_states'], data['dones'])
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    # random action q before
    rand_a = torch.zeros_like(data['actions'])
    before = value(data['states'], rand_a).mean().item()
    tune_value_function(policy, value, loader, {'gamma':0.5,'critic_lr':1e-2})
    after = value(data['states'], rand_a).mean().item()
    assert after < before


def test_augmented_replay_buffer():
    data = make_dataset()
    elite = select_top_percent(data, 50)
    buf = build_augmented_replay_buffer(elite, data, capacity=5, elite_fraction=0.5, device='cpu')
    assert len(buf) == 5
    # elite rewards should remain after refresh
    rewards_before = torch.cat([t[2] for t in buf.elite])
    buf.refresh_random([(torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([1.]))])
    rewards_after = torch.cat([t[2] for t in buf.elite])
    assert torch.allclose(rewards_before, rewards_after)

