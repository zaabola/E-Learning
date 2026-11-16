"""
Reinforcement Learning Architectures for the Adaptive Game Master
-----------------------------------------------------------------
1. `DuelingDoubleDQN` (Value-Based Deep RL):
   Separates state value V(s) and action advantage A(s, a) with LayerNorm
   stabilization. Trained via Double Q-Learning + Experience Replay.

2. `PPOActorCritic` (Policy-Gradient Deep RL):
   Decoupled Policy Actor pi(a|s) and State-Value Critic V(s) with orthogonal
   initialization. Trained via Clipped Surrogate Objective + GAE.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class DuelingDoubleDQN(nn.Module):
    """
    Dueling Network Architecture (Wang et al., ICML 2016) trained with
    Double Q-Learning targets (van Hasselt et al., AAAI 2016).
    """
    def __init__(self, state_dim=12, num_actions=5, hidden_dim=128):
        super(DuelingDoubleDQN, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.feature_trunk = nn.Sequential(
            _ortho_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # State-Value stream V(s)
        self.value_stream = nn.Sequential(
            _ortho_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim // 2, 1), gain=1.0),
        )

        # Action-Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            _ortho_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim // 2, num_actions), gain=0.1),
        )

    def forward(self, x):
        feats = self.feature_trunk(x)
        val = self.value_stream(feats)
        adv = self.advantage_stream(feats)
        # Identifiable dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q_values = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_values

    @torch.no_grad()
    def select_action(self, state_vec, device='cpu'):
        was_training = self.training
        self.eval()
        s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        q_vals = self.forward(s_t)
        act = int(torch.argmax(q_vals, dim=-1).item())
        if was_training:
            self.train()
        return act


class PPOActorCritic(nn.Module):
    """
    Actor-Critic Network for Proximal Policy Optimization (Schulman et al., 2017).
    Uses separate Actor and Critic MLPs to prevent policy-value gradient interference.
    """
    def __init__(self, state_dim=12, num_actions=5, hidden_dim=128):
        super(PPOActorCritic, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.actor = nn.Sequential(
            _ortho_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, num_actions), gain=0.01),
        )

        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def select_action(self, state_vec, device='cpu', deterministic=True):
        was_training = self.training
        self.eval()
        s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = self.forward(s_t)
        if deterministic:
            act = int(torch.argmax(logits, dim=-1).item())
        else:
            act = int(Categorical(logits=logits).sample().item())
        if was_training:
            self.train()
        return act
