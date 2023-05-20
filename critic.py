import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLP, ParallelizedEnsembleFlattenMLP

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, n_layers, dropout_rate=None):
        super(Critic, self).__init__()

        self.critic1 = MLP(state_dim + action_dim, 1, hidden_dim, n_layers, dropout_rate = dropout_rate)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_dim, n_layers, dropout_rate = dropout_rate)
    
    def forward(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2
    
    def Q1(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q1 = self.critic1(sa)
        
        return q1

class TD3Critic(nn.Module):
    """
    From TD3+BC
    """

    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class EnsembleCritic(nn.Module):
    """Ensemble MLP critic networks."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, num_qs, dropout_rate=None,
    ):
        super().__init__()

        self.mlp = ParallelizedEnsembleFlattenMLP(
            num_qs, hidden_dim, state_dim+action_dim, 1, dropout_rate=dropout_rate
        )

    def forward(
        self, states, actions
    ):
        if len(states.shape) == 3:
            sa = torch.cat([states, actions], 2)
        else:
            sa = torch.cat([states, actions], 1)
        # (ensemble_size, batch_size, output_size)
        mu = self.mlp(sa)
        return mu

    def sample(self, states, actions):
        if len(states.shape) == 3:
            sa = torch.cat([states, actions], 2)
        else:
            sa = torch.cat([states, actions], 1)
        mu = self.mlp.sample(sa)
        return mu