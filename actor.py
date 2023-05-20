import torch
import torch.nn as nn
import torch.distributions
from common import MLP, ParallelizedEnsembleFlattenMLP

from torch.distributions import Distribution, Normal


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, n_layers, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        self.mlp = MLP(
            state_dim, 2 * action_dim, hidden_dim, n_layers, dropout_rate=dropout_rate
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self, states
    ):
        mu, log_std = self.mlp(states).chunk(2, dim=-1)
        mu = torch.tanh(mu)
        return mu

    def get_action(self, states):
        mu = self.forward(states)
        return mu

class EnsembleActor(nn.Module):
    """Ensemble MLP actor networks."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, num_as, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        self.mlp = ParallelizedEnsembleFlattenMLP(
            num_as, hidden_dim, state_dim, 2 * action_dim, dropout_rate=dropout_rate
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_as = num_as

    def forward(
        self, states
    ):
        # (ensemble_size, batch_size, output_size)
        mu, log_std = self.mlp(states).chunk(2, dim=-1)
        mu = torch.tanh(mu)
        return mu

    def get_action(self, states):
        mu = self.forward(states)
        return mu
    
    def sample(self, states):
        mu = self.forward(states)
        elite = torch.randint(0, self.num_as, [1])[0]

        return mu[elite,:,:]
