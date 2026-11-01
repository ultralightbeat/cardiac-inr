"""SIREN: a multilayer perceptron with sine activations (Sitzmann et al., arXiv:2006.09661).

Each hidden unit computes sin(omega_0 * (W x + b)). The last layer is linear.

Initialisation (Sec. 3.2 of the paper):
  * first layer:  W ~ U(-1/n, 1/n), so that sin(omega_0 * W x) spans several periods
    over inputs in [-1, 1];
  * other layers: W ~ U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0), which keeps the
    pre-activation distribution approximately standard normal from layer to layer,
    so a deep network trains stably.
n is the number of inputs of the layer.
"""

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """Linear layer followed by sin(omega_0 * x), with the SIREN initialisation."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.linear.in_features
        bound = 1.0 / n if self.is_first else math.sqrt(6.0 / n) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-1.0 / math.sqrt(n), 1.0 / math.sqrt(n))

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """coords [B, in_features] -> intensity [B, out_features].

    hidden_layers counts the sine layers after the first one, so the default
    network (4 -> 256, three 256 -> 256 layers, 256 -> 1) has 198,913 parameters.
    """

    def __init__(self, in_features=4, hidden_features=256, hidden_layers=3,
                 out_features=1, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, first_omega_0, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, hidden_omega_0))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_features, out_features)
        bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
        with torch.no_grad():
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, coords):
        return self.head(self.body(coords))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
