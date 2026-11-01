"""Comparison model: a ReLU MLP, optionally with Fourier-feature input encoding
(Tancik et al., arXiv:2006.10739).

num_frequencies = 0 gives the plain ReLU control network (same width and depth as
SIREN). With num_frequencies > 0 the input x is mapped to
[sin(2*pi*B x), cos(2*pi*B x)] with a fixed random Gaussian matrix B (std = sigma).
"""

import math

import torch
from torch import nn


class FourierMLP(nn.Module):
    def __init__(self, in_features=4, hidden_features=256, hidden_layers=3, out_features=1,
                 num_frequencies=0, sigma=10.0, seed=0):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies > 0:
            g = torch.Generator().manual_seed(seed)
            B = torch.randn(in_features, self.num_frequencies, generator=g) * sigma
            self.register_buffer("B", B)
            first_in = 2 * self.num_frequencies
        else:
            self.register_buffer("B", torch.empty(0))
            first_in = in_features
        layers = [nn.Linear(first_in, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def encode(self, x):
        if self.num_frequencies == 0:
            return x
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, coords):
        return self.net(self.encode(coords))


def ReLUMLP(in_features=4, hidden_features=256, hidden_layers=3, out_features=1):
    """Plain ReLU network of the same size as SIREN (the control model)."""
    return FourierMLP(in_features, hidden_features, hidden_layers, out_features, num_frequencies=0)
