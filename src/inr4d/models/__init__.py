"""Coordinate networks and time encodings. See docs/PROJECT_PLAN.md, section 3."""

from inr4d.models.fourier_mlp import FourierMLP, ReLUMLP
from inr4d.models.siren import SIREN, count_parameters
from inr4d.models.temporal import NeuralField, TimeEncoding


def build_field(model="siren", time_mode="scaled", time_scale=0.25, num_frames=30,
                hidden_features=256, hidden_layers=3, omega_0=30.0, seed=0, z_scale=1.0,
                fourier_freqs=64, fourier_sigma=10.0):
    """Create a NeuralField: time encoding + SIREN, plain ReLU, or ReLU with Fourier features
    (fourier_freqs random Gaussian frequencies with std fourier_sigma, Tancik et al. 2020)."""
    import torch

    torch.manual_seed(seed)
    enc = TimeEncoding(time_mode, time_scale, num_frames, z_scale)
    if model == "siren":
        net = SIREN(enc.out_features, hidden_features, hidden_layers, 1, omega_0, omega_0)
    elif model == "relu":
        net = ReLUMLP(enc.out_features, hidden_features, hidden_layers, 1)
    elif model == "fourier":
        net = FourierMLP(enc.out_features, hidden_features, hidden_layers, 1, num_frequencies=fourier_freqs,
                         sigma=fourier_sigma, seed=seed)
    else:
        raise ValueError(f"unknown model {model!r}")
    return NeuralField(enc, net)


__all__ = ["SIREN", "FourierMLP", "ReLUMLP", "TimeEncoding", "NeuralField", "build_field",
           "count_parameters"]
