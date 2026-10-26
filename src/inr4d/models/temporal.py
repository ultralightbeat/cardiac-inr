"""How the time coordinate enters the network.

Input coordinates are (x, y, z, t), all in [-1, 1]; frame k of T frames has
t = 2k/(T-1) - 1. Encodings:

  'none'     : drop t, use (x, y, z)            -> static 3D fit, 3 inputs
  'raw'      : (x, y, z, t)                      -> 4 inputs
  'scaled'   : (x, y, z, s*t)                    -> 4 inputs; s < 1 makes the field
               change more slowly in time (prototype: s in 0.05-0.25 worked best)
  'periodic' : (x, y, z, s*sin(phi), s*cos(phi)), phi = 2*pi*k/T
               -> 5 inputs; the last frame connects smoothly to the first

The prototype study that motivated these options is logged in docs/DECISIONS.md.
"""

import math

import torch
from torch import nn

MODES = ("none", "raw", "scaled", "periodic")


class TimeEncoding(nn.Module):
    def __init__(self, mode="scaled", scale=0.25, num_frames=30, z_scale=1.0):
        super().__init__()
        # z_scale < 1 makes the field change more slowly ACROSS slices (same idea as the time
        # scale): slices are 10 mm apart, so a sine network with omega_0 = 30 otherwise
        # oscillates between them and fills unseen positions with high-frequency noise.
        self.z_scale = float(z_scale)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.scale = float(scale)
        self.num_frames = int(num_frames)

    @property
    def out_features(self) -> int:
        return {"none": 3, "raw": 4, "scaled": 4, "periodic": 5}[self.mode]

    def forward(self, coords):
        xyz, t = coords[:, :3], coords[:, 3:4]
        if self.z_scale != 1.0:
            xyz = xyz * xyz.new_tensor([1.0, 1.0, self.z_scale])
        if self.mode == "none":
            return xyz
        if self.mode == "raw":
            return torch.cat([xyz, t], dim=-1)
        if self.mode == "scaled":
            return torch.cat([xyz, self.scale * t], dim=-1)
        # periodic: recover the frame index k from t, then the phase on the circle
        T = self.num_frames
        k = (t + 1.0) * (T - 1) / 2.0
        phi = 2.0 * math.pi * k / T
        return torch.cat([xyz, self.scale * torch.sin(phi), self.scale * torch.cos(phi)], dim=-1)

    def extra_repr(self):
        return f"mode={self.mode}, scale={self.scale}, num_frames={self.num_frames}, z_scale={self.z_scale}"


class NeuralField(nn.Module):
    """Time encoding followed by a coordinate network: (x, y, z, t) -> intensity."""

    def __init__(self, encoding: TimeEncoding, network: nn.Module):
        super().__init__()
        self.encoding = encoding
        self.network = network

    def forward(self, coords):
        return self.network(self.encoding(coords))
