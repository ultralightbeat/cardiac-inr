"""Role 1: loading ACDC 4D NIfTI series."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Volume4D:
    data: np.ndarray            # float32 [H, W, D, T]
    spacing: tuple              # (sx, sy, sz) in mm
    affine: np.ndarray          # 4x4 voxel-index -> mm


def load_acdc_patient(path: str) -> Volume4D:
    """Load one patient's 4D series. TODO(Role 1)."""
    raise NotImplementedError
