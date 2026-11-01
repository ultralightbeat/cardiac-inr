"""Loading a 4D cine-MRI series (ACDC layout, or any 4D NIfTI file)."""

import glob
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np


@dataclass
class Volume4D:
    data: np.ndarray            # float32 [H, W, D, T]
    spacing: tuple              # (sx, sy, sz) in mm
    affine: np.ndarray          # 4x4 voxel-index -> mm


def find_4d_image(path: str) -> str:
    """Return the 4D NIfTI file for a patient.

    path can be the file itself, or a folder. In a folder the ACDC name
    <folder>_4d.nii(.gz) is preferred, then any *_4d.nii(.gz), then any 4D NIfTI."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} does not exist")
    patient_id = os.path.basename(os.path.normpath(path))
    # 未压缩和 gzip 压缩的 NIfTI 文件都支持。
    for name in (f"{patient_id}_4d.nii", f"{patient_id}_4d.nii.gz"):
        if os.path.exists(os.path.join(path, name)):
            return os.path.join(path, name)
    found = sorted(glob.glob(os.path.join(path, "*_4d.nii"))) + sorted(glob.glob(os.path.join(path, "*_4d.nii.gz")))
    if found:
        return found[0]
    for f in sorted(glob.glob(os.path.join(path, "*.nii")) + glob.glob(os.path.join(path, "*.nii.gz"))):
        try:
            if len(nib.load(f).shape) == 4:
                return f
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find a 4D NIfTI image for {patient_id} in {path}")


def load_acdc_patient(path: str) -> Volume4D:
    """Load one patient's 4D series from a patient folder or a 4D NIfTI file."""
    img = nib.load(find_4d_image(path))
    data = img.get_fdata(dtype=np.float32)

    affine = img.affine
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    if data.ndim != 4:
        raise ValueError(f"image must be 4D [H, W, D, T], got shape {data.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"NIfTI affine must have shape (4, 4), got {affine.shape}")
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError(f"NIfTI spacing must contain three positive finite values, got {spacing}")

    return Volume4D(data=data, spacing=spacing, affine=affine)
