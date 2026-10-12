"""Role 1: loading ACDC 4D NIfTI series."""
import os
from dataclasses import dataclass
import numpy as np
import nibabel as nib

@dataclass
class Volume4D:
    data: np.ndarray            # float32 [H, W, D, T]
    spacing: tuple              # (sx, sy, sz) in mm
    affine: np.ndarray          # 4x4 voxel-index -> mm

# 加载器同时支持未压缩和 gzip 压缩的 NIfTI 文件。
def load_acdc_patient(path: str) -> Volume4D:
    """Load one patient's 4D series from a patient directory path."""
    patient_id = os.path.basename(os.path.normpath(path))
    
    # 先查找未压缩文件，再查找 gzip 压缩文件。
    file_nii = os.path.join(path, f"{patient_id}_4d.nii")
    file_gz = os.path.join(path, f"{patient_id}_4d.nii.gz")
    
    if os.path.exists(file_nii):
        img_path = file_nii
    elif os.path.exists(file_gz):
        img_path = file_gz
    else:
        raise FileNotFoundError(f"Could not find 4D image for {patient_id} in {path}")
        
    img = nib.load(img_path)
    data = img.get_fdata(dtype=np.float32)

    affine = img.affine
    spacing = tuple(img.header.get_zooms()[:3])
    if data.ndim != 4:
        raise ValueError(f"ACDC image must be 4D, got shape {data.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"NIfTI affine must have shape (4, 4), got {affine.shape}")
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError(f"NIfTI spacing must contain three positive finite values, got {spacing}")

    return Volume4D(data=data, spacing=spacing, affine=affine)