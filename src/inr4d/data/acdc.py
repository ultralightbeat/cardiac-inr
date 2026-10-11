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

# AI-assisted: Robust NIfTI loader that handles both .nii and .nii.gz
def load_acdc_patient(path: str) -> Volume4D:
    """Load one patient's 4D series from a patient directory path."""
    patient_id = os.path.basename(os.path.normpath(path))
    
    # Try finding either .nii or .nii.gz
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
    
    return Volume4D(data=data, spacing=spacing, affine=affine)