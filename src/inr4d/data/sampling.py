"""Coordinate grids for training subsets and for reconstruction.

Uses exactly the same normalisation as CoordinateDataset (dataset.py):
  * spatial coordinates from voxel centres in millimetres, centred, divided by half
    of the largest physical extent of the WHOLE volume (so they stay comparable
    across crops and subsets);
  * t = 2k/(T-1) - 1 for frame k of T;
  * intensity min-max scaled to [-1, 1] over the whole 4D volume.

On top of CoordinateDataset this module can select a subset of slices (for the
held-out-slice test), a subset of frames, and an in-plane crop around the heart
(to keep CPU training fast). It also builds coordinates at positions that are NOT
on the voxel grid, e.g. 40 slices instead of 10, or frame 6.5.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Normaliser:
    shape_hwd: tuple          # (H, W, D) of the full volume
    spacing: tuple            # (sx, sy, sz) in mm
    num_frames: int
    vmin: float
    vmax: float

    @classmethod
    def from_volume(cls, volume):
        data = np.asarray(volume.data)
        return cls(tuple(data.shape[:3]), tuple(float(s) for s in volume.spacing),
                   int(data.shape[3]), float(data.min()), float(data.max()))

    @property
    def scale(self) -> float:
        extents = (np.asarray(self.shape_hwd, dtype=np.float64) - 1.0) * np.asarray(self.spacing)
        return max(float(extents.max()) / 2.0, 1.0)

    def axis(self, dim: int, index) -> np.ndarray:
        """Normalised coordinate of (possibly fractional) voxel index along x/y/z."""
        n = self.shape_hwd[dim]
        index = np.asarray(index, dtype=np.float64)
        if n == 1:
            return np.zeros_like(index, dtype=np.float32)
        extent = (n - 1) * self.spacing[dim]
        return ((index * self.spacing[dim] - extent / 2.0) / self.scale).astype(np.float32)

    def time(self, frame) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float64)
        if self.num_frames == 1:
            return np.zeros_like(frame, dtype=np.float32)
        return (2.0 * frame / (self.num_frames - 1) - 1.0).astype(np.float32)

    def intensity(self, x) -> np.ndarray:
        if self.vmax == self.vmin:
            return np.zeros_like(x, dtype=np.float32)
        return (2.0 * (np.asarray(x, dtype=np.float32) - self.vmin) / (self.vmax - self.vmin) - 1.0)

    def grid(self, rows, cols, slices, frames) -> np.ndarray:
        """Coordinates [len(rows), len(cols), len(slices), len(frames), 4] (float32).
        Indices may be fractional (e.g. slices=np.linspace(0, 9, 40))."""
        ax = [self.axis(0, rows), self.axis(1, cols), self.axis(2, slices)]
        tt = self.time(frames)
        g = np.meshgrid(ax[0], ax[1], ax[2], tt, indexing="ij")
        return np.stack(g, axis=-1).astype(np.float32)


@dataclass
class Subset:
    coords: np.ndarray        # [N, 4]
    values: np.ndarray        # [N, 1]
    shape: tuple              # (h, w, n_slices, n_frames) of the selected block


def full_range(n):
    return np.arange(n)


def select(volume, norm: Normaliser, crop=None, slices=None, frames=None) -> Subset:
    """Training/test samples for a crop x slice-subset x frame-subset block."""
    data = np.asarray(volume.data, dtype=np.float32)
    H, W, D, T = data.shape
    r0, r1, c0, c1 = crop if crop is not None else (0, H, 0, W)
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    slices = np.arange(D) if slices is None else np.asarray(slices, dtype=int)
    frames = np.arange(T) if frames is None else np.asarray(frames, dtype=int)
    block = data[r0:r1, c0:c1][:, :, slices][:, :, :, frames]
    coords = norm.grid(rows, cols, slices, frames).reshape(-1, 4)
    values = norm.intensity(block).reshape(-1, 1)
    return Subset(coords, values.astype(np.float32), block.shape)


def block(volume, norm: Normaliser, crop=None, slices=None, frames=None) -> np.ndarray:
    """Ground-truth intensities of a block, normalised to [-1, 1]."""
    s = select(volume, norm, crop, slices, frames)
    return s.values.reshape(s.shape)


def select_shape(volume, crop=None, slices=None, frames=None):
    H, W, D, T = np.asarray(volume.data).shape
    r0, r1, c0, c1 = crop if crop is not None else (0, H, 0, W)
    nz = D if slices is None else len(slices)
    nt = T if frames is None else len(frames)
    return (r1 - r0, c1 - c0, nz, nt)


# --------------------------------------------------------------------------- ACDC helpers

def read_info(patient_dir: str) -> dict:
    """Parse Info.cfg (ED, ES, Group, Height, Weight, NbFrame). Missing file -> {}."""
    if os.path.isfile(patient_dir):
        patient_dir = os.path.dirname(patient_dir)
    info = {}
    path = os.path.join(patient_dir, "Info.cfg")
    if not os.path.exists(path):
        return info
    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass
                info[key.strip()] = value
    return info


def load_labels(patient_dir: str) -> dict:
    """Expert labels {frame_index (0-based): int array [H, W, D]} from *_frameXX_gt.nii(.gz).
    Labels: 0 background, 1 right ventricle, 2 myocardium, 3 left ventricle."""
    import nibabel as nib

    if os.path.isfile(patient_dir):
        patient_dir = os.path.dirname(patient_dir)
    out = {}
    for name in sorted(os.listdir(patient_dir)):
        m = re.search(r"_frame(\d+)_gt\.nii(\.gz)?$", name)
        if m:
            img = nib.load(os.path.join(patient_dir, name))
            out[int(m.group(1)) - 1] = np.asarray(img.dataobj).astype(np.int16)
    return out


def heart_mask(labels: dict, shape_hwd, dilate: int = 5) -> np.ndarray:
    """Union of all labelled heart structures over the labelled frames (ED and ES),
    dilated in-plane by `dilate` pixels so it also covers the wall in between.
    Returns bool [H, W, D]. Falls back to all-True if no labels exist."""
    if not labels:
        return np.ones(shape_hwd, dtype=bool)
    mask = np.zeros(shape_hwd, dtype=bool)
    for lab in labels.values():
        mask |= lab > 0
    if dilate > 0:
        from skimage.morphology import dilation, disk

        fp = disk(dilate)
        for z in range(mask.shape[2]):
            mask[:, :, z] = dilation(mask[:, :, z].astype(np.uint8), fp) > 0
    return mask


def heart_crop(mask: np.ndarray, pad: int = 12, square: bool = True):
    """In-plane bounding box (r0, r1, c0, c1) of the heart mask, padded."""
    H, W = mask.shape[:2]
    rows = np.where(mask.any(axis=(1, 2)))[0]
    cols = np.where(mask.any(axis=(0, 2)))[0]
    if rows.size == 0:
        return (0, H, 0, W)
    r0, r1 = rows.min() - pad, rows.max() + 1 + pad
    c0, c1 = cols.min() - pad, cols.max() + 1 + pad
    if square:
        side = max(r1 - r0, c1 - c0)
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        r0, r1 = rc - side // 2, rc - side // 2 + side
        c0, c1 = cc - side // 2, cc - side // 2 + side
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, H), min(c1, W)
    return (int(r0), int(r1), int(c0), int(c1))


def split_alternate(n: int):
    """Train on even indices (plus the last one), test on odd indices in between.
    Test items always have a known neighbour on both sides, so interpolation is fair."""
    test = np.arange(1, n - 1, 2)
    train = np.setdiff1d(np.arange(n), test)
    return train, test


def split_stride(n: int, k: int = 2):
    """Keep indices 0, k, 2k, ... and the last one; hide everything in between.
    k = 2 is split_alternate. Larger k means larger gaps (harder for interpolation)."""
    if k < 2:
        raise ValueError("stride must be >= 2")
    train = np.union1d(np.arange(0, n, k), [n - 1])
    test = np.setdiff1d(np.arange(n), train)
    return train, test
