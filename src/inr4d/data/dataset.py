"""Coordinate/intensity samples for a cine-MRI volume."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class CoordinateDataset:
    """A tensor dataset containing every voxel from the selected frames.

    Spatial coordinates are normalised from voxel centres expressed in
    millimetres. Intensities use min-max scaling over the complete volume.
    """

    def __init__(self, volume, frames=None):
        # 先验证体数据和体素间距，避免归一化产生无效值。
        data = np.asarray(volume.data, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"volume.data must have shape [H, W, D, T], got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("volume.data contains NaN or infinite values")

        spacing = np.asarray(volume.spacing, dtype=np.float32)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("volume.spacing must contain three positive finite values")

        total_frames = data.shape[3]
        if frames is None:
            selected_frames = np.arange(total_frames, dtype=np.int64)
        else:
            selected_frames = np.asarray(list(frames), dtype=np.int64)
            if selected_frames.ndim != 1 or selected_frames.size == 0:
                raise ValueError("frames must be a non-empty iterable of frame indices")
            if np.any(selected_frames < 0) or np.any(selected_frames >= total_frames):
                raise IndexError("frames contains an index outside the volume")
            if np.unique(selected_frames).size != selected_frames.size:
                raise ValueError("frames must not contain duplicate indices")

        h, w, d = data.shape[:3]
        # 坐标由体素中心的物理位置生成，而不是由索引直接生成。
        physical_extents = (np.asarray((h, w, d), dtype=np.float32) - 1.0) * spacing
        coordinate_scale = max(float(physical_extents.max()) / 2.0, 1.0)
        axes = [
            (np.arange(size, dtype=np.float32) * step - extent / 2.0) / coordinate_scale
            if size > 1 else np.zeros(1, dtype=np.float32)
            for size, step, extent in zip((h, w, d), spacing, physical_extents)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        spatial = np.broadcast_to(grid[..., None, :], (h, w, d, selected_frames.size, 3))
        time = (2.0 * selected_frames.astype(np.float32) / (total_frames - 1) - 1.0
                if total_frames > 1 else np.zeros(1, dtype=np.float32))
        time_grid = np.broadcast_to(time.reshape(1, 1, 1, -1, 1),
                                    (h, w, d, selected_frames.size, 1))
        coords = np.concatenate((spatial, time_grid), axis=-1).reshape(-1, 4)

        selected_data = data[..., selected_frames]
        minimum = float(data.min())
        maximum = float(data.max())
        # 使用全体积的最小值和最大值把强度映射到 [-1, 1]。
        if maximum == minimum:
            values = np.zeros_like(selected_data, dtype=np.float32)
        else:
            values = (2.0 * (selected_data - minimum) / (maximum - minimum) - 1.0)

        self.coords: Tensor = torch.from_numpy(coords)
        self.values: Tensor = torch.from_numpy(values.reshape(-1, 1).astype(np.float32))
        self.frame_ids: Tensor = torch.from_numpy(
            np.broadcast_to(selected_frames.reshape(1, 1, 1, -1), (h, w, d, selected_frames.size))
            .reshape(-1).copy()
        ).long()
        if not torch.isfinite(self.coords).all() or not torch.isfinite(self.values).all():
            raise ValueError("normalisation produced NaN or infinite values")

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Return a random mini-batch of ``n`` ``(coords, values)`` pairs."""
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        indices = torch.randint(self.coords.shape[0], (n,))
        return self.coords[indices], self.values[indices]
