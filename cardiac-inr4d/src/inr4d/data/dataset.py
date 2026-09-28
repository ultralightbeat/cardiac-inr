"""Role 1: (x, y, z, t) -> intensity samples.

Contract (docs/PROJECT_PLAN.md, section 3):
  coords [N, 4] in [-1, 1], normalised by PHYSICAL extent (mm), not voxel index
  values [N, 1] in [-1, 1]
"""


class CoordinateDataset:
    """TODO(Role 1): implement coords, values, frame_ids, sample(n)."""

    def __init__(self, volume, frames=None):
        raise NotImplementedError

    def sample(self, n: int):
        """Random mini-batch of n points -> (coords, values)."""
        raise NotImplementedError
