"""Overlap and surface-distance metrics for binary masks (segmentation quality).

dice     2|A∩B| / (|A| + |B|)                           1 = identical
jaccard  |A∩B| / |A∪B|
Surface distances use the voxel spacing in mm (anisotropic grids are handled):
hd       Hausdorff distance: the largest distance from a boundary point of one mask to the
         boundary of the other (worst local miss), mm
hd95     95th percentile of those distances (robust to a few outlier voxels), mm
assd     average symmetric surface distance, mm
Empty masks give NaN for the distances (and dice 1 if both are empty).
"""

from __future__ import annotations

import numpy as np


def dice(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def jaccard(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def _boundary(m):
    from scipy import ndimage

    return m & ~ndimage.binary_erosion(m, border_value=0)


def surface_distances(a, b, spacing):
    """Distances (mm) from the boundary of a to the boundary of b and back."""
    from scipy import ndimage

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return None
    ba, bb = _boundary(a), _boundary(b)
    da = ndimage.distance_transform_edt(~bb, sampling=spacing)[ba]
    db = ndimage.distance_transform_edt(~ba, sampling=spacing)[bb]
    return da, db


def compare(pred, ref, spacing) -> dict:
    """All metrics for one structure; spacing = voxel size in mm per axis."""
    out = {"dice": dice(pred, ref), "jaccard": jaccard(pred, ref),
           "voxels_pred": int(np.sum(pred)), "voxels_ref": int(np.sum(ref))}
    sd = surface_distances(pred, ref, spacing)
    if sd is None:
        out.update(hd=float("nan"), hd95=float("nan"), assd=float("nan"))
    else:
        da, db = sd
        alld = np.concatenate([da, db])
        out.update(hd=float(max(da.max(), db.max())), hd95=float(np.percentile(alld, 95)),
                   assd=float(alld.mean()))
    return out


def components(mask) -> int:
    from scipy import ndimage

    return int(ndimage.label(np.asarray(mask, bool))[1])
