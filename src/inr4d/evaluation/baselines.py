"""Baselines the network has to beat.

linear_along_axis fills missing positions along one axis by linear interpolation
between the nearest known positions on both sides (outside the known range it
copies the nearest known one):
  * along z (slices): with all in-plane pixels known this is exactly trilinear
    interpolation of the slice stack;
  * along t (frames): frame k = weighted average of the two neighbouring frames,
    e.g. (frame 5 + frame 7) / 2 for frame 6.
"""

from __future__ import annotations

import numpy as np


def linear_along_axis(known: np.ndarray, known_idx, query_idx, axis: int) -> np.ndarray:
    """known: array whose `axis` has len(known_idx) entries at positions known_idx.
    Returns the array with that axis replaced by values at query_idx (can be fractional)."""
    known = np.asarray(known, np.float32)
    known_idx = np.asarray(known_idx, np.float64)
    query_idx = np.asarray(query_idx, np.float64)
    order = np.argsort(known_idx)
    known_idx = known_idx[order]
    known = np.take(known, order, axis=axis)
    hi = np.clip(np.searchsorted(known_idx, query_idx, side="left"), 0, len(known_idx) - 1)
    lo = np.clip(hi - 1, 0, len(known_idx) - 1)
    exact = known_idx[hi] == query_idx
    lo = np.where(exact, hi, lo)
    span = known_idx[hi] - known_idx[lo]
    w = np.where(span > 0, (query_idx - known_idx[lo]) / np.where(span > 0, span, 1), 0.0)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    a = np.take(known, lo, axis=axis)
    b = np.take(known, hi, axis=axis)
    shape = [1] * known.ndim
    shape[axis] = len(query_idx)
    w = w.reshape(shape)
    return (1.0 - w) * a + w * b


def interpolate_slices(block, known_slices, query_slices):
    """block [h, w, len(known_slices), t] -> [h, w, len(query_slices), t]."""
    return linear_along_axis(block, known_slices, query_slices, axis=2)


def interpolate_frames(block, known_frames, query_frames):
    """block [h, w, d, len(known_frames)] -> [h, w, d, len(query_frames)]."""
    return linear_along_axis(block, known_frames, query_frames, axis=3)
