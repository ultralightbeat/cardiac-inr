"""Checks of the evaluation protocol (metrics, interpolation baseline, geometry metrics)."""

import math

import numpy as np
import pytest

from inr4d.evaluation.baselines import interpolate_frames
from inr4d.evaluation.geometry import compare, dice, jaccard
from inr4d.evaluation.metrics import psnr, temporal_error
from inr4d.viz import heart3d as H3


def test_psnr_identical_and_known_mse():
    x = np.random.default_rng(0).uniform(-1, 1, (8, 8, 3))
    assert psnr(x, x) == math.inf
    assert psnr(x + 0.1, x) == pytest.approx(10 * math.log10(4 / 0.01), abs=1e-6)


def test_psnr_rejects_nan_and_empty_mask():
    x = np.zeros((4, 4))
    with pytest.raises(ValueError):
        psnr(x * np.nan, x)
    with pytest.raises(ValueError):
        psnr(x, x + 1, np.zeros((4, 4), bool))


def test_psnr_mask_broadcasting():
    x, y = np.zeros((4, 4, 2, 3)), np.zeros((4, 4, 2, 3))
    y[0, 0] = 1.0
    m = np.zeros((4, 4, 2, 1), bool)
    m[0, 0] = True
    assert psnr(x, y, m) == pytest.approx(10 * math.log10(4 / 1.0))


def test_temporal_error_ignores_constant_offset():
    g = np.random.default_rng(1).normal(size=(5, 5, 2, 6))
    assert temporal_error(g, g) == 0.0
    assert temporal_error(g + 0.3, g) == pytest.approx(0.0, abs=1e-12)
    assert temporal_error(g * 2, g) > 0


def test_interpolation_exact_on_linear_data():
    t = np.arange(10, dtype=np.float32)
    block = np.broadcast_to(2 * t + 1, (3, 3, 2, 10)).copy()
    known = [0, 4, 8, 9]
    out = interpolate_frames(block[..., known], known, np.arange(10))
    assert np.allclose(out, block)


def test_dice_jaccard_and_surface_distance():
    a = np.zeros((20, 20, 6), bool)
    a[5:15, 5:15, 1:4] = True
    assert dice(a, a) == 1.0 and jaccard(a, a) == 1.0
    b = np.roll(a, 1, axis=2)                    # one slice further, slices 10 mm apart
    m = compare(a, b, (1.5, 1.5, 10.0))
    assert m["hd"] == pytest.approx(10.0)
    assert 0 < m["dice"] < 1


def test_mesh_volume_matches_mask_volume():
    z, y, x = np.mgrid[-20:21, -20:21, -20:21]
    ball = x ** 2 + y ** 2 + z ** 2 <= 15 ** 2
    verts, faces = H3.mesh(ball, (1.0, 1.0, 1.0), smooth=0, polish=0)
    assert H3.mesh_volume(verts, faces) == pytest.approx(ball.sum(), rel=0.03)


def test_ejection_fraction():
    edv, esv, ef = H3.ejection_fraction([200, 150, 100, 180])
    assert (edv, esv) == (200, 100) and ef == pytest.approx(50.0)
