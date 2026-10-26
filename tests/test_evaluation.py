import numpy as np
import torch

from inr4d.data import sampling as S
from inr4d.data.dataset import CoordinateDataset
from inr4d.evaluation.baselines import interpolate_frames, interpolate_slices, linear_along_axis
from inr4d.evaluation.metrics import psnr, reconstruct_volume
from inr4d.training.trainer import TensorSampler, TrainConfig, train


class Volume:
    def __init__(self, data, spacing):
        self.data = data
        self.spacing = spacing


def make_volume():
    rng = np.random.default_rng(0)
    return Volume(rng.random((6, 5, 4, 3)).astype(np.float32) * 255, (1.5, 1.5, 10.0))


def test_psnr_known_values():
    a = np.zeros((4, 4))
    assert psnr(a, a) == float("inf")
    # MSE = 0.04 on range 2 -> 10*log10(4/0.04) = 20 dB
    assert abs(psnr(a + 0.2, a) - 20.0) < 1e-9


def test_linear_interpolation_is_exact_on_linear_data():
    x = np.arange(7, dtype=np.float32)[None, None, :, None] * 3.0 + np.zeros((2, 2, 7, 1))
    known = [0, 2, 4, 6]
    out = linear_along_axis(x[:, :, known], known, [1, 3, 5], axis=2)
    np.testing.assert_allclose(out, x[:, :, [1, 3, 5]])
    assert interpolate_slices(x[:, :, known], known, [3]).shape == (2, 2, 1, 1)
    assert interpolate_frames(np.moveaxis(x, 2, 3)[..., known], known, [3]).shape == (2, 2, 1, 1)


def test_normaliser_matches_coordinate_dataset():
    vol = make_volume()
    ds = CoordinateDataset(vol)
    norm = S.Normaliser.from_volume(vol)
    sub = S.select(vol, norm)
    np.testing.assert_allclose(sub.coords, ds.coords.numpy(), atol=1e-6)
    np.testing.assert_allclose(sub.values, ds.values.numpy(), atol=1e-6)


def test_split_alternate():
    train_idx, test_idx = S.split_alternate(10)
    assert test_idx.tolist() == [1, 3, 5, 7]
    assert train_idx.tolist() == [0, 2, 4, 6, 8, 9]


def test_training_overfits_a_tiny_volume():
    from inr4d.models import build_field

    vol = make_volume()
    norm = S.Normaliser.from_volume(vol)
    sub = S.select(vol, norm)
    field = build_field("siren", "raw", hidden_features=64, hidden_layers=2)
    before = psnr(reconstruct_volume(field, sub.coords[:, None, :])[:, 0], sub.values[:, 0])
    train(field, TensorSampler(sub.coords, sub.values), TrainConfig(steps=300, batch_size=len(sub.coords),
                                                                     lr=1e-3, lr_final=1e-4), verbose=False)
    after = psnr(reconstruct_volume(field, sub.coords[:, None, :])[:, 0], sub.values[:, 0])
    assert after > before + 10
    assert torch.isfinite(torch.tensor(after))


def test_split_stride():
    train_idx, test_idx = S.split_stride(30, 3)
    assert train_idx.tolist() == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]
    assert len(train_idx) + len(test_idx) == 30
    assert S.split_stride(10, 2)[1].tolist() == S.split_alternate(10)[1].tolist()
