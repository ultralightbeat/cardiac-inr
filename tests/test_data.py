import numpy as np
import pytest
import torch

from inr4d.data.dataset import CoordinateDataset


class Volume:
    def __init__(self, data, spacing):
        self.data = data
        self.spacing = spacing


def make_volume():
    data = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    return Volume(data, (1.0, 2.0, 10.0))


def test_coordinates_values_and_frame_ids_are_well_formed():
    # 合同要求坐标、强度和帧编号具有固定形状并且都是有限值。
    dataset = CoordinateDataset(make_volume())
    assert dataset.coords.shape == (72, 4)
    assert dataset.values.shape == (72, 1)
    assert dataset.frame_ids.shape == (72,)
    assert torch.all((dataset.coords >= -1) & (dataset.coords <= 1))
    assert torch.all((dataset.values >= -1) & (dataset.values <= 1))
    assert torch.isfinite(dataset.coords).all()
    assert torch.isfinite(dataset.values).all()


def test_coordinates_use_physical_spacing():
    # 厚层间距必须影响 z 坐标，避免沿 z 轴错误拉伸解剖结构。
    dataset = CoordinateDataset(make_volume())
    z_values = dataset.coords[:, 2].reshape(2, 3, 4, 3)[0, 0, :, 0]
    assert torch.allclose(z_values, torch.tensor([-1.0, -1 / 3, 1 / 3, 1.0]))
    x_value = dataset.coords[:, 0].reshape(2, 3, 4, 3)[0, 0, 0, 0]
    assert torch.isclose(x_value, torch.tensor(-1 / 30))


def test_frame_subset_and_random_sampling():
    # 选帧功能支持留出某个时间帧进行实验。
    dataset = CoordinateDataset(make_volume(), frames=[2, 0])
    assert set(dataset.frame_ids.tolist()) == {0, 2}
    coords, values = dataset.sample(1024)
    assert coords.shape == (1024, 4)
    assert values.shape == (1024, 1)


def test_round_trip_reconstructs_selected_volume():
    # 归一化后的样本应能恢复回原始合成体数据。
    volume = make_volume()
    dataset = CoordinateDataset(volume)
    shape = volume.data.shape
    reconstructed = np.empty(shape, dtype=np.float32)
    coords = dataset.coords.reshape(shape[0], shape[1], shape[2], shape[3], 4)
    values = dataset.values.reshape(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                for t in range(shape[3]):
                    reconstructed[x, y, z, t] = float(values[x, y, z, t])
    minimum, maximum = volume.data.min(), volume.data.max()
    reconstructed = (reconstructed + 1) * (maximum - minimum) / 2 + minimum
    np.testing.assert_allclose(reconstructed, volume.data, rtol=1e-6, atol=1e-5)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        CoordinateDataset(Volume(np.full((2, 2, 2, 1), np.nan, dtype=np.float32), (1, 1, 1)))
