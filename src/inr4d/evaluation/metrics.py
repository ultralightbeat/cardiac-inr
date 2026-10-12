"""Roles 4 and 5: metrics and reconstruction."""

# 指标实现需要明确输入数据的强度范围。

def psnr(pred, target) -> float:
    """TODO(Role 4). State the intensity range you assume."""
    raise NotImplementedError


def ssim(pred, target) -> float:
    """TODO(Role 4)."""
    raise NotImplementedError


def reconstruct_volume(model, shape_hwd, t, device="cpu"):
    """Query the network on a full grid at time t -> np.ndarray [H, W, D]. TODO(Role 4)."""
    raise NotImplementedError
