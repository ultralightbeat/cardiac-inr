"""Quality metrics and reconstruction.

All intensities are assumed to be normalised to [-1, 1] (see data/sampling.py),
so the data range is 2.0:
    PSNR = 10 * log10(data_range^2 / MSE)   (higher is better; +10 dB = 10x lower MSE)
SSIM is computed per 2D slice (in-plane) with scikit-image and averaged ("ssim_2d_mean";
scikit-image defaults: 7x7 uniform window, sample covariance, K1 = 0.01, K2 = 0.03).
No clipping is applied before any metric.
Temporal error (4D): MSE of the frame-to-frame change, ((p[t+1] - p[t]) - (g[t+1] - g[t]))^2,
which penalises flicker that per-frame PSNR/SSIM cannot see.
"""

from __future__ import annotations

import numpy as np

DATA_RANGE = 2.0
PROTOCOL = {"normalization": "min-max of the 4D scan to [-1, 1]", "data_range": DATA_RANGE,
            "clip_policy": "none", "ssim": "ssim_2d_mean: per in-plane 2D image, skimage defaults "
            "(win_size=7, uniform window, use_sample_covariance=True, K1=0.01, K2=0.03)",
            "temporal_error": "MSE of frame-to-frame differences inside the heart mask"}


def mse(pred, target, mask=None) -> float:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if not (np.isfinite(pred).all() and np.isfinite(target).all()):
        raise ValueError("NaN or Inf in prediction or target")
    diff = (pred - target) ** 2
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, bool), diff.shape)
        diff = diff[mask]
    if diff.size == 0:
        raise ValueError("empty mask")
    return float(diff.mean())


def psnr(pred, target, mask=None, data_range=DATA_RANGE) -> float:
    m = mse(pred, target, mask)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / m))


def ssim(pred, target, data_range=DATA_RANGE) -> float:
    """Mean SSIM over all 2D images of arrays shaped [H, W, ...]."""
    from skimage.metrics import structural_similarity

    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    h, w = pred.shape[:2]
    p = pred.reshape(h, w, -1)
    t = target.reshape(h, w, -1)
    vals = [structural_similarity(t[:, :, i], p[:, :, i], data_range=data_range)
            for i in range(p.shape[2])]
    return float(np.mean(vals))


def temporal_error(pred, target, mask=None) -> float:
    """pred, target [..., T] over consecutive frames: MSE of the frame-to-frame differences."""
    dp = np.diff(np.asarray(pred, np.float64), axis=-1)
    dg = np.diff(np.asarray(target, np.float64), axis=-1)
    return mse(dp, dg, None if mask is None else np.broadcast_to(np.asarray(mask, bool)[..., :1], dp.shape))


def evaluate(pred, target, heart=None) -> dict:
    """PSNR/SSIM on the whole block and PSNR on the heart region only.
    heart: bool array broadcastable to pred (e.g. [h, w, nz, 1])."""
    s2 = ssim(pred, target)
    out = {"psnr": psnr(pred, target), "ssim": s2, "ssim_2d_mean": s2, "crop_voxels": int(np.size(pred))}
    if heart is not None:
        out["psnr_heart"] = psnr(pred, target, heart)
        out["heart_voxels"] = int(np.broadcast_to(np.asarray(heart, bool), np.shape(pred)).sum())
    return out


def predict(model, coords, chunk=65536, device=None) -> np.ndarray:
    """Run the network on coords [N, d] in chunks -> np.ndarray [N] (on the model's device by default)."""
    import torch

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), chunk):
            c = torch.as_tensor(coords[i:i + chunk], dtype=torch.float32, device=device)
            out.append(model(c).reshape(-1).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def reconstruct_volume(model, grid, chunk=65536, device=None) -> np.ndarray:
    """Query the network on a coordinate grid [..., 4] (from Normaliser.grid) and
    return intensities with the grid's leading shape, e.g. [H, W, D, T]."""
    grid = np.asarray(grid, np.float32)
    return predict(model, grid.reshape(-1, grid.shape[-1]), chunk, device).reshape(grid.shape[:-1])
