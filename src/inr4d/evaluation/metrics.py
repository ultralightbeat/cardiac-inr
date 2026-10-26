"""Quality metrics and reconstruction.

All intensities are assumed to be normalised to [-1, 1] (see data/sampling.py),
so the data range is 2.0:
    PSNR = 10 * log10(data_range^2 / MSE)   (higher is better; +10 dB = 10x lower MSE)
SSIM is computed per 2D slice (in-plane) with scikit-image and averaged.
"""

from __future__ import annotations

import numpy as np

DATA_RANGE = 2.0


def mse(pred, target, mask=None) -> float:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    diff = (pred - target) ** 2
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, bool), diff.shape)
        diff = diff[mask]
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


def evaluate(pred, target, heart=None) -> dict:
    """PSNR/SSIM on the whole block and PSNR on the heart region only.
    heart: bool array broadcastable to pred (e.g. [h, w, nz, 1])."""
    out = {"psnr": psnr(pred, target), "ssim": ssim(pred, target)}
    if heart is not None:
        out["psnr_heart"] = psnr(pred, target, heart)
    return out


def predict(model, coords, chunk=65536, device="cpu") -> np.ndarray:
    """Run the network on coords [N, d] in chunks -> np.ndarray [N]."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), chunk):
            c = torch.as_tensor(coords[i:i + chunk], dtype=torch.float32, device=device)
            out.append(model(c).reshape(-1).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def reconstruct_volume(model, grid, chunk=65536, device="cpu") -> np.ndarray:
    """Query the network on a coordinate grid [..., 4] (from Normaliser.grid) and
    return intensities with the grid's leading shape, e.g. [H, W, D, T]."""
    grid = np.asarray(grid, np.float32)
    return predict(model, grid.reshape(-1, grid.shape[-1]), chunk, device).reshape(grid.shape[:-1])
