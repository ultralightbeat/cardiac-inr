"""Figures: slice comparisons with error maps, long-axis reslices, bar charts,
per-frame curves, training curves and GIF animations.

Images are rotated by 90 degrees for display only (the usual radiological view of
ACDC short-axis slices); the data itself is never changed.
"""

from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _show(x):
    return np.rot90(np.asarray(x))


def slice_comparison(path, gt, panels: dict, title=""):
    """gt [h, w]; panels {name: (image [h, w], psnr or None)}. Row 1 images, row 2 |error|."""
    plt = _plt()
    names = list(panels)
    n = len(names) + 1
    fig, ax = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    ax[0, 0].imshow(_show(gt), cmap="gray", vmin=-1, vmax=1)
    ax[0, 0].set_title("ground truth")
    ax[1, 0].axis("off")
    for i, name in enumerate(names, start=1):
        img, score = panels[name]
        ax[0, i].imshow(_show(img), cmap="gray", vmin=-1, vmax=1)
        ax[0, i].set_title(f"{name}" + (f"\n{score:.1f} dB" if score is not None else ""))
        ax[1, i].imshow(_show(np.abs(img - gt)), cmap="magma", vmin=0, vmax=0.5)
        ax[1, i].set_title("|error|")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def long_axis(path, measured, dense, row, spacing, dense_slices, title=""):
    """Cut across the slices at a fixed image row.
    measured [h, w, D] (the 10 scanned slices), dense [h, w, Dd] (network queried at
    fractional slice positions dense_slices). Physical aspect ratio in mm."""
    plt = _plt()
    sx, sz = float(spacing[1]), float(spacing[2])
    w = measured.shape[1]
    D = measured.shape[2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].imshow(measured[row].T, cmap="gray", vmin=-1, vmax=1, interpolation="nearest",
                 extent=[0, w * sx, (D - 0.5) * sz, -0.5 * sz], aspect="equal")
    ax[0].set_title(f"scanner: {D} slices, {sz:g} mm apart")
    step = (dense_slices[1] - dense_slices[0]) * sz if len(dense_slices) > 1 else sz
    top, bottom = dense_slices[0] * sz - step / 2, dense_slices[-1] * sz + step / 2
    ax[1].imshow(dense[row].T, cmap="gray", vmin=-1, vmax=1, interpolation="nearest",
                 extent=[0, w * sx, bottom, top], aspect="equal")
    ax[1].set_title(f"neural field: {len(dense_slices)} slices, {step:.1f} mm apart")
    for a in ax:
        a.set_ylim((D - 0.5) * sz, -0.5 * sz)
        a.set_xlabel("mm")
        a.set_ylabel("mm (base to apex)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bars(path, rows: dict, keys=("psnr", "psnr_heart"), title=""):
    """rows {method: metrics dict}."""
    plt = _plt()
    names = list(rows)
    x = np.arange(len(names))
    wdt = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
    for i, k in enumerate(keys):
        vals = [rows[n].get(k, np.nan) for n in names]
        b = ax.bar(x + i * wdt - 0.4 + wdt / 2, vals, wdt, label=k)
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("PSNR, dB (higher is better)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_frame(path, frames, curves: dict, title=""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, vals in curves.items():
        ax.plot(frames, vals, marker="o", label=name)
    ax.set_xlabel("hidden frame")
    ax.set_ylabel("PSNR, dB")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_curves(path, histories: dict, title=""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, h in histories.items():
        ax.plot(h["step"], h["train_psnr"], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("training PSNR (running), dB")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gif(path, frames, fps=20, size=320):
    """frames [n, h, w] in [-1, 1] -> animated GIF."""
    from PIL import Image

    imgs = []
    for f in frames:
        u = np.clip((_show(f) + 1) * 127.5, 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(u).resize((size, size), Image.BICUBIC))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
