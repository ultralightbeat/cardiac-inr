"""Re-evaluate a finished E2 run from its checkpoint (no retraining, ~1 minute).

Adds the blend 0.5 * SIREN + 0.5 * interpolation. The weight is fixed in advance
(not tuned on the hidden frames), so the comparison stays fair.

Why a blend can beat both: the two methods make different kinds of errors.
Interpolation fails where the motion between two known frames is not a straight
line (fast contraction); the field fails on fine texture it could not memorise.
Averaging two predictions with partly independent errors reduces the error.

Also prints a sanity check: PSNR of the field on its own training frames, and of
interpolation on the hidden frames, recomputed from scratch.

    python scripts/evaluate_blend.py results/e2_stride4_w512
    python scripts/evaluate_blend.py results/e2_long results/e2_stride4
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from inr4d.data import sampling as S
from inr4d.data.acdc import load_acdc_patient
from inr4d.evaluation.baselines import interpolate_frames
from inr4d.evaluation.metrics import evaluate, psnr, reconstruct_volume
from inr4d.models import build_field
from inr4d.viz import figures as F


def run(folder):
    with open(os.path.join(folder, "results.json"), encoding="utf-8") as f:
        info = json.load(f)
    a = info["args"]
    e2 = info["e2"]
    vol = load_acdc_patient(a["patient"])
    H, W, D, T = vol.data.shape
    norm = S.Normaliser.from_volume(vol)
    heart3d = S.heart_mask(S.load_labels(a["patient"]), (H, W, D))
    r0, r1, c0, c1 = crop = tuple(info["crop"])
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    train_t, test_t = np.array(e2["train_frames"]), np.array(e2["test_frames"])

    gt = S.block(vol, norm, crop, None, test_t)
    known = S.block(vol, norm, crop, None, train_t)
    heart = heart3d[r0:r1, c0:c1][..., None]
    interp = interpolate_frames(known, train_t, test_t)

    out = {"interpolation": evaluate(interp, gt, heart)}
    per_frame = {"interpolation": [psnr(interp[..., i], gt[..., i], heart[..., 0]) for i in range(len(test_t))]}
    for name in e2["methods"]:
        if not name.startswith(("siren", "relu")):
            continue
        model, enc = name.split("_")[:2]
        field = build_field(model, enc, a["time_scale"], T, a["width"], a["layers"], a["omega"], a["seed"])
        state = torch.load(os.path.join(folder, f"e2_{name}.pt"), map_location="cpu", weights_only=False)
        field.load_state_dict(state["state_dict"])
        pred = reconstruct_volume(field, norm.grid(rows, cols, np.arange(D), test_t))
        blend = 0.5 * pred + 0.5 * interp
        fit = reconstruct_volume(field, norm.grid(rows, cols, np.arange(D), train_t))
        out[name] = {**evaluate(pred, gt, heart), "train_psnr": psnr(fit, known)}
        out[f"blend_{name}"] = evaluate(blend, gt, heart)
        per_frame[name] = [psnr(pred[..., i], gt[..., i], heart[..., 0]) for i in range(len(test_t))]
        per_frame[f"blend_{name}"] = [psnr(blend[..., i], gt[..., i], heart[..., 0])
                                      for i in range(len(test_t))]

    lines = [f"# Blend check: {folder}", "",
             f"training frames {train_t.tolist()}, hidden frames {test_t.tolist()}", "",
             "| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |",
             "|---|---|---|---|---|"]
    base = np.array(per_frame["interpolation"])
    for name, m in out.items():
        won = "-" if name == "interpolation" else f"{int((np.array(per_frame[name]) > base).sum())}/{len(base)}"
        lines.append(f"| {name} | {m['psnr']:.2f} | {m['psnr_heart']:.2f} | {m['ssim']:.3f} | {won} |")
    for name, m in out.items():
        if "train_psnr" in m:
            lines.append(f"\n{name}: PSNR on its own training frames {m['train_psnr']:.2f} dB")
    md = "\n".join(lines) + "\n"
    print(md)
    with open(os.path.join(folder, "blend.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(folder, "blend.json"), "w", encoding="utf-8") as f:
        json.dump({"methods": out, "per_frame": per_frame, "test_frames": test_t.tolist()}, f, indent=2)
    F.per_frame(os.path.join(folder, "blend_per_frame.png"), test_t, per_frame,
                "Heart-region PSNR per hidden frame (blend = 0.5 field + 0.5 interpolation)")


if __name__ == "__main__":
    for folder in sys.argv[1:] or ["results/e2_stride4_w512"]:
        run(folder)
