"""Main experiments of the project (one command, CPU friendly).

E1  static 3D (level L1): one frame (end-diastole). Every second slice is hidden,
    the field is trained on the rest and must rebuild the hidden slices.
    Compared: SIREN, ReLU network of the same size, linear interpolation between
    the neighbouring slices (= trilinear interpolation of the stack).

E2  4D (level L2): all slices, every second frame hidden. Compared: SIREN with
    different time encodings (raw t, scaled t, periodic) and linear interpolation
    between the neighbouring frames.

Metrics: PSNR and SSIM on the hidden data inside an in-plane crop around the heart,
plus PSNR on the heart region only (expert labels, dilated).

Examples
    python scripts/run_experiments.py --quick                  # smoke test, ~1-2 min
    python scripts/run_experiments.py                          # E1 + E2, patient051
    python scripts/run_experiments.py --exp e2 --encodings scaled periodic --steps-e2 6000
    python scripts/run_experiments.py --patient data/acdc/patient020 --exp e1
    python scripts/run_experiments.py --patient path/to/any_scan_4d.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from inr4d.data import patients as P
from inr4d.data import sampling as S
from inr4d.data.acdc import load_acdc_patient
from inr4d.evaluation.baselines import interpolate_frames, interpolate_slices
from inr4d.evaluation.metrics import PROTOCOL, evaluate, psnr, reconstruct_volume, temporal_error
from inr4d.models import build_field, count_parameters
from inr4d.training.trainer import TensorSampler, TrainConfig, resolve_device, train
from inr4d.viz import figures as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default=None,
                   help="patient folder (ACDC layout) or a 4D NIfTI file; default: first patient under data/")
    p.add_argument("--exp", nargs="+", default=["e1", "e2"], choices=["e1", "e2"])
    p.add_argument("--out", default=None, help="output folder (default results/<patient>/experiments/<name>)")
    p.add_argument("--name", default=None, help="run name, used for the default output folder (default: main / quick)")
    p.add_argument("--models", nargs="+", default=["siren", "relu"], help="E1 models")
    p.add_argument("--e2-models", nargs="+", default=["siren"], choices=["siren", "relu", "fourier"],
                   help="E2 networks (relu = control with the same size)")
    p.add_argument("--encodings", nargs="+", default=["raw", "scaled", "periodic"], help="E2 time encodings")
    p.add_argument("--z-scale", type=float, default=1.0,
                   help="multiply the slice coordinate z by this inside the model (E1 and E2); <1 = smoother across slices")
    p.add_argument("--time-scale", type=float, default=0.1, help="t is multiplied by this (best: 0.1, see DECISIONS)")
    p.add_argument("--holdout-frame", type=int, default=None,
                   help="E2: hide only this frame (0-based) instead of every second frame")
    p.add_argument("--frame-stride", type=int, default=2,
                   help="E2: keep every k-th frame (plus the last), hide the rest. 2 = every second frame hidden")
    p.add_argument("--fourier-freqs", type=int, default=64,
                   help="'fourier' model: number of random Fourier frequencies (ReLU + Fourier features)")
    p.add_argument("--fourier-sigma", type=float, default=10.0, help="'fourier' model: std of the frequencies")
    p.add_argument("--val-frames", type=int, nargs="+", default=None,
                   help="E2: training frames moved to a validation set (used only to choose settings; "
                        "the test frames stay the hidden ones)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--steps-e1", type=int, default=3000)
    p.add_argument("--steps-e2", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pad", type=int, default=12, help="crop padding around the heart, px")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", help="cpu / cuda / xpu / auto (GPU if there is one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    a = p.parse_args()
    if a.quick:
        a.steps_e1, a.steps_e2, a.width, a.batch = 150, 150, 64, 4096
    return a


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def make_cfg(a, steps):
    return TrainConfig(steps=steps, batch_size=a.batch, lr=a.lr, lr_final=a.lr / 10,
                       log_every=max(steps // 20, 1), seed=a.seed, threads=a.threads, device=a.device)


# ----------------------------------------------------------------------------- E1

def run_e1(a, vol, norm, crop, heart3d, frame, out):
    H, W, D, T = vol.data.shape
    r0, r1, c0, c1 = crop
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    train_z, test_z = S.split_alternate(D)
    print(f"\n=== E1 static 3D | frame {frame} | train slices {train_z.tolist()} | "
          f"hidden slices {test_z.tolist()}")
    train_set = S.select(vol, norm, crop, train_z, [frame])
    gt = S.block(vol, norm, crop, test_z, [frame])
    known = S.block(vol, norm, crop, train_z, [frame])
    heart = heart3d[r0:r1, c0:c1][:, :, test_z][..., None]
    res = {"frame": int(frame), "train_slices": train_z, "test_slices": test_z,
           "train_points": len(train_set.coords), "methods": {}}

    interp = interpolate_slices(known, train_z, test_z)
    res["methods"]["interpolation"] = evaluate(interp, gt, heart)
    preds, hists, fields = {}, {}, {}
    test_grid = norm.grid(rows, cols, test_z, [frame])
    for m in a.models:
        field = build_field(m, "none", hidden_features=a.width, hidden_layers=a.layers, z_scale=a.z_scale,
                            omega_0=a.omega, seed=a.seed)
        print(f"-- {m}: {count_parameters(field):,} parameters, {len(train_set.coords):,} training points")
        r = train(field, TensorSampler(train_set.coords, train_set.values, a.seed),
                  make_cfg(a, a.steps_e1), checkpoint_path=os.path.join(out, f"e1_{m}.pt"))
        pred = reconstruct_volume(field, test_grid)
        fit = reconstruct_volume(field, norm.grid(rows, cols, train_z, [frame]))
        res["methods"][m] = {**evaluate(pred, gt, heart), "train_psnr": psnr(fit, known),
                             "params": count_parameters(field), "seconds": r.seconds}
        preds[m], hists[m], fields[m] = pred, r.history, field

    # figures: middle hidden slice, long-axis cut with 4x more slices
    k = len(test_z) // 2
    panels = {"interpolation": (interp[:, :, k, 0], res["methods"]["interpolation"]["psnr"])}
    for m in preds:
        panels[m] = (preds[m][:, :, k, 0], res["methods"][m]["psnr"])
    F.slice_comparison(os.path.join(out, "e1_hidden_slice.png"), gt[:, :, k, 0], panels,
                       f"E1: hidden slice {test_z[k]} (frame {frame})")
    if "siren" in fields:
        dense_z = np.linspace(0, D - 1, 4 * (D - 1) + 1)
        dense = reconstruct_volume(fields["siren"], norm.grid(rows, cols, dense_z, [frame]))[..., 0]
        measured = S.block(vol, norm, crop, None, [frame])[..., 0]
        row = int(np.argmax(heart3d[r0:r1, c0:c1].sum(axis=(1, 2))))
        F.long_axis(os.path.join(out, "e1_long_axis.png"), measured, dense, row, vol.spacing,
                    dense_z, "Cut across the slices: scanner vs. SIREN queried between slices")
    F.training_curves(os.path.join(out, "e1_training.png"), hists, "E1 training")
    F.bars(os.path.join(out, "e1_bars.png"), res["methods"], title="E1: hidden slices")
    return res


# ----------------------------------------------------------------------------- E2

def run_e2(a, vol, norm, crop, heart3d, out):
    H, W, D, T = vol.data.shape
    r0, r1, c0, c1 = crop
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    if a.holdout_frame is not None:
        test_t = np.array([a.holdout_frame])
        train_t = np.setdiff1d(np.arange(T), test_t)
    else:
        train_t, test_t = S.split_stride(T, a.frame_stride)
    val_t = np.array(sorted(a.val_frames or []), int)
    if len(val_t):
        if not np.isin(val_t, train_t).all():
            raise SystemExit(f"--val-frames must be training frames {train_t.tolist()}")
        train_t = np.setdiff1d(train_t, val_t)
    print(f"\n=== E2 4D | {len(train_t)} training frames | hidden frames {test_t.tolist()}")
    train_set = S.select(vol, norm, crop, None, train_t)
    gt = S.block(vol, norm, crop, None, test_t)
    known = S.block(vol, norm, crop, None, train_t)
    heart = heart3d[r0:r1, c0:c1][..., None]
    full_t = np.arange(T)
    gt_full = S.block(vol, norm, crop, None, full_t)
    res = {"vmin_vmax_training_frames_only": [float(vol.data[..., train_t].min()), float(vol.data[..., train_t].max())],
           "train_frames": train_t, "validation_frames": val_t, "test_frames": test_t, "time_scale": a.time_scale,
           "frame_stride": a.frame_stride,
           "train_points": len(train_set.coords), "methods": {}, "per_frame": {}}

    interp = interpolate_frames(known, train_t, test_t)
    res["methods"]["interpolation"] = evaluate(interp, gt, heart)
    res["methods"]["interpolation"]["temporal_error_heart"] = temporal_error(
        interpolate_frames(known, train_t, full_t), gt_full, heart)
    if len(val_t):
        res["methods"]["interpolation"]["val"] = evaluate(interpolate_frames(known, train_t, val_t),
                                                          S.block(vol, norm, crop, None, val_t), heart)
    res["per_frame"]["interpolation"] = [psnr(interp[..., i], gt[..., i], heart[..., 0])
                                         for i in range(len(test_t))]
    test_grid = norm.grid(rows, cols, np.arange(D), test_t)
    preds, hists, fields = {}, {}, {}
    for model, enc in [(m, e) for m in a.e2_models for e in a.encodings]:
        tag = f"fourier{a.fourier_freqs}s{a.fourier_sigma:g}" if model == "fourier" else model
        name = f"{tag}_{enc}" + (f"_{a.time_scale:g}" if enc in ("scaled", "periodic") else "")
        field = build_field(model, enc, a.time_scale, T, a.width, a.layers, a.omega, a.seed, z_scale=a.z_scale,
                            fourier_freqs=a.fourier_freqs, fourier_sigma=a.fourier_sigma)
        print(f"-- {name}: {count_parameters(field):,} parameters, {len(train_set.coords):,} training points")
        r = train(field, TensorSampler(train_set.coords, train_set.values, a.seed),
                  make_cfg(a, a.steps_e2), checkpoint_path=os.path.join(out, f"e2_{name}.pt"))
        pred = reconstruct_volume(field, test_grid)
        full = reconstruct_volume(field, norm.grid(rows, cols, np.arange(D), full_t))     # every frame
        fit = full[..., train_t]                                                          # exact, all training voxels
        res["methods"][name] = {**evaluate(pred, gt, heart), "params": count_parameters(field),
                                "seconds": r.seconds,
                                "train_psnr": psnr(fit, known), "train_psnr_heart": psnr(fit, known, heart),
                                "smoothed_batch_psnr": float(r.history["train_psnr"][-1]) if r.history["train_psnr"] else None,
                                "temporal_error_heart": temporal_error(full, gt_full, heart),
                                "compression": float(len(rows) * len(cols) * D * T) / count_parameters(field)}  # voxels the field represents (crop)
        # blend fixed in advance: half network, half interpolation (they err in different places)
        res["methods"][f"blend_{name}"] = evaluate(0.5 * (pred + interp), gt, heart)
        if len(val_t):
            res["methods"][name]["val"] = evaluate(full[..., val_t], S.block(vol, norm, crop, None, val_t), heart)
        res["per_frame"][name] = [psnr(pred[..., i], gt[..., i], heart[..., 0])
                                  for i in range(len(test_t))]
        preds[name], hists[name], fields[name] = pred, r.history, field

    best = max(preds, key=lambda n: res["methods"][n]["psnr_heart"])
    res["best"] = best
    # figures: the hidden frame where interpolation is worst, middle slice
    i = int(np.argmin(res["per_frame"]["interpolation"]))
    z = D // 2
    panels = {"interpolation": (interp[:, :, z, i], res["per_frame"]["interpolation"][i]),
              best: (preds[best][:, :, z, i], res["per_frame"][best][i])}
    F.slice_comparison(os.path.join(out, "e2_hidden_frame.png"), gt[:, :, z, i], panels,
                       f"E2: hidden frame {test_t[i]}, slice {z} (heart-region PSNR)")
    F.per_frame(os.path.join(out, "e2_per_frame.png"), test_t,
                {k: v for k, v in res["per_frame"].items()}, "E2: heart-region PSNR per hidden frame")
    F.bars(os.path.join(out, "e2_bars.png"), res["methods"], title="E2: hidden frames")
    F.training_curves(os.path.join(out, "e2_training.png"), hists, "E2 training")
    # animation: the best field at twice the frame rate (every second frame never seen)
    dense_t = np.linspace(0, T - 1, 2 * T - 1)
    cyc = reconstruct_volume(fields[best], norm.grid(rows, cols, [z], dense_t))[:, :, 0, :]
    F.gif(os.path.join(out, "e2_cycle.gif"), np.moveaxis(cyc, -1, 0))
    return res


# ----------------------------------------------------------------------------- report

def table(summary):
    lines = []
    for exp in ("e1", "e2"):
        if exp not in summary:
            continue
        lines += [f"\n### {exp.upper()}", "",
                  "| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR (exact, crop) | params | time, s |",
                  "|---|---|---|---|---|---|---|"]
        for name, m in summary[exp]["methods"].items():
            params = f"{m['params']:,}" if "params" in m else "-"
            secs = f"{m['seconds']:.0f}" if "seconds" in m else "-"
            fit = f"{m['train_psnr']:.2f}" if m.get("train_psnr") is not None else "-"
            lines.append(f"| {name} | {m['psnr']:.2f} | {m.get('psnr_heart', float('nan')):.2f} | "
                         f"{m['ssim']:.3f} | {fit} | {params} | {secs} |")
    return "\n".join(lines)


def main():
    a = parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.patient = P.resolve_patient(a.patient)
    a.device = resolve_device(a.device)
    patient = P.patient_name(a.patient)
    out = a.out or P.results_dir(a.patient, "experiments/" + (a.name or ("quick" if a.quick else "main")))
    os.makedirs(out, exist_ok=True)

    vol = load_acdc_patient(a.patient)
    H, W, D, T = vol.data.shape
    info = S.read_info(a.patient)
    labels = S.load_labels(a.patient)
    norm = S.Normaliser.from_volume(vol)
    heart3d = S.heart_mask(labels, (H, W, D))
    crop = (0, H, 0, W) if a.no_crop else S.heart_crop(heart3d, a.pad)
    ed = int(info.get("ED", 1)) - 1
    print(f"patient {patient}: shape {vol.data.shape}, spacing {tuple(round(s, 3) for s in vol.spacing)} mm, "
          f"group {info.get('Group')}, ED frame {ed}, crop rows {crop[0]}:{crop[1]} cols {crop[2]}:{crop[3]}, "
          f"torch {torch.__version__}, {torch.get_num_threads()} threads")

    summary = {"patient": patient, "shape": vol.data.shape, "spacing": vol.spacing, "info": info,
               "crop": crop, "args": vars(a), "git_commit": git_commit(),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "platform": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "threads": torch.get_num_threads(), "device": a.device,
               "protocol": {**PROTOCOL, "vmin": norm.vmin, "vmax": norm.vmax,
                            "labels_for_training": False,
                            "labels_for_roi": "crop around the heart and the heart mask for PSNR (heart)",
                            "labels_for_threshold": False, "labels_for_evaluation": True}}
    if "e1" in a.exp:
        summary["e1"] = run_e1(a, vol, norm, crop, heart3d, ed, out)
    if "e2" in a.exp:
        summary["e2"] = run_e2(a, vol, norm, crop, heart3d, out)

    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    md = f"# Results: {patient}\n\nCrop {crop}, commit {summary['git_commit']}\n" + table(summary) + "\n"
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
