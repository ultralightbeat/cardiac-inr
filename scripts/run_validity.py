"""All validity checks in one command (one patient, CPU, ~3-4 h; safe to stop and restart:
finished runs are skipped).

  A. Validation protocol. Frames 12 and 20 are moved from the training frames to a VALIDATION
     set; the test frames stay the hidden ones (stride 4). Settings are chosen on validation only:
       SIREN time scale     0.05 / 0.1 / 0.25
       ReLU + Fourier features, sigma 3 / 10 / 30   (64 frequencies)
  B. Final runs on the standard split (all 9 training frames, 21 hidden test frames), with the
     chosen settings: SIREN with seeds 0, 1, 2; plain ReLU; ReLU + Fourier features.
     Every run stores exact train PSNR, temporal error, normalisation and label-use metadata.
  C. Geometry: Dice / Hausdorff / surface distance against the expert labels, EDV / ESV / EF
     errors, threshold / ROI / grid sensitivity, mesh smoothing sweep (validate_geometry.py).
  D. Summary: results/<patient>/validation/VALIDITY.md

    python scripts/run_validity.py --quick     # 3-5 min smoke test first
    python scripts/run_validity.py             # the real thing (leave it overnight)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from inr4d.data import patients as P  # noqa: E402

VAL_FRAMES = [12, 20]
TIME_SCALES = [0.05, 0.1, 0.25]
SIGMAS = [3.0, 10.0, 30.0]
SEEDS = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default="data/acdc/patient051")
    p.add_argument("--steps-val", type=int, default=6000, help="steps of the validation runs")
    p.add_argument("--steps", type=int, default=20000, help="steps of the final runs (as in the report)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", help="cpu / cuda / xpu / auto")
    p.add_argument("--quick", action="store_true", help="tiny networks and few steps: checks that all works")
    return p.parse_args()


def experiment(a, name, extra):
    """Run run_experiments.py once (skipped if its results.json exists); return its results."""
    out = P.results_dir(a.patient, "experiments/" + name)
    path = os.path.join(out, "results.json")
    if not os.path.exists(path):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiments.py"), "--patient", a.patient,
               "--exp", "e2", "--frame-stride", "4", "--encodings", "scaled", "--name", name] + extra
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        cmd += ["--device", a.device]
        if a.quick:
            cmd += ["--quick"]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    else:
        print(f"skip {name} (done)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["e2"]


def model_of(e2, prefix):
    return next(v for k, v in e2["methods"].items() if k.startswith(prefix))


def main():
    a = parse_args()
    a.patient = P.resolve_patient(a.patient)
    q = "quick_" if a.quick else ""
    steps_val = 150 if a.quick else a.steps_val
    steps = 150 if a.quick else a.steps
    t0 = time.time()
    val = ["--val-frames"] + [str(v) for v in VAL_FRAMES]

    # A. validation sweeps
    sweep_ts = {}
    for ts in TIME_SCALES:
        e2 = experiment(a, f"{q}val_siren_t{ts:g}", val + ["--e2-models", "siren", "--time-scale", str(ts),
                                                           "--steps-e2", str(steps_val)])
        sweep_ts[ts] = model_of(e2, "siren")
        a.interp_val = e2["methods"]["interpolation"]["val"]
    sweep_sigma = {}
    for sg in SIGMAS:
        e2 = experiment(a, f"{q}val_fourier_s{sg:g}", val + ["--e2-models", "fourier", "--time-scale", "0.1",
                                                             "--fourier-sigma", str(sg), "--steps-e2", str(steps_val)])
        sweep_sigma[sg] = model_of(e2, "fourier")
    best_ts = max(sweep_ts, key=lambda k: sweep_ts[k]["val"]["psnr_heart"])
    best_sg = max(sweep_sigma, key=lambda k: sweep_sigma[k]["val"]["psnr_heart"])
    print(f"\nchosen on validation: SIREN time scale {best_ts:g}, Fourier sigma {best_sg:g}")

    # B. final runs on the standard split
    common = ["--time-scale", str(best_ts), "--steps-e2", str(steps)]
    siren = {s: experiment(a, f"{q}final_siren_seed{s}", common + ["--e2-models", "siren", "--seed", str(s)])
             for s in SEEDS}
    relu = experiment(a, f"{q}final_relu", common + ["--e2-models", "relu"])
    fourier = experiment(a, f"{q}final_fourier", common + ["--e2-models", "fourier", "--fourier-sigma", str(best_sg)])

    # C. geometry
    vdir = P.results_dir(a.patient, f"{q}validation")
    os.makedirs(vdir, exist_ok=True)
    gpath = os.path.join(vdir, "geometry.json")
    if not os.path.exists(gpath):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "validate_geometry.py"), "--patient", a.patient,
               "--out", vdir, "--device", a.device]
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    # D. summary
    md = summary(a, sweep_ts, sweep_sigma, best_ts, best_sg, siren, relu, fourier)
    if os.path.exists(os.path.join(vdir, "GEOMETRY.md")):
        with open(os.path.join(vdir, "GEOMETRY.md"), encoding="utf-8") as f:
            md += "\n" + f.read().replace("# Geometry checks", "## Geometry checks")
    with open(os.path.join(vdir, "VALIDITY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\nall done in {(time.time() - t0) / 60:.0f} min -> {vdir}")


def ci95(x):
    """mean, std, half-width of the 95% t-interval (n = number of seeds)."""
    x = np.asarray(x, float)
    n = len(x)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return float(x.mean()), sd, t * sd / math.sqrt(n) if n > 1 else float("nan")


def summary(a, sweep_ts, sweep_sigma, best_ts, best_sg, siren, relu, fourier):
    s0 = siren[SEEDS[0]]
    L = [f"# Validity checks: {P.patient_name(a.patient)}", "",
         "One patient (ACDC 051), CPU. Hidden test frames: every frame except 0, 4, ..., 28, 29 "
         f"(21 frames). Validation frames {VAL_FRAMES} (taken out of the training frames, used only to choose "
         "settings). PSNR in dB, data range 2, intensities min-max normalised over the scan (identical to the "
         f"training-frames-only min/max: {s0.get('vmin_vmax_training_frames_only')}), no clipping. "
         "SSIM = mean 2D SSIM, skimage defaults. Temporal error = MSE of frame-to-frame differences "
         "in the heart over the whole cycle (lower is better).", "",
         "## A. Settings chosen on validation frames only", "",
         "| model | setting | val PSNR heart | val PSNR crop |", "|---|---|---|---|",
         f"| linear interpolation | - | {a.interp_val['psnr_heart']:.2f} | {a.interp_val['psnr']:.2f} |"]
    for ts, m in sweep_ts.items():
        L.append(f"| SIREN | time scale {ts:g}{' **chosen**' if ts == best_ts else ''} | {m['val']['psnr_heart']:.2f} | "
                 f"{m['val']['psnr']:.2f} |")
    for sg, m in sweep_sigma.items():
        L.append(f"| ReLU + Fourier (64 freq.) | sigma {sg:g}{' **chosen**' if sg == best_sg else ''} | "
                 f"{m['val']['psnr_heart']:.2f} | {m['val']['psnr']:.2f} |")
    iv = s0["methods"]["interpolation"]
    L += ["", "## B. Test (hidden frames), chosen settings, same split / masks / metrics for all", "",
          "| method | PSNR heart | PSNR crop | SSIM 2D | temporal error x1e3 | train PSNR (exact, crop) | params | time, min |",
          "|---|---|---|---|---|---|---|---|"]

    def row(name, m):
        tr = f"{m['train_psnr']:.2f}" if m.get("train_psnr") is not None else "-"
        pr = f"{m['params']:,}" if m.get("params") else "stores 9 frames"
        tm = f"{m['seconds'] / 60:.0f}" if m.get("seconds") else "-"
        return (f"| {name} | {m['psnr_heart']:.2f} | {m['psnr']:.2f} | {m['ssim']:.3f} | "
                f"{1e3 * m['temporal_error_heart']:.2f} | {tr} | {pr} | {tm} |")
    L.append(row("linear interpolation", iv))
    for s in SEEDS:
        L.append(row(f"SIREN, seed {s}", model_of(siren[s], "siren")))
    L.append(row("ReLU (plain)", model_of(relu, "relu")))
    L.append(row(f"ReLU + Fourier features (sigma {best_sg:g})", model_of(fourier, "fourier")))
    L += ["", "## Seeds (SIREN, 3 runs that differ only in the random initialisation and batches)", "",
          "| metric | mean | std | 95% CI (t, n=3) | interpolation |", "|---|---|---|---|---|"]
    for key, label, scale in (("psnr_heart", "PSNR heart", 1), ("psnr", "PSNR crop", 1),
                              ("ssim", "SSIM 2D", 1), ("temporal_error_heart", "temporal error x1e3", 1e3)):
        vals = [scale * model_of(siren[s], "siren")[key] for s in SEEDS]
        m, sd, h = ci95(vals)
        L.append(f"| {label} | {m:.3f} | {sd:.3f} | {m - h:.3f} ... {m + h:.3f} | {scale * iv[key]:.3f} |")
    L += ["", "Seed spread is variation of the training only (one patient): it is not a confidence interval "
          "over patients. Interpolation stores the 9 known frames (all their voxels); the networks store "
          "their parameters.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
