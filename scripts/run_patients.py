"""Multi-patient evaluation with settings fixed in advance (meant for a GPU, e.g. Google Colab T4).

Settings were chosen on the development patients (051, 015) and are NOT tuned here:
SIREN 4 x 256, omega 30, time scale 0.1, 20,000 steps; same split for every patient
(every 4th frame + the last one known, the rest hidden). Patients 051 and 015 are excluded from
the held-out set by default.

Per patient:
  1. E2 on hidden frames: linear interpolation, SIREN, plain ReLU, and the 50/50 blend
     (run_experiments.py, results/<patient>/experiments/multi_e2/).
  2. Geometry (validate_geometry.py): the final field trained on all frames (if
     models/<patient>/field.pt does not exist yet), Dice / HD95 / ASSD against the expert labels,
     EDV / ESV / EF errors at the ED / ES frames from Info.cfg.
Summary over patients: mean, std, 95% CI (t-distribution over patients), paired differences
SIREN - interpolation and blend - interpolation with a Wilcoxon signed-rank test.

    python scripts/run_patients.py                                  # default held-out set
    python scripts/run_patients.py --patients data/acdc/patient016 data/acdc/patient022
    python scripts/run_patients.py --quick                          # smoke test
Output: results/multi_patient/MULTI_PATIENT.md and multi_patient.json. Restart-safe.
"""

from __future__ import annotations

import argparse
import glob
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
from inr4d.data import sampling as S  # noqa: E402

# held-out: 5 dilated cardiomyopathy (DCM), 5 hypertrophic (HCM), 4 previous infarction (MINF)
DEFAULT = ["016", "017", "018", "019", "020", "022", "025", "030", "035", "040", "041", "045", "048", "050"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", nargs="+", default=None, help="patient folders (default: held-out list)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-geometry", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def ci95(x):
    from scipy import stats

    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan"), n
    sd = float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return float(x.mean()), sd, h, n


def main():
    a = parse_args()
    pats = a.patients or [os.path.join("data", "acdc", f"patient{n}") for n in DEFAULT]
    pats = [p for p in pats if glob.glob(os.path.join(ROOT, p, "*_4d.nii*")) or glob.glob(os.path.join(p, "*_4d.nii*"))]
    if not pats:
        raise SystemExit("no patients found (put ACDC folders into data/acdc/)")
    q = "quick_" if a.quick else ""
    steps = 150 if a.quick else a.steps
    t0 = time.time()
    rows, md = [], "no patient finished"
    for p in pats:
        name = os.path.basename(os.path.normpath(p))
        info = S.read_info(os.path.join(ROOT, p) if not os.path.isabs(p) else p)
        e2_dir = os.path.join(ROOT, "results", name, "experiments", f"{q}multi_e2")
        if not os.path.exists(os.path.join(e2_dir, "results.json")):
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiments.py"), "--patient", p,
                   "--exp", "e2", "--frame-stride", "4", "--encodings", "scaled", "--time-scale", "0.1",
                   "--e2-models", "siren", "relu", "--steps-e2", str(steps), "--name", f"{q}multi_e2",
                   "--device", a.device] + (["--quick"] if a.quick else [])
            try:
                run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"!! {name}: experiments failed ({e}); skipped")
                continue
        with open(os.path.join(e2_dir, "results.json"), encoding="utf-8") as f:
            e2 = json.load(f)["e2"]["methods"]
        row = {"patient": name, "group": info.get("Group"), "frames": None}
        for key, m in e2.items():
            short = ("interpolation" if key == "interpolation" else "blend" if key.startswith("blend_siren")
                     else "siren" if key.startswith("siren") else "relu" if key.startswith("relu") else None)
            if short:
                row[short] = {"psnr_heart": m["psnr_heart"], "psnr": m["psnr"], "ssim": m["ssim"],
                              "temporal_error_heart": m.get("temporal_error_heart")}
        gdir = os.path.join(ROOT, "results", name, f"{q}validation")
        if not a.skip_geometry:
            gpath = os.path.join(gdir, "geometry.json")
            if not os.path.exists(gpath):
                try:
                    run([sys.executable, os.path.join(ROOT, "scripts", "validate_geometry.py"), "--patient", p,
                         "--out", gdir, "--device", a.device, "--train-steps", str(steps)]
                        + (["--checkpoint", os.path.join("models", name, "quick", "field.pt")] if a.quick else []))
                except subprocess.CalledProcessError as e:
                    print(f"!! {name}: geometry failed ({e})")
            if os.path.exists(gpath):
                with open(gpath, encoding="utf-8") as f:
                    g = json.load(f)
                base = next(iter(g["configs"].values()))
                row["geometry"] = {k: {"dice_ED": base["overlap"][k]["ED"]["dice"],
                                       "dice_ES": base["overlap"][k]["ES"]["dice"],
                                       "hd95_ED": base["overlap"][k]["ED"]["hd95"],
                                       "assd_ED": base["overlap"][k]["ED"]["assd"],
                                       "EF": base["volumes"][k]["EF"], "EF_err": base["volumes"][k]["EF_err"],
                                       "EF_expert": base["volumes"][k]["EF"] - base["volumes"][k]["EF_err"]}
                                   for k in ("LV", "RV")}
                ref = g["reference_threshold_on_scanner_slices"]
                row["geometry_scanner_threshold"] = {k: {"dice_ED": ref["overlap"][k]["ED"]["dice"],
                                                         "EF_err": ref["volumes"][k]["EF_err"]} for k in ("LV", "RV")}
        rows.append(row)
        md = save(rows, q)                    # updated after every patient: stopping early still leaves a table
    print(md)
    print(f"\n{len(rows)} patients in {(time.time() - t0) / 60:.0f} min -> results/{q}multi_patient")


def save(rows, q):
    out = os.path.join(ROOT, "results", f"{q}multi_patient")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "multi_patient.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    md = report(rows)
    with open(os.path.join(out, "MULTI_PATIENT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def report(rows):
    from scipy import stats

    L = ["# Held-out patients: hidden frames (E2) and geometry", "",
         "Settings fixed on the development patient (051) before this run: SIREN 4 x 256, omega 30, "
         "time scale 0.1, 20,000 steps, blend = 0.5 SIREN + 0.5 interpolation. Hidden test frames: all except "
         "every 4th and the last. PSNR in the heart (dB). Geometry: label-assisted threshold on the final "
         "field (trained on all frames), ED / ES from Info.cfg, Dice on the scanner slices.", "",
         "## Per patient", "",
         "| patient | group | interp. | SIREN | ReLU | blend | SIREN - interp. | LV Dice ED | LV EF (expert) | LV EF err | RV EF err |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def g(r, *keys):
        v = r
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                return None
            v = v[k]
        return v

    def f(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    for r in rows:
        si, it = g(r, "siren", "psnr_heart"), g(r, "interpolation", "psnr_heart")
        L.append(f"| {r['patient']} | {r['group'] or '-'} | {f(it)} | {f(si)} | {f(g(r, 'relu', 'psnr_heart'))} | "
                 f"{f(g(r, 'blend', 'psnr_heart'))} | {f(None if si is None or it is None else si - it, '{:+.2f}')} | "
                 f"{f(g(r, 'geometry', 'LV', 'dice_ED'), '{:.3f}')} | {f(g(r, 'geometry', 'LV', 'EF'), '{:.1f}')} "
                 f"({f(g(r, 'geometry', 'LV', 'EF_expert'), '{:.1f}')}) | {f(g(r, 'geometry', 'LV', 'EF_err'), '{:+.1f}')} | "
                 f"{f(g(r, 'geometry', 'RV', 'EF_err'), '{:+.1f}')} |")
    L += ["", "## Over patients (mean, std, 95% CI over patients)", "",
          "| quantity | n | mean | std | 95% CI |", "|---|---|---|---|---|"]
    items = [("PSNR heart, interpolation", lambda r: g(r, "interpolation", "psnr_heart")),
             ("PSNR heart, SIREN", lambda r: g(r, "siren", "psnr_heart")),
             ("PSNR heart, ReLU", lambda r: g(r, "relu", "psnr_heart")),
             ("PSNR heart, blend", lambda r: g(r, "blend", "psnr_heart")),
             ("SIREN - ReLU (paired)", lambda r: None if g(r, "relu", "psnr_heart") is None else
              g(r, "siren", "psnr_heart") - g(r, "relu", "psnr_heart")),
             ("SIREN - interpolation (paired)", lambda r: None if g(r, "siren", "psnr_heart") is None else
              g(r, "siren", "psnr_heart") - g(r, "interpolation", "psnr_heart")),
             ("blend - interpolation (paired)", lambda r: None if g(r, "blend", "psnr_heart") is None else
              g(r, "blend", "psnr_heart") - g(r, "interpolation", "psnr_heart")),
             ("temporal error x1e3, interpolation", lambda r: None if g(r, "interpolation", "temporal_error_heart") is None
              else 1e3 * g(r, "interpolation", "temporal_error_heart")),
             ("temporal error x1e3, SIREN", lambda r: None if g(r, "siren", "temporal_error_heart") is None
              else 1e3 * g(r, "siren", "temporal_error_heart")),
             ("LV Dice ED", lambda r: g(r, "geometry", "LV", "dice_ED")),
             ("LV Dice ES", lambda r: g(r, "geometry", "LV", "dice_ES")),
             ("LV HD95 ED, mm", lambda r: g(r, "geometry", "LV", "hd95_ED")),
             ("LV EF error, points", lambda r: g(r, "geometry", "LV", "EF_err")),
             ("LV absolute EF error, points", lambda r: None if g(r, "geometry", "LV", "EF_err") is None
              else abs(g(r, "geometry", "LV", "EF_err"))),
             ("RV Dice ED", lambda r: g(r, "geometry", "RV", "dice_ED")),
             ("RV EF error, points", lambda r: g(r, "geometry", "RV", "EF_err")),
             ("LV Dice ED, same threshold on scanner slices", lambda r: g(r, "geometry_scanner_threshold", "LV", "dice_ED"))]
    for label, fn in items:
        m, sd, h, n = ci95([fn(r) for r in rows])
        L.append(f"| {label} | {n} | {m:.3f} | {sd:.3f} | {m - h:.3f} ... {m + h:.3f} |")
    L += ["", "## Paired tests (Wilcoxon signed-rank, two-sided)", ""]
    for label, a_key, b_key in (("SIREN vs interpolation", "siren", "interpolation"),
                                ("blend vs interpolation", "blend", "interpolation"),
                                ("SIREN vs ReLU", "siren", "relu")):
        d = [g(r, a_key, "psnr_heart") - g(r, b_key, "psnr_heart") for r in rows
             if g(r, a_key, "psnr_heart") is not None and g(r, b_key, "psnr_heart") is not None]
        if len(d) >= 5:
            p = stats.wilcoxon(d).pvalue
            L.append(f"- {label}: better on {sum(x > 0 for x in d)} of {len(d)} patients, "
                     f"median difference {np.median(d):+.2f} dB, p = {p:.3g}")
        else:
            L.append(f"- {label}: too few patients for a test ({len(d)})")
    L += ["", "Patients are the unit of analysis (voxels are not independent observations). "
          "One seed per patient; seed spread on patient 051 is reported in VALIDITY.md."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
