"""Checks of the 3D heart (geometry of the chambers) against the expert labels.

The 3D pipeline (render_4d.py, viz/heart3d.py) is LABEL-ASSISTED: the network is trained on
intensities only, but the blood threshold, the chamber regions and the wall region come from the
expert labels. This script measures how good and how stable the result is:

  1. Overlap with the expert labels at end-diastole (ED) and end-systole (ES), on the scanner
     slices: Dice, Jaccard, Hausdorff distance (HD, worst miss), HD95, average symmetric surface
     distance (ASSD), all distances in mm. Reference row: the same threshold on the scanner data.
  2. Volumes EDV, ESV and EF with ED / ES fixed from Info.cfg (not chosen from the curves),
     absolute and relative error against the expert volumes.
  3. Sensitivity of the volumes to the choices we made: threshold (median vs mean midpoint,
     +-0.05), chamber ROI (shape interpolation vs union of neighbouring slices), grid density
     (31 / 61 / 122 slices = dz convergence).
  4. Mesh smoothing sweep: volumes are counted on the binary mask, the smoothed mesh is only the
     picture; for Gaussian smoothing x Taubin iterations we report the mesh volume against the
     mask volume and how far the mesh surface moves from the mask boundary (mm).

Note: the final field was trained on all frames, so this is reconstruction quality, not
generalisation to hidden data (that is what run_experiments.py measures).

    python scripts/validate_geometry.py                       # patient051, models/patient051/field.pt
    python scripts/validate_geometry.py --source scanner      # same checks on the interpolated scanner stack
Outputs: results/<patient>/validation/geometry.json and GEOMETRY.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_4d as R  # noqa: E402

from inr4d.data import patients as P  # noqa: E402
from inr4d.data import sampling as S  # noqa: E402
from inr4d.data.acdc import load_acdc_patient  # noqa: E402
from inr4d.evaluation.geometry import compare, components  # noqa: E402
from inr4d.viz import heart3d as H3  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default=None)
    p.add_argument("--checkpoint", default=None, help="default models/<patient>/field.pt")
    p.add_argument("--source", choices=["field", "scanner"], default="field")
    p.add_argument("--out", default=None, help="default results/<patient>/validation")
    p.add_argument("--pad", type=int, default=12)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", help="cpu / cuda / xpu / auto")
    p.add_argument("--train-steps", type=int, default=20000,
                   help="if the patient has no models/<patient>/field.pt yet: train it with this many steps")
    return p.parse_args()


def main():
    a = parse_args()
    a.patient = P.resolve_patient(a.patient)
    out = a.out or P.results_dir(a.patient, "validation")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    vol = load_acdc_patient(a.patient)
    Hh, Ww, D, T = vol.data.shape
    sx, sy, sz = (float(s) for s in vol.spacing)
    info = S.read_info(a.patient)
    labels = S.load_labels(a.patient)
    norm = S.Normaliser.from_volume(vol)
    crop = S.heart_crop(S.heart_mask(labels, (Hh, Ww, D)), a.pad)
    r0, r1, c0, c1 = crop
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    ed, es = int(info.get("ED", 1)) - 1, int(info.get("ES", 1)) - 1
    block = S.block(vol, norm, crop)
    lab_c = {k: v[r0:r1, c0:c1] for k, v in labels.items()}
    if ed not in lab_c or es not in lab_c:
        raise SystemExit("needs expert labels at ED and ES")

    if a.source == "field":
        ckpt = a.checkpoint or os.path.join(P.model_dir(a.patient), "field.pt")
        train_new = not os.path.exists(ckpt)
        if train_new:
            print(f"no {ckpt}: training the final field on all frames ({a.train_steps} steps)")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        fa = SimpleNamespace(checkpoint=None if train_new else ckpt, threads=a.threads, width=256, layers=3,
                             omega=30.0, time_scale=0.1, seed=0, device=a.device, steps=a.train_steps,
                             batch=16384, lr=1e-4)
        field, _ = R.get_field(fa, vol, norm, crop, os.path.dirname(ckpt))
        source = R.FieldSource(field, norm, rows, cols)
    else:
        source = R.ScannerSource(block)

    cache = {}

    def dense(nz, f):
        if (nz, f) not in cache:
            z = (np.arange(nz) + 0.5) * (D / nz) - 0.5
            cache[(nz, f)] = (z, source(z, float(f)))
        return cache[(nz, f)]

    thr_med = H3.blood_threshold(block[..., ed], lab_c[ed])
    blood, muscle = block[..., ed][lab_c[ed] == 3], block[..., ed][lab_c[ed] == 2]
    thr_mean = 0.5 * (float(blood.mean()) + float(muscle.mean()))
    rois = H3.chamber_rois(labels, crop)
    spacing_scan = (sx, sy, sz)
    expert = {k: {f: lab_c[f] == code for f in (ed, es)} for k, code in H3.CHAMBERS.items()}
    expert_vol = {k: {tag: float(expert[k][f].sum() * sx * sy * sz / 1000) for tag, f in (("ED", ed), ("ES", es))}
                  for k in expert}

    def volumes_row(vols):
        row = {}
        for k in H3.CHAMBERS:
            edv, esv = vols[k][ed], vols[k][es]
            gedv, gesv = expert_vol[k]["ED"], expert_vol[k]["ES"]
            ef, gef = 100 * (edv - esv) / edv, 100 * (gedv - gesv) / gedv
            row[k] = {"EDV": edv, "ESV": esv, "EF": ef, "EDV_err": edv - gedv, "ESV_err": esv - gesv,
                      "EDV_rel_err_%": 100 * (edv - gedv) / gedv, "ESV_rel_err_%": 100 * (esv - gesv) / gesv,
                      "EF_err": ef - gef}
        return row

    def run(nz=None, thr=thr_med, roi="sdf", scanner=False):
        vols = {k: {} for k in H3.CHAMBERS}
        overlap = {k: {} for k in H3.CHAMBERS}
        raw_components = {}
        for f in (ed, es):
            if scanner:
                v, z, rr, dz = block[..., f], np.arange(D), rois, sz
            else:
                z, v = dense(nz, f)
                dz = D * sz / nz
                rr = {k: (H3.roi_interp(m, z) if roi == "sdf" else H3.roi_at(m, z)) for k, m in rois.items()}
            idx = [int(np.argmin(np.abs(z - s))) for s in range(D)]          # the scanner slices
            for k in H3.CHAMBERS:
                m = H3.segment(v, rr[k], thr)
                raw_components[f"{k}_{'ED' if f == ed else 'ES'}"] = components((v > thr) & rr[k])
                vols[k][f] = float(m.sum() * sx * sy * dz / 1000)
                overlap[k]["ED" if f == ed else "ES"] = compare(m[:, :, idx], expert[k][f], spacing_scan)
        return {"volumes": volumes_row(vols), "overlap": overlap, "components_before_largest": raw_components}

    configs = [
        ("baseline: median-midpoint threshold, shape-interpolated ROI, 61 slices x 1.64 mm",
         dict(nz=61)),
        ("threshold: mean-midpoint", dict(nz=61, thr=thr_mean)),
        ("threshold: median-midpoint - 0.05", dict(nz=61, thr=thr_med - 0.05)),
        ("threshold: median-midpoint + 0.05", dict(nz=61, thr=thr_med + 0.05)),
        ("ROI: union of neighbouring slices", dict(nz=61, roi="union")),
        ("grid: 31 slices x 3.3 mm", dict(nz=31)),
        ("grid: 122 slices x 0.8 mm", dict(nz=122)),
    ]
    res = {"patient": P.patient_name(a.patient), "source": a.source, "ED": ed, "ES": es,
           "thresholds": {"median_midpoint": thr_med, "mean_midpoint": thr_mean},
           "expert_volumes": expert_vol,
           "labels": {"labels_for_training": False, "labels_for_roi": True, "labels_for_threshold": True,
                      "labels_for_evaluation": True},
           "reference_threshold_on_scanner_slices": run(scanner=True), "configs": {}}
    for name, kw in configs:
        print(name, flush=True)
        res["configs"][name] = run(**kw)

    # mesh smoothing sweep, LV at ED on the baseline mask
    nz = 61
    z, v = dense(nz, ed)
    dz = D * sz / nz
    lv = H3.segment(v, H3.roi_interp(rois["LV"], z), thr_med)
    mask_ml = float(lv.sum() * sx * sy * dz / 1000)
    from scipy.spatial import cKDTree
    raw_v, _ = H3.mesh(lv, (sx, sy, dz), step=1, smooth=0.0, polish=0)     # unsmoothed marching cubes
    tree = cKDTree(raw_v)
    sweep = []
    for smooth in (0.0, 0.6, 1.2, 2.0):
        for polish in (0, 10, 30):
            verts, faces = H3.mesh(lv, (sx, sy, dz), step=1, smooth=smooth, polish=polish)
            d = tree.query(verts)[0]          # how far smoothing moved the surface, mm
            mv = H3.mesh_volume(verts, faces) / 1000
            sweep.append({"gaussian_sigma_vox": smooth, "taubin_iterations": polish, "mesh_ml": mv,
                          "mask_ml": mask_ml, "diff_%": 100 * (mv - mask_ml) / mask_ml,
                          "surface_shift_mean_mm": float(d.mean()), "surface_shift_max_mm": float(d.max()),
                          "used_in_viewer": smooth == 1.2 and polish == 10})
    res["mesh_sweep_LV_ED"] = sweep
    res["seconds"] = time.time() - t0

    with open(os.path.join(out, "geometry.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    md = report(res)
    with open(os.path.join(out, "GEOMETRY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Saved to {out} ({res['seconds']:.0f} s)")


def report(res):
    L = [f"# Geometry checks: {res['patient']} ({res['source']})", "",
         "Label-assisted 3D pipeline: network trained on intensities only; threshold, chamber ROI and wall "
         "region from the expert labels; expert labels are also the reference. ED / ES frames fixed from "
         f"Info.cfg (ED = frame {res['ED'] + 1}, ES = frame {res['ES'] + 1}). Overlap measured on the 10 "
         "scanner slices (voxel 1.64 x 1.64 x 10 mm). Expert labels include the papillary muscles in the "
         "cavity, the threshold does not.", "",
         "## Overlap with the expert labels", "",
         "| segmentation | chamber | frame | Dice | Jaccard | HD, mm | HD95, mm | ASSD, mm |",
         "|---|---|---|---|---|---|---|---|"]
    rows = [("threshold on scanner slices", res["reference_threshold_on_scanner_slices"])]
    base = next(iter(res["configs"]))
    rows.append(("threshold on neural field (baseline)", res["configs"][base]))
    for name, r in rows:
        for k, frames in r["overlap"].items():
            for fr, m in frames.items():
                L.append(f"| {name} | {k} | {fr} | {m['dice']:.3f} | {m['jaccard']:.3f} | {m['hd']:.1f} | "
                         f"{m['hd95']:.1f} | {m['assd']:.2f} |")
    L += ["", "## Volumes and their sensitivity to our choices", "",
          "| setting | LV EDV | LV ESV | LV EF | LV EF err | RV EDV | RV ESV | RV EF | RV EF err | LV Dice ED |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ev = res["expert_volumes"]
    g = {k: (ev[k]["ED"], ev[k]["ES"]) for k in ev}
    ef = {k: 100 * (g[k][0] - g[k][1]) / g[k][0] for k in ev}
    L.append(f"| expert labels | {g['LV'][0]:.0f} | {g['LV'][1]:.0f} | {ef['LV']:.1f}% | - | {g['RV'][0]:.0f} | "
             f"{g['RV'][1]:.0f} | {ef['RV']:.1f}% | - | 1 |")
    allrows = [("threshold on scanner slices", res["reference_threshold_on_scanner_slices"])] + \
        list(res["configs"].items())
    for name, r in allrows:
        lv, rv = r["volumes"]["LV"], r["volumes"]["RV"]
        L.append(f"| {name} | {lv['EDV']:.0f} | {lv['ESV']:.0f} | {lv['EF']:.1f}% | {lv['EF_err']:+.1f} | "
                 f"{rv['EDV']:.0f} | {rv['ESV']:.0f} | {rv['EF']:.1f}% | {rv['EF_err']:+.1f} | "
                 f"{r['overlap']['LV']['ED']['dice']:.3f} |")
    L += ["", "Volumes in ml on the binary masks; EF err = our EF - expert EF, percentage points.", "",
          "## Mesh smoothing sweep (LV, ED)", "",
          "Volumes above come from the binary mask; the mesh is only drawn. How much does smoothing change it?", "",
          "| Gaussian sigma, voxels | Taubin iterations | mesh volume, ml | mask volume, ml | diff | shift vs unsmoothed mesh, mean / max, mm |",
          "|---|---|---|---|---|---|"]
    for s in res["mesh_sweep_LV_ED"]:
        star = " (viewer)" if s["used_in_viewer"] else ""
        L.append(f"| {s['gaussian_sigma_vox']}{star} | {s['taubin_iterations']} | {s['mesh_ml']:.1f} | "
                 f"{s['mask_ml']:.1f} | {s['diff_%']:+.1f}% | {s['surface_shift_mean_mm']:.2f} / "
                 f"{s['surface_shift_max_mm']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
