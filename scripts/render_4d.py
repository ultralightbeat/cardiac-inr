"""The 3D / 4D result of the project: a beating 3D heart produced by the neural field.

What it does
  1. Gets a neural field of the whole cardiac cycle of one patient:
       --train            fit a new SIREN on ALL slices and ALL frames (the final model, saved to
                          models/<patient>/field.pt), or reuse that file / --checkpoint PATH.
  2. Queries the field on a grid that the scanner never measured: isotropic voxels
     (slice spacing = in-plane pixel size, ~6x denser than the 10 mm slices) and
     any number of time points.
  3. Extracts the blood pool of the left and right ventricle at every time point,
     builds 3D surfaces (marching cubes) and measures chamber volumes.
  4. Validates the volumes against the expert labels of ACDC (end-diastolic and
     end-systolic volume, ejection fraction).

Outputs (in --out):
  heart_4d.html      interactive beating heart (open in any browser, works offline)
  heart_3d.gif       the same as an animation for slides
  planes_4d.gif      short-axis + cut across the slices: scanner vs neural field, over time
  volume_curve.png   LV / RV volume over the heartbeat (field vs scanner vs expert labels)
  {MYO,LV,RV}_{ED,ES}.obj   meshes (Blender, MeshLab, Windows 3D Viewer); y axis = base-to-apex
  summary.md, volumes.json

Examples
  python scripts/render_4d.py --train --time-scale 0.1 --steps 20000   # final model -> models/<patient>/, ~30 min
  python scripts/render_4d.py                                          # re-render with models/<patient>/field.pt
  python scripts/render_4d.py --source scanner                       # no network: interpolated scanner stack
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from inr4d.data import patients as P
from inr4d.data import sampling as S
from inr4d.data.acdc import load_acdc_patient
from inr4d.evaluation.baselines import linear_along_axis
from inr4d.viz import heart3d as H3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default=None,
                   help="patient folder (ACDC layout, needs the *_gt labels) or 4D NIfTI; default: first under data/")
    p.add_argument("--out", default=None, help="default results/<patient>/final/heart4d")
    p.add_argument("--source", choices=["field", "scanner"], default="field")
    p.add_argument("--train", action="store_true", help="train a new field on all slices and frames")
    p.add_argument("--model-dir", default=None, help="where --train saves the field (default models/<patient>)")
    p.add_argument("--checkpoint", default=None,
                   help="trained field; default models/<patient>/field.pt if it exists")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--time-scale", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pad", type=int, default=12)
    p.add_argument("--frames", type=int, default=30, help="time points to render over the cycle")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", help="cpu / cuda / xpu / auto (GPU if there is one)")
    return p.parse_args()


# ----------------------------------------------------------------------------- volume providers

class FieldSource:
    def __init__(self, field, norm, rows, cols):
        self.field, self.norm, self.rows, self.cols = field, norm, rows, cols

    def __call__(self, z, t):
        from inr4d.evaluation.metrics import reconstruct_volume

        return reconstruct_volume(self.field, self.norm.grid(self.rows, self.cols, z, [t]))[..., 0]


class ScannerSource:
    """Linear interpolation of the scanner stack in z and t (no network)."""

    def __init__(self, block):
        self.block = block                                   # [h, w, D, T]
        self.D, self.T = block.shape[2], block.shape[3]

    def __call__(self, z, t):
        b = linear_along_axis(self.block, np.arange(self.T), [t], axis=3)[..., 0]
        return linear_along_axis(b, np.arange(self.D), z, axis=2)


def get_field(a, vol, norm, crop, out):
    import torch

    from inr4d.models import build_field, count_parameters
    from inr4d.training.trainer import TensorSampler, TrainConfig, resolve_device, train

    if a.threads:
        torch.set_num_threads(a.threads)
    T = vol.data.shape[3]
    meta_path = None
    if a.checkpoint:
        meta_path = os.path.join(os.path.dirname(a.checkpoint), "results.json")
        if os.path.exists(meta_path):                       # checkpoint from run_experiments.py
            with open(meta_path, encoding="utf-8") as f:
                args = json.load(f)["args"]
            a.width, a.layers, a.omega, a.time_scale = args["width"], args["layers"], args["omega"], args["time_scale"]
        meta = os.path.join(os.path.dirname(a.checkpoint), "field.json")
        if os.path.exists(meta):                            # checkpoint from this script
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            a.width, a.layers, a.omega, a.time_scale = m["width"], m["layers"], m["omega"], m["time_scale"]
    field = build_field("siren", "scaled", a.time_scale, T, a.width, a.layers, a.omega, a.seed)
    if a.checkpoint:
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        field.load_state_dict(state["state_dict"])
        field.to(resolve_device(getattr(a, "device", "auto")))
        print(f"loaded {a.checkpoint} ({count_parameters(field):,} parameters, time scale {a.time_scale})")
        return field, None
    data = S.select(vol, norm, crop)
    print(f"training the final field on all {T} frames and all slices: {len(data.coords):,} points, "
          f"{count_parameters(field):,} parameters, {a.steps} steps")
    cfg = TrainConfig(steps=a.steps, batch_size=a.batch, lr=a.lr, lr_final=a.lr / 10,
                      log_every=max(a.steps // 20, 1), seed=a.seed, threads=a.threads,
                      device=getattr(a, "device", "auto"))
    path = os.path.join(out, "field.pt")
    r = train(field, TensorSampler(data.coords, data.values, a.seed), cfg, checkpoint_path=path)
    with open(os.path.join(out, "field.json"), "w", encoding="utf-8") as f:
        json.dump({"width": a.width, "layers": a.layers, "omega": a.omega, "time_scale": a.time_scale,
                   "steps": a.steps, "params": count_parameters(field), "seconds": r.seconds,
                   "train_psnr": r.history["train_psnr"][-1]}, f, indent=2)
    return field, r


# ----------------------------------------------------------------------------- figures

def planes_gif(path, frames_field, scanner_block, dense_z, dense_t, row, sa_z, labels_text):
    """frames_field: list of [h, w, nz]; scanner_block [h, w, D, T]."""
    from PIL import Image, ImageDraw

    D = scanner_block.shape[2]
    zi = np.clip(np.round(dense_z).astype(int), 0, D - 1)

    def u8(x):
        return np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)

    up, gap, head = 3, 6, 22
    imgs = []
    for k, vf in enumerate(frames_field):
        tk = int(np.clip(round(dense_t[k]), 0, scanner_block.shape[3] - 1))
        sa = Image.fromarray(u8(np.rot90(vf[:, :, sa_z]))).convert("RGB")
        la_scan = Image.fromarray(u8(scanner_block[row, :, :, tk][:, zi].T)).convert("RGB")  # nearest slice
        la_field = Image.fromarray(u8(vf[row].T)).convert("RGB")
        sa, la_scan, la_field = (im.resize((im.width * up, im.height * up), Image.BICUBIC)
                                 for im in (sa, la_scan, la_field))
        # mark the cut on the short-axis image (row r of the data = column r after rot90)
        dr = ImageDraw.Draw(sa)
        x = row * up + up // 2
        for y in range(0, sa.height, 12):
            dr.line([(x, y), (x, y + 5)], fill=(240, 200, 60), width=2)
        right_h = 2 * (la_scan.height + head) + gap
        H = max(sa.height + head, right_h)
        W = sa.width + gap + la_scan.width
        frame = Image.new("RGB", (W, H), (20, 20, 22))
        frame.paste(sa, (0, head))
        x0 = sa.width + gap
        frame.paste(la_scan, (x0, head))
        frame.paste(la_field, (x0, 2 * head + la_scan.height + gap))
        dr = ImageDraw.Draw(frame)
        dr.text((4, 5), labels_text[0] + f"   {100 * dense_t[k] / (scanner_block.shape[3] - 1):.0f}% of cycle",
                fill=(235, 235, 235))
        dr.text((x0 + 4, 5), labels_text[1], fill=(235, 235, 235))
        dr.text((x0 + 4, head + la_scan.height + gap + 5), labels_text[2], fill=(235, 235, 235))
        imgs.append(frame)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0)


def heart_gif(path, meshes, size=520, caps=None, cap_geom=None):
    """Rendered beating heart: muscle cut open at the front, blood of LV and RV inside.
    The camera swings slowly left-right so the 3D shape is readable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    def tm(v):                                   # (row, col, slice) -> x, y (depth), z up = -slice
        return np.stack([v[:, 0], v[:, 1], -v[:, 2]], axis=1) if len(v) else v

    allv = np.concatenate([tm(v) for fr in meshes for v, _ in fr.values() if len(v)])
    lo, hi = allv.min(0), allv.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 0.85
    plane = cap_geom["col_mm"] if cap_geom else c[1]
    light = matplotlib.colors.LightSource(300, 35)
    imgs = []
    n = len(meshes)
    for k, fr in enumerate(meshes):
        fig = plt.figure(figsize=(10.4, 5.2), dpi=size / 5.2)
        for panel, cut in ((1, False), (2, True)):
            ax = fig.add_subplot(1, 2, panel, projection="3d")
            tris, cols = [], []
            for name in H3.STRUCTURES:
                if name not in fr or not len(fr[name][1]):
                    continue
                v, f = fr[name]
                t = tm(v)[f]
                if name in ("MYO", "LV") and cut:
                    t = t[t[:, :, 1].mean(1) > plane]    # remove the front half of the LV (wall + blood)
                if name == "LV" and not cut:
                    continue                              # hidden inside the muscle anyway
                tris.append(t)
                cols += [H3.COLORS[name]] * len(t)
            if cut and caps:                              # the cut face: network image of the wall
                img, alpha = caps[k]
                ii, kk = np.nonzero(alpha > 127)
                sxx, dzz = cap_geom["sx"], cap_geom["dz"]
                x0, x1 = (ii - 0.5) * sxx, (ii + 0.5) * sxx
                z0, z1 = -(kk - 0.5) * dzz, -(kk + 0.5) * dzz
                yy = np.full_like(x0, plane, dtype=float)
                q1 = np.stack([np.stack([x0, yy, z0], 1), np.stack([x1, yy, z0], 1), np.stack([x1, yy, z1], 1)], 1)
                q2 = np.stack([np.stack([x0, yy, z0], 1), np.stack([x1, yy, z1], 1), np.stack([x0, yy, z1], 1)], 1)
                g = img[ii, kk] / 255.0
                gc = [matplotlib.colors.to_hex((v, v * 0.93, v * 0.9)) for v in g]
                tris += [q1, q2]
                cols += gc + gc
            pc = Poly3DCollection(np.concatenate(tris), facecolors=cols, edgecolors=cols, linewidths=0,
                                  antialiased=False, shade=True, lightsource=light)
            ax.add_collection3d(pc)
            for i, setlim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setlim(c[i] - r, c[i] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=12, azim=-90 + 35 * np.sin(2 * np.pi * k / n))
            ax.set_axis_off()
            ax.set_title("cut open: blood inside" if cut else "outside: muscle (LV) + right ventricle",
                         fontsize=10, color="#444", y=0.95)
        fig.subplots_adjust(0, 0, 1, 1, wspace=0)
        fig.text(0.015, 0.03, f"{100 * k / max(n - 1, 1):.0f}% of cycle", fontsize=10, color="#555")
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=90, loop=0)


def volume_plot(path, phase, vols, scan_phase, scan_vols, gt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in vols:
        col = H3.COLORS[name]
        ax.plot(phase, vols[name], color=col, lw=2, label=f"{name}: neural field (dense)")
        ax.plot(scan_phase, scan_vols[name], "o", ms=3.5, color=col, alpha=0.55,
                label=f"{name}: same method on scanner slices")
        for ph, v in gt.get(name, []):
            ax.plot([ph], [v], marker="*", ms=13, color=col, markeredgecolor="black")
    ax.plot([], [], "*", color="gray", markeredgecolor="black", ms=11, label="expert labels (ED, ES)")
    ax.set_xlabel("% of the cardiac cycle")
    ax.set_ylabel("volume, ml")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Chamber volumes over one heartbeat")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    a = parse_args()
    a.patient = P.resolve_patient(a.patient)
    a.out = a.out or P.results_dir(a.patient, "final/heart4d")
    os.makedirs(a.out, exist_ok=True)
    mdir = a.model_dir or P.model_dir(a.patient)
    if a.source == "field" and not a.train and not a.checkpoint:
        if os.path.exists(os.path.join(mdir, "field.pt")):
            a.checkpoint = os.path.join(mdir, "field.pt")
        else:
            raise SystemExit(f"no trained field in {mdir}/: run with --train (or give --checkpoint)")
    t0 = time.time()
    vol = load_acdc_patient(a.patient)
    Hh, Ww, D, T = vol.data.shape
    sx, sy, sz = (float(s) for s in vol.spacing)
    info = S.read_info(a.patient)
    labels = S.load_labels(a.patient)
    if not labels:
        raise SystemExit("this step needs expert labels (*_frameXX_gt.nii next to the 4D image) to tell the left "
                         "and right ventricle apart and to set the blood threshold; the experiments "
                         "(run_experiments.py) work without them")
    norm = S.Normaliser.from_volume(vol)
    heart = S.heart_mask(labels, (Hh, Ww, D))
    crop = S.heart_crop(heart, a.pad)
    r0, r1, c0, c1 = crop
    rows, cols = np.arange(r0, r1), np.arange(c0, c1)
    ed = int(info.get("ED", 1)) - 1
    es = int(info.get("ES", 1)) - 1
    block = S.block(vol, norm, crop)                              # [h, w, D, T] in [-1, 1]
    lab_c = {k: v[r0:r1, c0:c1] for k, v in labels.items()}

    # isotropic grid: slice step = in-plane pixel size
    nz = int(round(D * sz / sx))
    dz = D * sz / nz
    dense_z = (np.arange(nz) + 0.5) * (D / nz) - 0.5
    dense_t = np.linspace(0, T - 1, a.frames)
    phase = 100 * dense_t / (T - 1)
    print(f"grid: {len(rows)}x{len(cols)} px of {sx:.2f} mm, {nz} slices of {dz:.2f} mm "
          f"(scanner: {D} of {sz:g} mm), {a.frames} time points")

    if a.source == "field":
        os.makedirs(mdir, exist_ok=True)
        field, _ = get_field(a, vol, norm, crop, mdir)
        source = FieldSource(field, norm, rows, cols)
    else:
        source = ScannerSource(block)

    thr = H3.blood_threshold(block[..., ed], lab_c[ed])
    band = H3.muscle_band(block[..., ed], lab_c[ed], thr)
    rois = H3.chamber_rois(labels, crop)
    rois_dense = {k: H3.roi_interp(v, dense_z) for k, v in rois.items()}
    wall_dense = H3.roi_interp(H3.wall_roi(labels, crop), dense_z)

    def structures(v, step):
        out, masks = {}, {}
        for k in H3.CHAMBERS:
            masks[k] = H3.segment(v, rois_dense[k], thr)
        masks["MYO"] = H3.lv_solid(v, masks["LV"], wall_dense, band)
        for k in H3.STRUCTURES:
            out[k] = H3.mesh(masks[k], (sx, sy, dz), step=step, smooth=1.2)
        return masks, out

    # reference: the same segmentation on the scanner slices themselves (10 mm voxels)
    scan_vols = {k: [] for k in H3.CHAMBERS}
    for t in range(T):
        for k in H3.CHAMBERS:
            m = H3.segment(block[..., t], rois[k], thr)
            scan_vols[k].append(m.sum() * sx * sy * sz / 1000)
    gt = {}
    for k, code in H3.CHAMBERS.items():
        gt[k] = [(100 * f / (T - 1), float((lab_c[f] == code).sum() * sx * sy * sz / 1000)) for f in (ed, es)
                 if f in lab_c]

    # dense field: masks, volumes, meshes
    vols = {k: [] for k in H3.CHAMBERS}
    lv_lab = lab_c[ed] == 3
    ci = int(round(np.nonzero(lv_lab.any(axis=(0, 2)))[0].mean())) if lv_lab.any() else len(cols) // 2
    cap_geom = {"col_mm": ci * sy, "sx": sx, "dz": dz}

    def u8(x):
        return np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)

    meshes, fields, caps = [], [], []
    for i, t in enumerate(dense_t):
        v = source(dense_z, t)
        fields.append(v)
        masks, fr = structures(v, 2)
        wall = masks["MYO"] & ~masks["LV"]
        caps.append((u8(v[:, ci, :]), (wall[:, ci, :] * 255).astype(np.uint8)))
        for k in H3.CHAMBERS:
            vols[k].append(masks[k].sum() * sx * sy * dz / 1000)
        meshes.append(fr)
        print(f"\rtime point {i + 1}/{len(dense_t)}", end="", flush=True)
    print()

    # meshes at ED / ES (full resolution) for external viewers
    for tag, f in (("ED", ed), ("ES", es)):
        _, ms = structures(source(dense_z, float(f)), 1)
        for k, (verts, faces) in ms.items():
            H3.save_obj(os.path.join(a.out, f"{k}_{tag}.obj"), H3.display_coords(verts), faces)

    # table
    def ef_row(name, d):
        cells = [name]
        for k in H3.CHAMBERS:
            edv, esv, ef = H3.ejection_fraction(d[k])
            cells += [f"{edv:.0f}", f"{esv:.0f}", f"{ef:.0f}%"]
        return cells

    gt_vals = {k: [v for _, v in gt[k]] for k in gt}
    rows_tab = [ef_row("Expert labels (ED, ES)", gt_vals),
                ef_row("Threshold, scanner slices", scan_vols),
                ef_row(f"Threshold, {'neural field' if a.source == 'field' else 'interpolated stack'}", vols)]
    cols_tab = ["LV EDV", "LV ESV", "LV EF", "RV EDV", "RV ESV", "RV EF"]
    md = ["# 4D heart: " + os.path.basename(os.path.normpath(a.patient)), "",
          f"source: {a.source}; grid {len(rows)}x{len(cols)}x{nz} (voxel {sx:.2f}x{sy:.2f}x{dz:.2f} mm), "
          f"{a.frames} time points; blood threshold {thr:.3f}", "",
          "| method | " + " | ".join(cols_tab) + " |", "|---|" + "---|" * len(cols_tab)]
    md += ["| " + " | ".join(r) + " |" for r in rows_tab]
    md += ["", "Volumes in ml. EF = (EDV - ESV) / EDV. Expert labels include the papillary muscles in the "
           "cavity, the intensity threshold does not, so threshold volumes are expected to be somewhat lower."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "volumes.json"), "w", encoding="utf-8") as f:
        json.dump({"phase": phase.tolist(), "field": vols, "scanner": scan_vols, "expert": gt,
                   "threshold": thr, "crop": crop, "dz": dz}, f, indent=2)
    print("\n".join(md))

    volume_plot(os.path.join(a.out, "volume_curve.png"), phase, vols, 100 * np.arange(T) / (T - 1), scan_vols, gt)
    lv = lab_c[ed] == 3
    row = int(np.argmax(lv.sum(axis=(1, 2))))
    zs = np.where(lv.any(axis=(0, 1)))[0]
    sa_z = int(np.argmin(np.abs(dense_z - zs.mean()))) if zs.size else nz // 2
    name = "neural field" if a.source == "field" else "interpolated"
    planes_gif(os.path.join(a.out, "planes_4d.gif"), fields, block, dense_z, dense_t, row, sa_z,
               ["short axis, " + name, f"cut along the dashed line: scanner, {D} slices x {sz:g} mm",
                f"same cut: {name}, {nz} slices x {dz:.1f} mm"])
    H3.write_viewer(os.path.join(a.out, "heart_4d.html"), meshes, vols, phase,
                    "Beating heart from a neural field" if a.source == "field" else "Beating heart (no network)",
                    f"{os.path.basename(os.path.normpath(a.patient))}: left and right ventricle extracted from a "
                    f"SIREN queried on a {sx:.1f} x {sy:.1f} x {dz:.1f} mm grid at {a.frames} time points "
                    f"(the scanner measured 10 mm slices)" if a.source == "field" else
                    "Interpolated scanner stack (no network)",
                    {"cols": cols_tab, "rows": rows_tab,
                     "note": "Volumes in ml, EF = ejection fraction. Threshold volumes exclude papillary muscles."},
                    caps=caps, cap_geom=cap_geom)
    print("rendering heart_3d.gif ...")
    heart_gif(os.path.join(a.out, "heart_3d.gif"), meshes, caps=caps, cap_geom=cap_geom)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
