# Decision log

Format: one entry per decision, newest at the bottom. Keep it short; it feeds the report.

```
## D-XXX: <title>
Date: YYYY-MM-DD | Author: <name>
Context: what problem, what constraints
Options: A ..., B ...
Decision: ...
Why: ...
```

## D-001: Scope is one patient, overfitted on purpose
Date: (fill in) | Author: (fill in)
Context: course project with limited time and compute; no ready-made project exists for SIREN on 4D cardiac MRI.
Options: (A) generalising model across patients; (B) per-patient implicit representation.
Decision: B, with levels L1 (static 3D), L2 (4D), L3 (stretch), and a fallback rule at the end of W3.
Why: B is the standard setting for implicit representations, fits the timeline, and still demonstrates the full method.

## D-002: Hide every second slice / frame and compare with linear interpolation
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: training PSNR alone says nothing about "continuity"; we need values the network never saw.
Options: (A) random voxel holdout; (B) whole hidden slices (E1) and whole hidden frames (E2).
Decision: B. Train on even indices plus the last one, test on the odd ones in between (`split_alternate`).
Why: this is exactly the use case (filling gaps between slices/frames), and every hidden item has known neighbours
on both sides, so linear interpolation is a fair, strong baseline.

## D-003: Scale the time coordinate (t * 0.25) by default
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: with raw t in [-1, 1] and omega_0 = 30 the field changes too fast in time and fails between frames.
Options: raw t; scaled t with s in {1, 0.5, 0.25, 0.1, 0.05, 0.01}; periodic (sin, cos) of the cardiac phase.
Decision: s = 0.25 as default; raw and periodic kept as compared variants in E2.
Why: prototype sweep on patient051 (hidden frame 6): 25.4 / 29.7 / 31.9 / 32.0 / 31.7 / 30.8 dB. Plateau for
s in [0.05, 0.25]; 0.25 is the largest value on the plateau, so the network still models motion.

## D-004: Crop around the heart and report heart-region PSNR
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: CPU only; the heart is ~5% of the 224x256 image, so whole-image PSNR is dominated by static background.
Decision: train and evaluate inside a square in-plane crop around the union of ED/ES labels (+12 px); additionally
report PSNR inside the dilated (5 px) label mask. Normalisation stays that of the whole volume (same as CoordinateDataset).
Why: ~6x fewer points per step, and the metric measures the moving anatomy we care about.

## D-005: `reconstruct_volume(model, grid)` takes a coordinate grid
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: we need to query the field at positions that are not on the voxel grid (37 slices, frame 6.5).
Decision: grids are built by `Normaliser.grid(rows, cols, slices, frames)` (fractional indices allowed);
`reconstruct_volume` returns an array with the grid's leading shape.

## D-006: Time scale 0.1 (updated from D-003)
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: with every 4th frame kept, SIREN fitted its training frames at 41 dB but hidden frames at 29 dB: it
over-fits in time. A smaller time scale makes the field smoother along t.
Result (heart-region PSNR, 256 units, 20k steps): s = 0.25 → 28.20, 0.1 → 28.87, 0.05 → 28.80; interpolation 29.23.
Decision: s = 0.1 everywhere. It matches a 4x larger network (512 units, s = 0.25: 28.92).

## D-007: ReLU control in E2 with identical settings
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: "would a plain MLP do as well with enough steps?"
Result: same size (198,913 parameters), same steps, same encoding: ReLU 22.28 dB vs SIREN 28.87 dB on hidden
frames; ReLU reaches only 24.5 dB even on its training frames (spectral bias).

## D-008: Blend 0.5 SIREN + 0.5 interpolation, weight fixed in advance
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: the two methods fail in different places (interpolation: fast non-linear motion; field: fine texture).
Decision: report the blend with a fixed weight 0.5, never tuned on hidden frames.
Result: every 4th frame kept: 29.65 dB vs 29.23 (interpolation), better on 18 of 21 frames. Every 2nd frame
hidden: 32.92 vs 33.14 (no gain when frames are dense). The blend needs the stored frames, so it is not compact.

## D-009: Compression is reported for the region the field represents
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: the midterm target "≈86x" divided the whole scan (17.2 M voxels) by 199k parameters, but the
experiments fit a crop around the heart (3.2 M voxels).
Decision: report both, as a trade-off. Heart crop: 16x fewer numbers (8x in bytes), 28.9 dB in the heart.
Whole scan (--no-crop): 86x (43x in bytes), 27.5 dB in the heart. Interpolation: 29.2 dB but stores all frames.

## D-010: 3D/4D heart by an iso-surface of the field
Date: 2026-10-26 | Author: Danil Bezborodnikov
Context: the field gives intensity, a 3D model needs a surface.
Decision: query the final field (all data) on a 1.64 mm isotropic grid; blood = intensity above the midpoint of
median blood and median myocardium (from the ED labels); regions of LV / RV / LV wall from the expert labels,
interpolated between slices by signed distance; largest 3D component; marching cubes + Taubin smoothing.
Labels only choose the region and the threshold; the shape and motion come from the field.
Validation: LV EF 49 % (labels 44 %, same method on scanner slices 47 %); threshold excludes papillary muscles,
so volumes are lower than the labels.
