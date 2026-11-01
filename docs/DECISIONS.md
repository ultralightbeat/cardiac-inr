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
Date: 2026-10 | Author: team
Context: course project with limited time and compute; no ready-made project exists for SIREN on 4D cardiac MRI.
Options: (A) generalising model across patients; (B) per-patient implicit representation.
Decision: B, with levels L1 (static 3D), L2 (4D), L3 (stretch), and a fallback rule at the end of W3.
Why: B is the standard setting for implicit representations, fits the timeline, and still demonstrates the full method.

## D-002: Hide every second slice / frame and compare with linear interpolation
Date: 2026-10-26 | Author: team
Context: training PSNR alone says nothing about "continuity"; we need values the network never saw.
Options: (A) random voxel holdout; (B) whole hidden slices (E1) and whole hidden frames (E2).
Decision: B. Train on even indices plus the last one, test on the odd ones in between (`split_alternate`).
Why: this is exactly the use case (filling gaps between slices/frames), and every hidden item has known neighbours
on both sides, so linear interpolation is a fair, strong baseline.

## D-003: Scale the time coordinate (t * 0.25) by default
Date: 2026-10-26 | Author: team
Context: with raw t in [-1, 1] and omega_0 = 30 the field changes too fast in time and fails between frames.
Options: raw t; scaled t with s in {1, 0.5, 0.25, 0.1, 0.05, 0.01}; periodic (sin, cos) of the cardiac phase.
Decision: s = 0.25 as default; raw and periodic kept as compared variants in E2.
Why: prototype sweep on patient051 (hidden frame 6): 25.4 / 29.7 / 31.9 / 32.0 / 31.7 / 30.8 dB. Plateau for
s in [0.05, 0.25]; 0.25 is the largest value on the plateau, so the network still models motion.

## D-004: Crop around the heart and report heart-region PSNR
Date: 2026-10-26 | Author: team
Context: CPU only; the heart is ~5% of the 224x256 image, so whole-image PSNR is dominated by static background.
Decision: train and evaluate inside a square in-plane crop around the union of ED/ES labels (+12 px); additionally
report PSNR inside the dilated (5 px) label mask. Normalisation stays that of the whole volume (same as CoordinateDataset).
Why: ~6x fewer points per step, and the metric measures the moving anatomy we care about.

## D-005: `reconstruct_volume(model, grid)` takes a coordinate grid
Date: 2026-10-26 | Author: team
Context: we need to query the field at positions that are not on the voxel grid (37 slices, frame 6.5).
Decision: grids are built by `Normaliser.grid(rows, cols, slices, frames)` (fractional indices allowed);
`reconstruct_volume` returns an array with the grid's leading shape.

## D-006: Time scale 0.1 (updated from D-003)
Date: 2026-10-26 | Author: team
Context: with every 4th frame kept, SIREN fitted its training frames at 41 dB but hidden frames at 29 dB: it
over-fits in time. A smaller time scale makes the field smoother along t.
Result (heart-region PSNR, 256 units, 20k steps): s = 0.25 → 28.20, 0.1 → 28.87, 0.05 → 28.80; interpolation 29.23.
Decision: s = 0.1 everywhere. It matches a 4x larger network (512 units, s = 0.25: 28.92).

## D-007: ReLU control in E2 with identical settings
Date: 2026-10-26 | Author: team
Context: "would a plain MLP do as well with enough steps?"
Result: same size (198,913 parameters), same steps, same encoding: ReLU 22.28 dB vs SIREN 28.87 dB on hidden
frames; ReLU reaches only 24.5 dB even on its training frames (spectral bias).

## D-008: Blend 0.5 SIREN + 0.5 interpolation, weight fixed in advance
Date: 2026-10-26 | Author: team
Context: the two methods fail in different places (interpolation: fast non-linear motion; field: fine texture).
Decision: report the blend with a fixed weight 0.5, never tuned on hidden frames.
Result: every 4th frame kept: 29.65 dB vs 29.23 (interpolation), better on 18 of 21 frames. Every 2nd frame
hidden: 32.92 vs 33.14 (no gain when frames are dense). The blend needs the stored frames, so it is not compact.

## D-009: Compression is reported for the region the field represents
Date: 2026-10-26 | Author: team
Context: the midterm target "≈86x" divided the whole scan (17.2 M voxels) by 199k parameters, but the
experiments fit a crop around the heart (3.2 M voxels).
Decision: report both, as a trade-off. Heart crop: 16x fewer numbers (8x in bytes), 28.9 dB in the heart.
Whole scan (--no-crop): 86x (43x in bytes), 27.5 dB in the heart. Interpolation: 29.2 dB but stores all frames.

## D-010: 3D/4D heart by an iso-surface of the field
Date: 2026-10-26 | Author: team
Context: the field gives intensity, a 3D model needs a surface.
Decision: query the final field (all data) on a 1.64 mm isotropic grid; blood = intensity above the midpoint of
median blood and median myocardium (from the ED labels); regions of LV / RV / LV wall from the expert labels,
interpolated between slices by signed distance; largest 3D component; marching cubes + Taubin smoothing.
Labels only choose the region and the threshold; the shape and motion come from the field.
Validation: LV EF 49 % (labels 44 %, same method on scanner slices 47 %); threshold excludes papillary muscles,
so volumes are lower than the labels.

## D-011: Scale the slice coordinate z as well (E1)
Date: 2026-10-26 | Author: team
Context: in E1 the hidden slices lie 20 mm from the known ones. With raw z and omega_0 = 30 the SIREN memorised the
known slices (41 dB) but filled the gap with high-frequency noise: same failure as raw t in E2.
Options: z scale 1 (raw), 0.5, 0.25, 0.1 (--z-scale; multiplies z inside the model, like the time scale).
Result (heart-region / crop PSNR on hidden slices): patient051: raw 19.95 / 19.72, 0.5 19.87 / 19.85,
0.25 20.09 / 19.79; interpolation 20.01 / 19.55. patient015: 0.5 20.42 / 20.84, 0.25 20.13 / 20.45,
0.1 20.03 / 20.25; interpolation 20.68 / 20.06. The noise disappears and SIREN is level with interpolation
(slightly better on patient051). A 20 mm gap is a limit of the data: the heart changes shape over that distance.

## D-012: Separate validation frames and a stronger baseline
Date: 2026-10-27 | Author: team
Context: the time scale had been explored on the hidden test frames; plain ReLU is a weak control.
Decision: frames 12 and 20 are removed from training and used only to choose settings (scripts/run_validity.py);
add ReLU + Fourier features (64 frequencies, sigma chosen on validation); repeat SIREN with 3 seeds; store exact
train PSNR, temporal error and normalisation metadata in every run.
Result: validation picks the same time scale 0.1 (24.0 dB vs 23.7 for 0.05 and 20.7 for 0.25). SIREN 3 seeds:
28.91 ± 0.04 dB; Fourier-ReLU 26.63 dB; ReLU 22.28; interpolation 29.23. Normalisation: training-frame min/max
identical to the whole scan (0, 255), so no leakage.

## D-013: 14 held-out patients with settings fixed in advance
Date: 2026-10-28 | Author: team
Context: one development patient does not show generalisation.
Decision: run the fixed protocol (SIREN t × 0.1, 20,000 steps, blend 0.5) on 14 ACDC patients from three groups
(016-020 DCM, 022/025/030/035/040 HCM, 041/045/048/050 MINF) on a GPU (scripts/run_patients.py); patients are the
unit of analysis (95 % t-interval, paired Wilcoxon test).
Result: blend better than interpolation on 14/14 (+0.37 dB, p = 0.0001); SIREN better than ReLU on 14/14 (+7.4 dB);
SIREN alone behind interpolation on 13/14 (−0.57 dB).

## D-014: Ejection fraction at the ED / ES frames of Info.cfg
Date: 2026-10-27 | Author: team
Context: EF was taken from the maximum and minimum of the volume curve.
Decision: use the ED / ES frames given with the scan, the clinical standard; report Dice, HD95, ASSD against the
expert labels and the same threshold on the scanner data as reference (scripts/validate_geometry.py).
Result: patient 051 LV EF 42.7 % (labels 43.6 %, previously 49 % with curve extremes); LV Dice 0.897 at ED
(scanner 0.891). 14 patients: LV Dice 0.87, mean absolute EF error 4.6 points. The right ventricle at ES is not
segmented reliably by a threshold (Dice 0.66 on patient 051): reported as a limitation. Sensitivity: threshold
±0.05 changes LV EF 40.6-44.6 %; grid density and display smoothing do not change the volumes (< 1.2 %).

## D-015: Beyond linear interpolation (exploratory, not in the main results)
Date: 2026-10-28 | Author: team
Context: can a model beat interpolation on its own?
Result (patient 051, every 4th frame kept): cubic spline 28.54, PCHIP 29.17, Akima 29.05 dB, all below linear
29.23; TV-L1 optical-flow interpolation 29.49 (29.78 blended with linear). A motion-based field (SIREN template
+ displacement + residual, also guided by optical flow) reached 22.9 dB on validation vs SIREN 24.0 and
interpolation 24.5, so it was dropped. Direction for future work: motion-aware fields and priors learned
across patients.

