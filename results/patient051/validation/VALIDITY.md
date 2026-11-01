# Validity checks: patient051

One patient (ACDC 051), CPU. Hidden test frames: every frame except 0, 4, ..., 28, 29 (21 frames). Validation frames [12, 20] (taken out of the training frames, used only to choose settings). PSNR in dB, data range 2, intensities min-max normalised over the scan (identical to the training-frames-only min/max: [0.0, 255.0]), no clipping. SSIM = mean 2D SSIM, skimage defaults. Temporal error = MSE of frame-to-frame differences in the heart over the whole cycle (lower is better).

## A. Settings chosen on validation frames only

| model | setting | val PSNR heart | val PSNR crop |
|---|---|---|---|
| linear interpolation | - | 24.53 | 29.52 |
| SIREN | time scale 0.05 | 23.65 | 28.45 |
| SIREN | time scale 0.1 **chosen** | 24.03 | 28.75 |
| SIREN | time scale 0.25 | 20.74 | 22.59 |
| ReLU + Fourier (64 freq.) | sigma 3 **chosen** | 22.33 | 24.17 |
| ReLU + Fourier (64 freq.) | sigma 10 | 15.26 | 15.35 |
| ReLU + Fourier (64 freq.) | sigma 30 | 15.70 | 16.11 |

## B. Test (hidden frames), chosen settings, same split / masks / metrics for all

| method | PSNR heart | PSNR crop | SSIM 2D | temporal error x1e3 | train PSNR (exact, crop) | params | time, min |
|---|---|---|---|---|---|---|---|
| linear interpolation | 29.23 | 33.74 | 0.941 | 3.41 | - | stores 9 frames | - |
| SIREN, seed 0 | 28.87 | 32.83 | 0.920 | 3.66 | 36.49 | 198,913 | 23 |
| SIREN, seed 1 | 28.95 | 32.89 | 0.921 | 3.61 | 36.50 | 198,913 | 25 |
| SIREN, seed 2 | 28.92 | 32.89 | 0.921 | 3.65 | 36.59 | 198,913 | 24 |
| ReLU (plain) | 22.28 | 24.42 | 0.562 | 5.02 | 24.49 | 198,913 | 20 |
| ReLU + Fourier features (sigma 3) | 26.63 | 28.63 | 0.792 | 4.52 | 30.15 | 230,657 | 23 |

## Seeds (SIREN, 3 runs that differ only in the random initialisation and batches)

| metric | mean | std | 95% CI (t, n=3) | interpolation |
|---|---|---|---|---|
| PSNR heart | 28.913 | 0.039 | 28.817 ... 29.009 | 29.231 |
| PSNR crop | 32.872 | 0.035 | 32.784 ... 32.960 | 33.742 |
| SSIM 2D | 0.921 | 0.000 | 0.919 ... 0.922 | 0.941 |
| temporal error x1e3 | 3.640 | 0.028 | 3.570 ... 3.710 | 3.409 |

Seed spread is variation of the training only (one patient): it is not a confidence interval over patients. Interpolation stores the 9 known frames (all their voxels); the networks store their parameters.


## Geometry checks: patient051 (field)

Label-assisted 3D pipeline: network trained on intensities only; threshold, chamber ROI and wall region from the expert labels; expert labels are also the reference. ED / ES frames fixed from Info.cfg (ED = frame 1, ES = frame 11). Overlap measured on the 10 scanner slices (voxel 1.64 x 1.64 x 10 mm). Expert labels include the papillary muscles in the cavity, the threshold does not.

## Overlap with the expert labels

| segmentation | chamber | frame | Dice | Jaccard | HD, mm | HD95, mm | ASSD, mm |
|---|---|---|---|---|---|---|---|
| threshold on scanner slices | LV | ED | 0.891 | 0.803 | 19.2 | 5.9 | 1.28 |
| threshold on scanner slices | LV | ES | 0.761 | 0.614 | 15.5 | 10.0 | 3.06 |
| threshold on scanner slices | RV | ED | 0.880 | 0.786 | 14.5 | 4.9 | 0.99 |
| threshold on scanner slices | RV | ES | 0.671 | 0.505 | 21.2 | 11.0 | 4.45 |
| threshold on neural field (baseline) | LV | ED | 0.897 | 0.813 | 19.2 | 5.9 | 1.21 |
| threshold on neural field (baseline) | LV | ES | 0.769 | 0.624 | 19.5 | 10.1 | 3.19 |
| threshold on neural field (baseline) | RV | ED | 0.880 | 0.786 | 13.8 | 4.9 | 0.98 |
| threshold on neural field (baseline) | RV | ES | 0.664 | 0.497 | 21.2 | 11.3 | 4.68 |

## Volumes and their sensitivity to our choices

| setting | LV EDV | LV ESV | LV EF | LV EF err | RV EDV | RV ESV | RV EF | RV EF err | LV Dice ED |
|---|---|---|---|---|---|---|---|---|---|
| expert labels | 212 | 120 | 43.6% | - | 142 | 67 | 52.9% | - | 1 |
| threshold on scanner slices | 172 | 98 | 43.2% | -0.4 | 113 | 70 | 37.7% | -15.2 | 0.891 |
| baseline: median-midpoint threshold, shape-interpolated ROI, 61 slices x 1.64 mm | 173 | 99 | 42.7% | -0.8 | 112 | 72 | 35.7% | -17.2 | 0.897 |
| threshold: mean-midpoint | 178 | 103 | 41.8% | -1.8 | 118 | 76 | 36.1% | -16.9 | 0.906 |
| threshold: median-midpoint - 0.05 | 184 | 109 | 40.6% | -3.0 | 125 | 80 | 36.2% | -16.7 | 0.915 |
| threshold: median-midpoint + 0.05 | 162 | 90 | 44.6% | +1.0 | 99 | 65 | 34.4% | -18.5 | 0.873 |
| ROI: union of neighbouring slices | 173 | 99 | 42.7% | -0.8 | 112 | 72 | 35.2% | -17.7 | 0.897 |
| grid: 31 slices x 3.3 mm | 172 | 99 | 42.7% | -0.8 | 112 | 72 | 35.9% | -17.0 | 0.893 |
| grid: 122 slices x 0.8 mm | 172 | 99 | 42.7% | -0.9 | 112 | 72 | 35.7% | -17.2 | 0.896 |

Volumes in ml on the binary masks; EF err = our EF - expert EF, percentage points.

## Mesh smoothing sweep (LV, ED)

Volumes above come from the binary mask; the mesh is only drawn. How much does smoothing change it?

| Gaussian sigma, voxels | Taubin iterations | mesh volume, ml | mask volume, ml | diff | shift vs unsmoothed mesh, mean / max, mm |
|---|---|---|---|---|---|
| 0.0 | 0 | 172.4 | 172.6 | -0.1% | 0.00 / 0.00 |
| 0.0 | 10 | 172.5 | 172.6 | -0.1% | 0.37 / 0.88 |
| 0.0 | 30 | 172.7 | 172.6 | +0.1% | 0.46 / 1.13 |
| 0.6 | 0 | 172.0 | 172.6 | -0.4% | 0.20 / 1.14 |
| 0.6 | 10 | 172.1 | 172.6 | -0.3% | 0.41 / 1.28 |
| 0.6 | 30 | 172.3 | 172.6 | -0.1% | 0.47 / 1.48 |
| 1.2 | 0 | 170.4 | 172.6 | -1.3% | 0.50 / 1.85 |
| 1.2 (viewer) | 10 | 170.5 | 172.6 | -1.2% | 0.54 / 1.86 |
| 1.2 | 30 | 170.8 | 172.6 | -1.0% | 0.57 / 1.92 |
| 2.0 | 0 | 167.4 | 172.6 | -3.0% | 0.72 / 2.96 |
| 2.0 | 10 | 167.5 | 172.6 | -2.9% | 0.75 / 3.18 |
| 2.0 | 30 | 167.9 | 172.6 | -2.7% | 0.76 / 3.15 |
