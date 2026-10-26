# patient051: all experiments

PSNR in dB on data hidden from the network; 'heart' = inside the dilated expert labels. Run folders: results/patient051/experiments/. Final 3D/4D heart: final/heart4d/.

## E1 hidden slices and E2 every second frame hidden (4000 steps)

`experiments/main`: E1 + E2 (every 2nd frame hidden), raw / scaled 0.25 / periodic, 4000 steps


Crop (48, 152, 85, 189), commit None

### E1

| method | PSNR (crop) | PSNR (heart) | SSIM | params | time, s |
|---|---|---|---|---|---|
| interpolation | 19.55 | 20.01 | 0.354 | - | - |
| siren | 19.72 | 19.95 | 0.340 | 198,657 | 228 |
| relu | 19.67 | 18.83 | 0.373 | 198,657 | 201 |

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | params | time, s |
|---|---|---|---|---|---|
| interpolation | 37.16 | 33.14 | 0.970 | - | - |
| siren_raw | 23.52 | 22.74 | 0.543 | 198,913 | 305 |
| siren_scaled_0.25 | 31.74 | 29.30 | 0.888 | 198,913 | 302 |
| siren_periodic_0.25 | 30.61 | 28.85 | 0.858 | 199,169 | 305 |

## E2 every second frame hidden, 20000 steps

`experiments/stride2_s0.25_20k`: E2 every 2nd frame hidden, scaled 0.25, 20000 steps


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 37.16 | 33.14 | 0.970 | - | - | - |
| siren_scaled_0.25 | 34.49 | 31.35 | 0.938 | 35.48 | 198,913 | 1480 |

## Blend on every second frame hidden

`experiments/stride2_s0.25_20k`: E2 every 2nd frame hidden, scaled 0.25, 20000 steps


training frames [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29], hidden frames [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |
|---|---|---|---|---|
| interpolation | 37.16 | 33.14 | 0.970 | - |
| siren_scaled_0.25 | 34.49 | 31.35 | 0.938 | 0/14 |
| blend_siren_scaled_0.25 | 36.57 | 32.92 | 0.963 | 2/14 |

siren_scaled_0.25: PSNR on its own training frames 35.48 dB

## Every 4th frame kept: time scale 0.25

`experiments/stride4_s0.25`: every 4th frame kept, scaled 0.25


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - | - | - |
| siren_scaled_0.25 | 32.05 | 28.20 | 0.903 | 36.00 | 198,913 | 1398 |

## Every 4th frame kept: time scale 0.25, 512 neurons

`experiments/stride4_s0.25_w512`: same, 512 neurons


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - | - | - |
| siren_scaled_0.25 | 32.93 | 28.92 | 0.919 | 41.28 | 791,041 | 4723 |

## Every 4th frame kept: time scale 0.1 (reference)

`experiments/stride4`: every 4th frame kept, scaled 0.1 (the reference run)


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - | - | - |
| siren_scaled_0.1 | 32.83 | 28.87 | 0.920 | 36.49 | 198,913 | 1655 |

## Every 4th frame kept: time scale 0.05

`experiments/stride4_s0.05`: every 4th frame kept, scaled 0.05


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - | - | - |
| siren_scaled_0.05 | 32.83 | 28.80 | 0.920 | 36.45 | 198,913 | 1469 |

## ReLU control (same settings as the reference)

`experiments/stride4_relu`: ReLU control, same settings as stride4


Crop (48, 152, 85, 189), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - | - | - |
| relu_scaled_0.1 | 24.42 | 22.28 | 0.562 | 24.51 | 198,913 | 1354 |

## Blend 0.5 SIREN + 0.5 interpolation (reference)

`experiments/stride4`: every 4th frame kept, scaled 0.1 (the reference run)


training frames [0, 4, 8, 12, 16, 20, 24, 28, 29], hidden frames [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27]

| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |
|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - |
| siren_scaled_0.1 | 32.83 | 28.87 | 0.920 | 3/21 |
| blend_siren_scaled_0.1 | 33.92 | 29.65 | 0.940 | 18/21 |

siren_scaled_0.1: PSNR on its own training frames 36.49 dB

## Whole scan without crop (86x compression)

`experiments/stride4_fullimage`: no crop: whole scan, 86x compression


Crop (0, 224, 0, 256), commit None

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | train PSNR | params | time, s |
|---|---|---|---|---|---|---|
| interpolation | 38.59 | 29.23 | 0.979 | - | - | - |
| siren_scaled_0.1 | 34.04 | 27.49 | 0.937 | 34.45 | 198,913 | 1381 |

## Final field on all data: volumes and ejection fraction


source: field; grid 104x104x61 (voxel 1.64x1.64x1.64 mm), 30 time points; blood threshold -0.247

| method | LV EDV | LV ESV | LV EF | RV EDV | RV ESV | RV EF |
|---|---|---|---|---|---|---|
| Expert labels (ED, ES) | 212 | 120 | 44% | 142 | 67 | 53% |
| Threshold, scanner slices | 176 | 93 | 47% | 129 | 61 | 53% |
| Threshold, neural field | 179 | 92 | 49% | 125 | 62 | 51% |

Volumes in ml. EF = (EDV - ESV) / EDV. Expert labels include the papillary muscles in the cavity, the intensity threshold does not, so threshold volumes are expected to be somewhat lower.

