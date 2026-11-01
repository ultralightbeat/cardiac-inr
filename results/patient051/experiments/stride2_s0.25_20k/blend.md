# Blend check: results/e2_long

training frames [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29], hidden frames [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |
|---|---|---|---|---|
| interpolation | 37.16 | 33.14 | 0.970 | - |
| siren_scaled_0.25 | 34.49 | 31.35 | 0.938 | 0/14 |
| blend_siren_scaled_0.25 | 36.57 | 32.92 | 0.963 | 2/14 |

siren_scaled_0.25: PSNR on its own training frames 35.48 dB
