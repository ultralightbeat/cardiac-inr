# Blend check: results/e2_stride4_s0.1

training frames [0, 4, 8, 12, 16, 20, 24, 28, 29], hidden frames [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27]

| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |
|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - |
| siren_scaled_0.1 | 32.83 | 28.87 | 0.920 | 3/21 |
| blend_siren_scaled_0.1 | 33.92 | 29.65 | 0.940 | 18/21 |

siren_scaled_0.1: PSNR on its own training frames 36.49 dB
