# Blend check: results/e2_stride4

training frames [0, 4, 8, 12, 16, 20, 24, 28, 29], hidden frames [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27]

| method | PSNR (crop) | PSNR (heart) | SSIM | frames won vs interpolation (heart) |
|---|---|---|---|---|
| interpolation | 33.74 | 29.23 | 0.941 | - |
| siren_scaled_0.25 | 32.05 | 28.20 | 0.903 | 3/21 |
| blend_siren_scaled_0.25 | 33.70 | 29.50 | 0.936 | 15/21 |

siren_scaled_0.25: PSNR on its own training frames 36.02 dB
