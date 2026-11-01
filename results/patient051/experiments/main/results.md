# Results: patient051

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
