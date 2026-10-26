# Results: patient051

Crop (48, 152, 85, 189), commit None

### E1

| method | PSNR (crop) | PSNR (heart) | SSIM | params | time, s |
|---|---|---|---|---|---|
| interpolation | 19.55 | 20.01 | 0.354 | - | - |
| siren | 18.62 | 19.47 | 0.332 | 12,801 | 1 |
| relu | 13.19 | 15.70 | 0.171 | 12,801 | 1 |

### E2

| method | PSNR (crop) | PSNR (heart) | SSIM | params | time, s |
|---|---|---|---|---|---|
| interpolation | 37.16 | 33.14 | 0.970 | - | - |
| siren_raw | 17.55 | 18.25 | 0.296 | 12,865 | 1 |
| siren_scaled_0.25 | 18.18 | 19.10 | 0.313 | 12,865 | 1 |
| siren_periodic_0.25 | 17.75 | 18.40 | 0.299 | 12,929 | 1 |
