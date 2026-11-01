# Held-out patients: hidden frames (E2) and geometry

Settings fixed on the development patient (051) before this run: SIREN 4 x 256, omega 30, time scale 0.1, 20,000 steps, blend = 0.5 SIREN + 0.5 interpolation. Hidden test frames: all except every 4th and the last. PSNR in the heart (dB). Geometry: label-assisted threshold on the final field (trained on all frames), ED / ES from Info.cfg, Dice on the scanner slices.

## Per patient

| patient | group | interp. | SIREN | ReLU | blend | SIREN - interp. | LV Dice ED | LV EF (expert) | LV EF err | RV EF err |
|---|---|---|---|---|---|---|---|---|---|---|
| patient016 | DCM | 33.31 | 32.42 | 21.74 | 33.45 | -0.89 | 0.909 | 11.4 (10.7) | +0.7 | -23.4 |
| patient017 | DCM | 33.55 | 33.01 | 25.50 | 33.89 | -0.53 | 0.892 | 13.7 (13.7) | +0.0 | -0.0 |
| patient018 | DCM | 28.40 | 27.82 | 23.65 | 28.75 | -0.58 | 0.917 | 29.4 (22.8) | +6.6 | +3.3 |
| patient019 | DCM | 29.45 | 29.08 | 22.45 | 29.89 | -0.37 | 0.917 | 34.3 (27.5) | +6.8 | +14.5 |
| patient020 | DCM | 33.53 | 32.50 | 24.29 | 33.62 | -1.03 | 0.835 | 19.7 (17.3) | +2.4 | +9.0 |
| patient022 | HCM | 34.46 | 34.51 | 26.43 | 35.19 | +0.05 | 0.851 | 75.3 (64.9) | +10.3 | +21.5 |
| patient025 | HCM | 36.41 | 35.79 | 31.00 | 36.54 | -0.62 | 0.856 | 68.8 (63.0) | +5.9 | -41.4 |
| patient030 | HCM | 40.95 | 40.29 | 29.90 | 41.51 | -0.66 | 0.811 | 83.0 (74.6) | +8.5 | -37.8 |
| patient035 | HCM | 27.50 | 27.20 | 19.91 | 28.04 | -0.30 | 0.860 | 65.8 (66.8) | -1.0 | -49.5 |
| patient040 | HCM | 30.20 | 29.02 | 21.00 | 30.35 | -1.18 | 0.825 | 54.0 (50.5) | +3.6 | -19.4 |
| patient041 | MINF | 26.10 | 25.52 | 21.44 | 26.55 | -0.59 | 0.903 | 38.6 (36.1) | +2.5 | -8.9 |
| patient045 | MINF | 27.58 | 27.45 | 19.95 | 28.17 | -0.13 | 0.912 | 35.2 (29.7) | +5.5 | +9.8 |
| patient048 | MINF | 31.74 | 31.25 | 23.17 | 32.07 | -0.49 | 0.832 | 27.7 (32.5) | -4.8 | -25.6 |
| patient050 | MINF | 36.48 | 35.82 | 27.24 | 36.85 | -0.67 | 0.887 | 22.5 (16.4) | +6.0 | -10.4 |

## Over patients (mean, std, 95% CI over patients)

| quantity | n | mean | std | 95% CI |
|---|---|---|---|---|
| PSNR heart, interpolation | 14 | 32.120 | 4.200 | 29.695 ... 34.545 |
| PSNR heart, SIREN | 14 | 31.548 | 4.149 | 29.153 ... 33.944 |
| PSNR heart, ReLU | 14 | 24.118 | 3.502 | 22.096 ... 26.140 |
| PSNR heart, blend | 14 | 32.491 | 4.182 | 30.077 ... 34.906 |
| SIREN - ReLU (paired) | 14 | 7.430 | 2.000 | 6.276 ... 8.585 |
| SIREN - interpolation (paired) | 14 | -0.571 | 0.329 | -0.761 ... -0.381 |
| blend - interpolation (paired) | 14 | 0.372 | 0.194 | 0.260 ... 0.484 |
| temporal error x1e3, interpolation | 14 | 2.658 | 2.323 | 1.316 ... 3.999 |
| temporal error x1e3, SIREN | 14 | 2.917 | 2.612 | 1.410 ... 4.425 |
| LV Dice ED | 14 | 0.872 | 0.038 | 0.850 ... 0.894 |
| LV Dice ES | 14 | 0.814 | 0.084 | 0.765 ... 0.863 |
| LV HD95 ED, mm | 14 | 6.862 | 1.731 | 5.863 ... 7.861 |
| LV EF error, points | 14 | 3.784 | 4.092 | 1.421 ... 6.147 |
| LV absolute EF error, points | 14 | 4.614 | 3.040 | 2.859 ... 6.369 |
| RV Dice ED | 14 | 0.753 | 0.105 | 0.692 ... 0.813 |
| RV EF error, points | 14 | -11.319 | 22.228 | -24.153 ... 1.515 |
| LV Dice ED, same threshold on scanner slices | 14 | 0.870 | 0.038 | 0.848 ... 0.891 |

## Paired tests (Wilcoxon signed-rank, two-sided)

- SIREN vs interpolation: better on 1 of 14 patients, median difference -0.58 dB, p = 0.000244
- blend vs interpolation: better on 14 of 14 patients, median difference +0.35 dB, p = 0.000122
- SIREN vs ReLU: better on 14 of 14 patients, median difference +7.77 dB, p = 0.000122

Patients are the unit of analysis (voxels are not independent observations). One seed per patient; seed spread on patient 051 is reported in VALIDITY.md.
