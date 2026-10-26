# Data

Data is **never committed** (ACDC terms of use; see `.gitignore`).

## ACDC (Automated Cardiac Diagnosis Challenge, MICCAI 2017)

1. Download the ACDC dataset from the official site (Human Heart Project, CREATIS / University of Lyon;
   search "ACDC challenge dataset") and accept its terms.
2. Copy patient folders here, one folder per patient:

```
data/acdc/patient051/
    patient051_4d.nii            (or .nii.gz) the 4D cine series [H, W, D, T]
    patient051_frame01_gt.nii    expert labels at end-diastole  (1 RV, 2 myocardium, 3 LV)
    patient051_frame11_gt.nii    expert labels at end-systole
    Info.cfg                     ED / ES frame numbers, group, height, weight
```

3. Run `python scripts/pipeline.py` (first patient found under `data/`) or pass `--patient`.

## Other data

Any 4D NIfTI file `[H, W, D, T]` can be passed with `--patient path/to/file.nii.gz`. Without labels:
- the experiments still run (the "heart" PSNR is then computed on the whole image, no crop);
- the 3D/4D heart step is skipped, because it needs the labels to tell the left and right ventricle
  apart and to set the blood threshold.

## Project record

- Main patient of the report: `patient051` (ACDC training set, group MINF, 224 × 256 × 10 slices × 30 frames,
  1.64 × 1.64 mm pixels, 10 mm slice spacing, ED = frame 1, ES = frame 11).
- Preprocessing: intensities min-max normalised to [-1, 1] over the whole 4D volume; coordinates from voxel
  centres in millimetres, centred and divided by half of the largest physical extent; t = 2k/(T-1) - 1,
  multiplied by the time scale (0.1) inside the model.
