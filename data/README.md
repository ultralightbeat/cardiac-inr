# Data

Data is **never committed** (see `.gitignore`).

## ACDC (Automated Cardiac Diagnosis Challenge)

1. Register / accept the dataset terms on the official challenge page (search "ACDC cardiac MICCAI 2017 challenge").
2. Download the training set and unpack it here as `data/acdc/patientXXX/...`.
3. Each patient folder contains a 4D NIfTI series (`patientXXX_4d.nii.gz`) plus the annotated end-diastole / end-systole frames. Confirm the exact file names on your download.
4. Set the chosen patient ID in `configs/default.yaml`.

## Project record

- Dataset: ACDC training set, downloaded according to the challenge terms (exact release
  version is not available in this checkout).
- Main patient: `patient001` (see `configs/default.yaml`).
- Preprocessing: NIfTI values are loaded as `float32`; `CoordinateDataset` min-max
  normalises intensities to `[-1, 1]` over the complete patient volume and normalises
  spatial coordinates using voxel spacing from the NIfTI header.
- No NIfTI files or model checkpoints are committed to git.
