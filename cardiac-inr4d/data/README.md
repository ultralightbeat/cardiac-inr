# Data

Data is **never committed** (see `.gitignore`).

## ACDC (Automated Cardiac Diagnosis Challenge)

1. Register / accept the dataset terms on the official challenge page (search "ACDC cardiac MICCAI 2017 challenge").
2. Download the training set and unpack it here as `data/acdc/patientXXX/...`.
3. Each patient folder contains a 4D NIfTI series (`patientXXX_4d.nii.gz`) plus the annotated end-diastole / end-systole frames. Confirm the exact file names on your download.
4. Set the chosen patient ID in `configs/default.yaml`.

Record in this file: dataset version, the patient ID used for the main experiments, and any preprocessing applied.
