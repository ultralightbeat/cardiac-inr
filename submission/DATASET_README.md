# Dataset used in the project (Group 4)

**Project:** Neural Fields for Continuous Cardiac Cine-MRI

## Source

ACDC — Automated Cardiac Diagnosis Challenge (MICCAI 2017), public cine cardiac MRI with expert labels.
Every use of the ACDC data must cite:

> O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac
> Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" IEEE Transactions on Medical Imaging,
> vol. 37, no. 11, pp. 2514–2525, Nov. 2018. doi: 10.1109/TMI.2018.2837502

The full dataset (150 patients) is available from the Human Heart Project / CREATIS (University of Lyon) after
accepting its terms of use. This archive contains only the two patients used in our experiments, unchanged.

## Contents

| Folder | Patient | Diagnosis group | Size | Role in the project |
|---|---|---|---|---|
| `patient051/` | 051 | MINF (previous myocardial infarction) | 224 × 256 px of 1.64 mm, 10 slices × 10 mm, 30 frames | main patient: all experiments, final model, 3D/4D heart |
| `patient015/` | 015 | DCM (dilated cardiomyopathy) | 256 × 216 px of 1.37 mm, 9 slices × 10 mm, 21 frames | second patient: hidden-slice test |

Files in each folder:

- `patientXXX_4d.nii` — the 4D cine series [H, W, slices, frames] (NIfTI)
- `patientXXX_frameNN.nii` — the end-diastolic and end-systolic frames as 3D volumes
- `patientXXX_frameNN_gt.nii` — expert labels for those frames: 0 background, 1 right-ventricle cavity,
  2 left-ventricle myocardium, 3 left-ventricle cavity
- `Info.cfg` — end-diastole (ED) and end-systole (ES) frame numbers (1-based), group, height, weight, number of frames
- `MANDATORY_CITATION.md` — the citation above

Patient 051: ED = frame 1, ES = frame 11. Patient 015: ED = frame 1, ES = frame 10.

## How the project uses it

Put the patient folders into `data/acdc/` of the code repository (github.com/ultralightbeat/cardiac-inr) and run

```
setup.bat
python scripts/pipeline.py --patient data/acdc/patient051
```

Preprocessing (done in code, the files are never modified): voxel centres converted to millimetres, centred
and divided by half of the largest physical extent; frame k of T mapped to t = 2k/(T−1) − 1; intensities
min–max scaled to [−1, 1] over the whole 4D volume. The expert labels are used only to place a crop around the
heart, to score PSNR inside the heart, and to choose the blood threshold and the ventricle regions for the 3D model.
