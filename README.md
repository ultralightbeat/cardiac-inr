# cardiac-inr4d — Neural Fields for Continuous Cardiac Cine-MRI

A SIREN neural network learns one cardiac cine-MRI scan as a continuous function of `(x, y, z, t)`
and reconstructs a beating 3D heart from it. Course project, Beihang University, Group 4.  

## 1. Install Python

Python 3.10–3.13: <https://www.python.org/downloads/> (on Windows tick **"Add python.exe to PATH"**).
No GPU needed.

## 2. Download the code

```bash
git clone https://github.com/ultralightbeat/cardiac-inr.git
```
or on GitHub: **Code → Download ZIP** and unzip.

## 3. Download the dataset

ACDC on Kaggle: <https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17>

Download and unzip it. Inside there is one folder per patient: `patient001` … `patient150`.

## 4. Put one patient into the project

Copy **any one patient folder** (the whole folder) into `data/acdc/`:

```
cardiac-inr/
└── data/
    └── acdc/
        └── patient001/
            ├── patient001_4d.nii.gz
            ├── patient001_frame01_gt.nii.gz
            ├── ...
            └── Info.cfg
```

## 5. Install

Windows: double-click **`setup.bat`**. Linux / macOS: `bash setup.sh`.
Wait for the line **`Ready.`** (3–10 min).

## 6. Run

Open a terminal in the project folder and run (replace `patient001` with your patient):

```bat
.venv\Scripts\python scripts\pipeline.py --patient data\acdc\patient001
```
Linux / macOS: `.venv/bin/python scripts/pipeline.py --patient data/acdc/patient001`

Takes about 1 hour on a laptop CPU. Finished when it prints `all done`.

## 7. Results

Saved to `results/patient001/`:

- `SUMMARY.md` — all result tables
- `final/heart4d/heart_4d.html` — interactive 3D beating heart (open in a browser)
- `experiments/` — figures of all experiments

The trained model is saved to `models/patient001/field.pt`.