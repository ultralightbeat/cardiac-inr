# cardiac-inr4d — Neural Fields for Continuous Cardiac Cine-MRI

A SIREN neural network learns one cardiac cine-MRI scan as a continuous function of `(x, y, z, t)`
and reconstructs a beating 3D heart from it. Course project, Beihang University, Group 4.

## 1. Install Python

Python 3.10–3.13: <https://www.python.org/downloads/> (on Windows tick **"Add python.exe to PATH"**).
On a Mac check `python3 --version`; if it is older than 3.10, install from python.org or `brew install python@3.12`.
No GPU needed (on Apple Silicon the code runs on the CPU; results are the same, only speed differs).

## 2. Download the code

```bash
git clone https://github.com/ultralightbeat/cardiac-inr.git
cd cardiac-inr
```
or on GitHub: **Code → Download ZIP** and unzip.

## 3. Download the dataset
**Option A (recommended): the 16 ACDC patients used in our report** — If you want to use the complete patient data, you can directly utilize the automatic download script we previously wrote. In the terminal (command line), input:

```bash
.venv\Scripts\python scripts\download_data.py
```

**Option B : the 16 ACDC patients used in our report** — `Group4_Dataset_ACDC_16patients.zip`
(private link in the submission e-mail). Unzip it **into the project folder**, so that the files end up in
`data/acdc/patient051/patient051_4d.nii` etc.:

```bash
unzip ~/Downloads/Group4_Dataset_ACDC_16patients.zip -d .
```

**Option C: the full ACDC dataset** (150 patients), e.g. on Kaggle:
<https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17>.
Inside there is one folder per patient, `patient001` … `patient150`; `.nii.gz` files work unchanged.

Data source and mandatory citation: ACDC, MICCAI 2017 — O. Bernard et al., IEEE Trans. Med. Imag. 37(11), 2018.
The data are never committed to this repository.

## 4. Put patients into the project

With option A this is already done. With option B copy **any patient folder** (the whole folder) into `data/acdc/`:

```
cardiac-inr/
└── data/
    └── acdc/
        └── patient051/
            ├── patient051_4d.nii(.gz)
            ├── patient051_frame01_gt.nii(.gz)
            ├── ...
            └── Info.cfg
```

## 5. Install

Windows: double-click **`setup.bat`**. Linux / macOS (incl. Apple Silicon): `bash setup.sh`.
Wait for the line **`Ready.`** (3–10 min). It creates `.venv`, installs everything and runs the tests.

## 6. Run

Open a terminal in the project folder and run (replace `patient051` with your patient):

```bat
.venv\Scripts\python scripts\pipeline.py --patient data\acdc\patient051 --preset quick
.venv\Scripts\python scripts\pipeline.py --patient data\acdc\patient051
```
Linux / macOS:
```bash
.venv/bin/python scripts/pipeline.py --patient data/acdc/patient051 --preset quick   # ~5 min check
.venv/bin/python scripts/pipeline.py --patient data/acdc/patient051                  # main study
```

The main run takes about 1 hour on a laptop CPU. Finished when it prints `all done`.

## 7. Results

Saved to `results/patient051/`:

- `SUMMARY.md` — all result tables
- `final/heart4d/heart_4d.html` — interactive 3D beating heart (open in a browser)
- `experiments/` — figures of all experiments

The trained model is saved to `models/patient051/field.pt`.

## Main findings (from the report)

- A sine network (SIREN) is much better than an ordinary ReLU network of the same size, on every patient tested.
- Slowing the network down along time (time coordinate × 0.1 inside the model) is the most important single choice.
- Alone, the network is slightly behind linear interpolation; an even mix of both is better than interpolation
  on all 14 new patients.
- The network stores the heart region with about 16× and the whole scan with about 86× fewer numbers.
- The 3D heart gives a left-ventricle ejection fraction close to the experts' (42.7 % vs 43.6 %, patient 051).

Details: `results/patient051/SUMMARY.md`, `results/multi_patient/MULTI_PATIENT.md`.

## More scripts

```bash
python scripts/run_experiments.py --help     # single experiments with every option
python scripts/run_patients.py               # the 14 held-out patients of the report
python scripts/render_4d.py --train          # final field + 3D/4D heart
pytest -q                                    # tests
```
Every script accepts `--device auto|cpu|cuda|xpu`.

Project plan: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) · Decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md) ·
References: [`docs/REFERENCES.md`](docs/REFERENCES.md)
