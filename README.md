# cardiac-inr4d — Neural Fields for Continuous Cardiac Cine-MRI

Implicit neural representation (SIREN) of a cardiac cine-MRI cycle: a small network that maps `(x, y, z, t)` to image intensity, so a 4D scan becomes a continuous, compact function.

<<<<<<< Updated upstream
Course project, team of 5. Longer-term motivation: continuous 4D cardiac representations as a building block for spatiotemporal alignment of multimodal medical data.

## Start here

- **Roles, interfaces, pipeline by stages:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
- Git workflow and rules: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Design decisions log: [`docs/DECISIONS.md`](docs/DECISIONS.md)
- Papers we rely on: [`docs/REFERENCES.md`](docs/REFERENCES.md)
- Who did what: [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md)
- Getting the data: [`data/README.md`](data/README.md)
=======
Course project (Beihang University, Group 4). Data: ACDC (MICCAI 2017), cine-MRI, one network per patient.
>>>>>>> Stashed changes

---

<<<<<<< Updated upstream
```bash
git clone <repo-url> && cd cardiac-inr4d
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make test
=======
## How to run it — step-by-step guide

**Short version:** install Python → download the ACDC dataset from Kaggle → copy any patient folder
into `data/acdc/` → double-click `setup.bat` → run one command → open `results/<patient>/`.

### 0. Want to see the results without running anything?

The results of our report (patient051) are already in the repository:

| Open this file | What you see |
|---|---|
| `results/patient051/final/heart4d/heart_4d.html` | **Interactive beating 3D heart** — double-click, opens in any browser, works offline |
| `results/patient051/SUMMARY.md` | All result tables in one place |
| `results/patient051/final/heart4d/heart_3d.gif`, `planes_4d.gif` | Animations (3D heart; scanner slices vs neural field) |
| `results/patient051/final/heart4d/summary.md`, `volume_curve.png` | Heart volumes and ejection fraction vs expert labels |
| `results/patient051/experiments/*/` | Figures (`*.png`, `*.gif`) and tables (`results.md`) of every experiment |
| `models/patient051/field.pt` | The trained neural field (0.8 MB) |

Steps 1–6 below are needed to run the model yourself.

### 1. What you need

- Windows 10/11, Linux or macOS. **No GPU needed** (a normal laptop CPU is enough).
- **Python 3.10, 3.11, 3.12 or 3.13** — <https://www.python.org/downloads/>.
  On Windows tick **"Add python.exe to PATH"** in the installer.
- About 3 GB of free disk space and an internet connection for the first setup.

Get the code (either way):
```bash
git clone https://github.com/ultralightbeat/cardiac-inr.git
cd cardiac-inr
```
or download the repository as ZIP from GitHub (green **Code** button → **Download ZIP**) and unzip it.

### 2. Download the dataset

The data is **not** in the repository. Download it from Kaggle:

**ACDC — Automated Cardiac Diagnosis Challenge (MICCAI'17):**
<https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17>

1. Open the link, sign in to Kaggle (free account) and press **Download**.
2. Unzip the downloaded archive anywhere.
3. Inside you will find one folder per patient: `patient001`, `patient002`, … `patient150`.

### 3. Put a patient into the project

The model is trained **per patient**, so you need **one patient folder — any of them**.
Copy the **whole folder** (not single files) into `data/acdc/` of the project.
For example, with `patient001` the result must look like this:

```
cardiac-inr/
└── data/
    └── acdc/
        └── patient001/                       ← any patientXXX folder from the dataset
            ├── patient001_4d.nii.gz          ← the 4D cine-MRI series (required)
            ├── patient001_frame01.nii.gz
            ├── patient001_frame01_gt.nii.gz  ← expert labels, end-diastole
            ├── patient001_frame12.nii.gz
            ├── patient001_frame12_gt.nii.gz  ← expert labels, end-systole
            └── Info.cfg
```

- The frame numbers in the file names differ from patient to patient — that is normal.
- `.nii` and `.nii.gz` both work; do not rename or unpack the files.
- You can put several patients into `data/acdc/` and choose one with `--patient` (step 5).
- The `*_gt` label files are needed for the 3D heart and the heart-region metrics. Patients 001–100
  (training part) have them; for patients without labels the experiments still run, the 3D heart step is skipped.
- To compare with the numbers of our report, use `patient051`.

### 4. Install (once)

**Windows:** double-click **`setup.bat`** in the project folder.
(Or in PowerShell: `powershell -ExecutionPolicy Bypass -File setup.ps1`.)

**Linux / macOS:** `bash setup.sh`

The script creates a virtual environment `.venv`, installs all packages (PyTorch, nibabel, …)
and runs the tests. It ends with the green line **`Ready.`** It takes 3–10 minutes, mostly downloading PyTorch.

> In mainland China pip downloads can be very slow: use `setup.bat -Mirror` (Linux/macOS: `bash setup.sh --mirror`),
> which uses the Tsinghua pip mirror.

### 5. Run the model

Open a terminal **in the project folder** (Windows: in Explorer type `cmd` in the address bar and press Enter).
The commands use the Python of `.venv` directly, so nothing needs to be "activated".
Replace `patient001` with the folder you copied in step 3.

**a) Quick check — about 5 minutes** (tiny networks, numbers are not meaningful, only proves everything works):

```bat
.venv\Scripts\python scripts\pipeline.py --patient data\acdc\patient001 --preset quick
```

**b) Full run — trains the model and runs all experiments:**

```bat
.venv\Scripts\python scripts\pipeline.py --patient data\acdc\patient001
```
About 1 hour on a laptop CPU. For the exact settings of our report (~3 hours) add `--preset full`.

On Linux/macOS write `.venv/bin/python scripts/pipeline.py --patient data/acdc/patient001 ...` instead.
Without `--patient` the first patient found in `data/` is used.

The terminal shows each step and its time. The run is finished when it prints
`all done in N min -> results\patient001\SUMMARY.md`.

**c) Only the 3D heart from an already trained model** (no training, after b):

```bat
.venv\Scripts\python scripts\render_4d.py --patient data\acdc\patient001
```
For `patient051` this works right away with our trained model `models/patient051/field.pt`.

### 6. Look at the results

Everything is written into `results/<patient>/`, e.g. `results/patient001/`:

```
results/patient001/
├── SUMMARY.md                 ← START HERE: all tables of all steps
├── experiments/
│   ├── main/                  E1 (hidden slices) + E2 (hidden frames, 3 time encodings)
│   │     results.md, e1_hidden_slice.png, e2_hidden_frame.png, e2_cycle.gif, *_bars.png, *_training.png
│   └── stride4/               only every 4th frame kept: SIREN vs ReLU vs interpolation (+ blend.md)
└── final/heart4d/
      heart_4d.html            ← interactive beating 3D heart (open in a browser, rotate with the mouse)
      heart_3d.gif             3D heart animation
      planes_4d.gif            scanner slices vs neural field over the heartbeat
      volume_curve.png         LV / RV volume over the heartbeat
      summary.md               EDV, ESV, ejection fraction vs expert labels
      LV/RV/MYO_ED/ES.obj      3D meshes (Windows 3D Viewer, Blender, MeshLab)
models/patient001/field.pt     the trained neural field
```

`.md` files open in any text editor, VS Code or on GitHub. With `--preset quick` the same files are in
`quick_main/`, `quick_stride4/`, `final/quick_heart4d/` and `quick_SUMMARY.md`.

### If something goes wrong

| Message / problem | Fix |
|---|---|
| `Python 3.10-3.13 not found` | Install Python from python.org with "Add to PATH" ticked, then run `setup.bat` again |
| `No patient data yet` (end of setup) | Data is not in place yet — do step 3, then continue with step 5 (setup does not need to be repeated) |
| `--patient ...: not found` / `no *_4d.nii(.gz) found` | Wrong folder structure: `data/acdc/patientXXX/patientXXX_4d.nii.gz` must exist (watch for a doubled folder like `patientXXX/patientXXX/`) |
| `no expert labels (*_gt.nii) ... skipping the 3D/4D heart step` | This patient has no `*_gt` files — copy the whole folder, or take a patient from 001–100 |
| `'.venv\Scripts\python' is not recognized` | Run the command from the project folder, and run `setup.bat` first |
| Downloads are very slow / time out | `setup.bat -Mirror` |
| PowerShell says scripts are disabled | Use `setup.bat`, or the `-ExecutionPolicy Bypass` command from step 4 |


---

## What the pipeline does

| Step | Question | Output folder |
|---|---|---|
| E1 | Hide every second slice of one frame. Can SIREN / a ReLU MLP of the same size / linear interpolation rebuild them? | `results/<patient>/experiments/main` |
| E2 | Hide every second frame (all slices kept). Which time encoding works: raw `t`, scaled `t·s`, periodic `(sin, cos)`? | `results/<patient>/experiments/main` |
| E2, sparse | Keep only every 4th frame. SIREN vs ReLU vs interpolation, plus the fixed 0.5 blend of SIREN and interpolation | `results/<patient>/experiments/stride4` |
| Final | Fit the field on all data, query it on a 1.6 mm isotropic grid at any time, extract LV / RV / LV wall, volumes and ejection fraction, 3D/4D views | `results/<patient>/final/heart4d`, weights in `models/<patient>/field.pt` |

Metrics are computed **only on data hidden from the network**: PSNR and SSIM inside a crop around
the heart, and PSNR inside the heart itself (dilated expert labels), which is the number that matters.

Presets of `pipeline.py`: `quick` ~5 min (check only), `standard` ~1 h (default), `full` ~3 h (settings of the report).
With a CUDA build of PyTorch the same code runs faster with the same results.
Any 4D NIfTI `[H, W, D, T]` can also be passed with `--patient path/to/scan_4d.nii.gz`
(the 3D heart step then needs ACDC-style labels and is skipped without them).

## Main results (patient051, see [`results/patient051/SUMMARY.md`](results/patient051/SUMMARY.md))

| Finding | Numbers (heart-region PSNR on hidden data) |
|---|---|
| SIREN vs ReLU MLP of the same size (199k parameters) | 28.9 vs 22.3 dB (+6.6 dB) |
| Time encoding matters | raw `t` 23.5 dB → `t·0.25` 31.7 dB (every 2nd frame hidden); best scale 0.1 |
| SIREN vs linear interpolation, every 4th frame kept | 28.9 vs 29.2 dB; SIREN better in the fast contraction phase |
| Blend 0.5 SIREN + 0.5 interpolation | 29.65 vs 29.23 dB, better on 18 of 21 hidden frames |
| Compression vs quality | heart crop: 16× fewer numbers, 28.9 dB · whole scan: 86×, 27.5 dB |
| Hidden slices 20 mm apart (E1) | raw z: SIREN fills the gap with high-frequency noise. With the slice coordinate scaled by 0.25: 20.09 vs 20.01 dB in the heart, 19.79 vs 19.55 dB in the crop (patient051); patient015 (z scale 0.5): 20.8 vs 20.1 dB in the crop, 20.4 vs 20.7 in the heart. No method truly recovers a 20 mm gap |
| Ejection fraction from the 4D field | LV 49 % (expert labels 44 %, same threshold on scanner slices 47 %), RV 51 % (53 %) |

## Individual scripts

```bash
python scripts/run_experiments.py --help     # E1 / E2 with every option (encodings, stride, width, steps ...)
python scripts/evaluate_blend.py <run dir>   # blend re-evaluation of a finished E2 run
python scripts/render_4d.py --train          # final field + 3D/4D heart; without --train re-renders models/<patient>/field.pt
pytest -q                                    # tests
>>>>>>> Stashed changes
```
(`python` = `.venv\Scripts\python` on Windows or `.venv/bin/python` on Linux/macOS, or activate `.venv` first.)

## Repository layout

```
src/inr4d/
  data/         loading ACDC, CoordinateDataset            (Role 1)
  models/       SIREN, Fourier-feature MLP, time encoding  (Roles 2, 4)
  training/     training loop, logging, checkpoints        (Role 3)
  evaluation/   metrics, reconstruction, baselines         (Roles 4, 5)
  viz/          figures, GIF/video                         (Role 5)
configs/        YAML run configurations
notebooks/      exploration and sanity checks
tests/          pytest
<<<<<<< Updated upstream
results/        figures and tables (small files only)
data/           local data, NOT committed
docs/           plan, decisions, references, contributions
```

## Status

See milestones M1-M4 in the plan. Update this section as tags are released.
=======
results/        per patient: experiments/, final/, SUMMARY.md (small files, committed)
models/         per patient: final field.pt (+ field.json), ~0.8 MB
data/           local data, never committed
docs/           plan, decision log, references
submission/     final report (PDF/DOCX), presentation
```

Design decisions and why: [`docs/DECISIONS.md`](docs/DECISIONS.md) · References: [`docs/REFERENCES.md`](docs/REFERENCES.md)

## Data citation

ACDC: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky et al., "Deep Learning Techniques for Automatic MRI
Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE TMI 37(11):2514–2525, 2018.
doi: 10.1109/TMI.2018.2837502
>>>>>>> Stashed changes
