# cardiac-inr4d

A cardiac cine-MRI scan turned into one continuous function. A small SIREN network maps
`(x, y, z, t)` (millimetres, time in the heartbeat) to image intensity. After training, the
network *is* the scan: it can be asked about any point and any moment, including places the
scanner never measured (between the 10 mm slices, between frames). From it we extract a beating
3D heart and its volumes.

Course project (Beihang University). Data: ACDC (MICCAI 2017), cine-MRI, one network per patient.

## Quick start

1. **Setup (once).** Creates `.venv`, installs everything, runs the tests.
   - Windows: double-click `setup.bat`, or `powershell -ExecutionPolicy Bypass -File setup.ps1`
     (add `-Mirror` for the Tsinghua pip mirror in mainland China)
   - Linux / macOS: `bash setup.sh` (`--mirror` likewise)
2. **Data.** Put an ACDC patient folder into `data/acdc/`, e.g. `data/acdc/patient051/` with
   `patient051_4d.nii(.gz)`, `Info.cfg` and the `*_frameXX_gt.nii(.gz)` label files.
   See [`data/README.md`](data/README.md). Any 4D NIfTI `[H, W, D, T]` also works for the experiments.
3. **Run the whole study** (activate first: `.venv\Scripts\activate` or `source .venv/bin/activate`):

```bash
python scripts/pipeline.py                                # first patient in data/, ~1 h on a laptop CPU
python scripts/pipeline.py --patient data/acdc/patient020 # a specific patient
python scripts/pipeline.py --patient scan_4d.nii.gz       # any 4D NIfTI (3D heart step needs ACDC labels)
python scripts/pipeline.py --preset quick                 # 5-minute check that everything runs
python scripts/pipeline.py --preset full                  # settings of the report, ~3 h
```

No GPU is needed. With a CUDA build of PyTorch the same code runs faster with the same results.

## What the pipeline does

| Step | Question | Output folder |
|---|---|---|
| E1 | Hide every second slice of one frame. Can SIREN / a ReLU MLP of the same size / linear interpolation rebuild them? | `results/<patient>/experiments/main` |
| E2 | Hide every second frame (all slices kept). Which time encoding works: raw `t`, scaled `t·s`, periodic `(sin, cos)`? | `results/<patient>/experiments/main` |
| E2, sparse | Keep only every 4th frame. SIREN vs ReLU vs interpolation, plus the fixed 0.5 blend of SIREN and interpolation | `results/<patient>/experiments/stride4` |
| Final | Fit the field on all data, query it on a 1.6 mm isotropic grid at any time, extract LV / RV / LV wall, volumes and ejection fraction, 3D/4D views | `results/<patient>/final/heart4d`, weights in `models/<patient>/field.pt` |

All tables are collected in `results/<patient>/SUMMARY.md`. Metrics are computed **only on data
hidden from the network**: PSNR and SSIM inside a crop around the heart, and PSNR inside the
heart itself (dilated expert labels), which is the number that matters.

Final outputs per patient (`results/<patient>/final/heart4d/`):
`heart_4d.html` (interactive beating heart, open in any browser, offline), `heart_3d.gif`,
`planes_4d.gif` (scanner slices vs neural field), `volume_curve.png`, `summary.md` (EDV, ESV, EF
vs expert labels), meshes `MYO/LV/RV_{ED,ES}.obj`.

## Main results (patient051, see [`results/patient051/SUMMARY.md`](results/patient051/SUMMARY.md))

| Finding | Numbers (heart-region PSNR on hidden data) |
|---|---|
| SIREN vs ReLU MLP of the same size (199k parameters) | 28.9 vs 22.3 dB (+6.6 dB) |
| Time encoding matters | raw `t` 23.5 dB → `t·0.25` 31.7 dB (every 2nd frame hidden); best scale 0.1 |
| SIREN vs linear interpolation, every 4th frame kept | 28.9 vs 29.2 dB; SIREN better in the fast contraction phase |
| Blend 0.5 SIREN + 0.5 interpolation | 29.65 vs 29.23 dB, better on 18 of 21 hidden frames |
| Compression vs quality | heart crop: 16× fewer numbers, 28.9 dB · whole scan: 86×, 27.5 dB |
| Hidden slices 20 mm apart (E1) | no method recovers them (≈ 20 dB for all): through-plane gap too large |
| Ejection fraction from the 4D field | LV 49 % (expert labels 44 %, same threshold on scanner slices 47 %), RV 51 % (53 %) |

## Individual scripts

```bash
python scripts/run_experiments.py --help     # E1 / E2 with every option (encodings, stride, width, steps ...)
python scripts/evaluate_blend.py <run dir>   # blend re-evaluation of a finished E2 run
python scripts/render_4d.py --train          # final field + 3D/4D heart; without --train re-renders models/<patient>/field.pt
pytest -q                                    # tests
```

## Repository layout

```
src/inr4d/
  data/         ACDC / NIfTI loading, coordinate normalisation, sampling, patient lookup
  models/       SIREN, ReLU / Fourier-feature MLP, time encodings
  training/     training loop (Adam, cosine learning rate, checkpoints)
  evaluation/   PSNR / SSIM / heart-region PSNR, linear-interpolation baselines
  viz/          figures, GIFs, 3D surfaces and the offline WebGL viewer
scripts/        pipeline.py (everything), run_experiments.py, evaluate_blend.py, render_4d.py
tests/          pytest
results/        per patient: experiments/, final/, SUMMARY.md (small files, committed)
models/         per patient: final field.pt (+ field.json), ~0.8 MB
data/           local data, never committed
docs/           plan, decision log, references
```

Design decisions and why: [`docs/DECISIONS.md`](docs/DECISIONS.md) · References: [`docs/REFERENCES.md`](docs/REFERENCES.md)
