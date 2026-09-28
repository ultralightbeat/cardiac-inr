# Project Plan: Continuous Representation of a Cardiac Cycle with an Implicit Neural Network

Team of 5. This document is the single source of truth for **scope, roles, interfaces and the development pipeline**. Update it via pull request when something changes; log the reasoning in `docs/DECISIONS.md`.

Timeline is given in relative weeks (W1-W6). Shift it to the real course deadline in the first meeting.

---

## 1. Goal and scope

**One-sentence idea.** A voxel grid stores a 3D/4D medical scan as a table of numbers. We replace it with a small neural network that takes a coordinate `(x, y, z, t)` and returns the image intensity at that point, so the whole cardiac cycle becomes one compact, continuous function.

**Data.** Cine cardiac MRI from the public ACDC challenge dataset (Bernard et al., IEEE TMI 2018). Each patient has a 4D series (a stack of short-axis slices over the cardiac cycle). Access requires accepting the dataset terms; **data is never committed to git** (see `data/README.md`).

**Scope is deliberately one patient, not a generalising model.** The network *overfits* one patient's cycle on purpose; that is the standard setting for implicit representations. Generalising across patients is out of scope.

| Level | What we deliver | Status |
|-------|-----------------|--------|
| **L1 - MVP (must have)** | SIREN fitted to one static 3D volume (a single time frame), compared with trilinear interpolation. | required |
| **L2 - Target** | 4D model `(x,y,z,t)` for one full cardiac cycle of one patient; evaluated by reconstructing a held-out time frame. | goal |
| **L3 - Stretch** | Ablation of two time-encoding strategies; repeat on 2-3 more patients; held-out slice experiment. | optional |

**Fallback rule.** If L1 does not work by the end of W3, we stop expanding and deliver L1 with a thorough ablation study instead of a shaky L2.

## 2. Why this model (draft for the report)

- Voxel grids scale as `H*W*D*T` in memory; a coordinate network's size is independent of resolution and can be queried at any point, including between slices and between time frames.
- A plain ReLU MLP suffers from *spectral bias*: it learns smooth, low-frequency structure first and blurs fine detail (papillary muscles, myocardial borders). SIREN (sine activations with a specific initialisation) and Fourier-feature encodings are the two well-known remedies. We implement SIREN as the main model and a ReLU+Fourier-features MLP as a comparison.
- This is a building block for the longer-term research direction: continuous 4D cardiac representations that later can be constrained by physics and aligned with other modalities.

## 3. Interface contracts (so everyone can work in parallel)

All code lives in `src/inr4d/`. Shapes are fixed here; change them only via PR + a `DECISIONS.md` entry.

```python
# data/  (Role 1)
load_acdc_patient(path) -> Volume4D          # .data: float32 [H, W, D, T], .spacing: (sx, sy, sz) in mm, .affine
class CoordinateDataset:                     # torch Dataset over ALL voxels of the selected frames
    coords: Tensor[N, 4]                     # (x, y, z, t), each normalised to [-1, 1] using PHYSICAL extent (mm), not voxel index
    values: Tensor[N, 1]                     # intensity normalised to [-1, 1]
    frame_ids: Tensor[N]                     # which original frame each sample came from
    def sample(self, n) -> (coords, values)  # random mini-batch of n points

# models/  (Role 2, Role 4)
class SIREN(nn.Module):
    def __init__(self, in_features=4, hidden_features=256, hidden_layers=3,
                 out_features=1, first_omega_0=30.0, hidden_omega_0=30.0): ...
    def forward(self, coords: Tensor[B, in_features]) -> Tensor[B, out_features]: ...

# training/  (Role 3)
train(model, dataset, cfg) -> TrainResult    # .checkpoint_path, .history (dict of lists)

# evaluation/  (Role 4, Role 5)
reconstruct_volume(model, shape_hwd, t, device) -> np.ndarray[H, W, D]   # query the network on a grid at time t
psnr(pred, target) -> float
ssim(pred, target) -> float
```

**Important detail:** ACDC slices are thick (roughly 5-10 mm) and there are few of them, while in-plane resolution is ~1.5-2 mm. If coordinates are normalised by voxel index, the anatomy is stretched incorrectly along `z`. Normalise by physical extent. (Verify the actual spacing values on your patient in the first data exploration notebook.)

## 4. Roles

One person can hold more than one hat if the team is short (see section 9). Everyone reviews at least one other role's pull requests.

### Role 1 - Data Engineer

**Mission:** turn raw NIfTI files into clean `(coordinate, intensity)` training data.

**Tasks**
- Get access to ACDC, load 4D NIfTI with `nibabel`, inspect shapes, spacing, number of frames, intensity range.
- Write `notebooks/01_data_exploration.ipynb` (slice viewer, intensity histograms, frame-by-frame montage).
- Implement `load_acdc_patient` and `CoordinateDataset` following the contract above (physical-extent normalisation, intensity normalisation, optional frame subset for hold-out experiments).
- Unit tests: coordinate ranges are in `[-1, 1]`, no NaNs, reshaping coords back to the volume reproduces the original.

**Deliverables:** `src/inr4d/data/*`, `tests/test_data.py`, exploration notebook, `data/README.md` kept accurate.

**Must understand:** what NIfTI is, voxel spacing vs. array shape, affine matrix (at the level of "maps voxel index to millimetres"), why we normalise, what a cardiac cine series is (frames = phases of one heartbeat).

**Done when:** a teammate can call the dataset and train on it without asking a single question.

### Role 2 - Model Architect (SIREN)

**Mission:** implement SIREN correctly and be able to explain why it works.

**Tasks**
- Read the SIREN paper carefully (Sitzmann et al., arXiv:2006.09661), especially the initialisation section.
- Implement the sine layer, first-layer vs. hidden-layer initialisation, `omega_0`.
- Sanity check on a **2D image** before any medical data (`notebooks/02_siren_2d_sanity.ipynb`): the network must reproduce a photo at high fidelity. This validates the implementation.
- Implement the comparison model: ReLU MLP with Fourier-feature input encoding (Tancik et al., arXiv:2006.10739).
- Write the "model choice" section of the report together with Role 5.

**Deliverables:** `src/inr4d/models/siren.py`, `models/fourier_mlp.py`, `tests/test_models.py`, 2D sanity notebook.

**Must understand:** what an MLP is, why activation functions matter, spectral bias (intuition is enough), why SIREN needs special initialisation (keeps activation distributions stable across depth).

**Done when:** 2D image fit reaches high PSNR in a few thousand steps, and a plain ReLU MLP is visibly blurrier on the same image.

### Role 3 - Training & Optimisation Engineer

**Mission:** a reliable, reproducible training loop.

**Tasks**
- Implement `train()` with Adam, mini-batch coordinate sampling (do not push all voxels through the network at once), MSE loss, LR schedule, checkpointing, fixed random seeds.
- Logging with TensorBoard (loss curve, periodic PSNR on a validation subset, periodic image preview).
- Config-driven runs via `configs/*.yaml`; every result must be reproducible from a config plus a git commit hash.
- Find stable hyperparameters (learning rate, batch size, steps) for L1, then hand over a "recipe" to Role 4.

**Deliverables:** `src/inr4d/training/*`, `configs/`, `tests/test_training.py` (a tiny model overfits a tiny synthetic volume), results table in `results/`.

**Must understand:** gradient descent and Adam, batch vs. epoch, overfitting (here it is the goal), how to read a loss curve and spot divergence.

**Done when:** one command trains L1 from a config and reproduces the same numbers with the same seed.

### Role 4 - Temporal Modelling & Evaluation Lead

**Mission:** extend from 3D to 4D and measure whether it actually works.

**Tasks**
- Implement two time-encoding options and compare them (this is the main original design decision of the project):
  - **A. Concatenation:** feed `t` as a 4th input coordinate to the same network.
  - **B. Per-frame latent code:** learn one small vector per training frame, feed `[x, y, z, latent(t)]` (auto-decoder idea from DeepSDF, arXiv:1901.05103). Think about how to obtain a latent for an *unseen* time (interpolate neighbouring codes?) and document the choice.
- Background reading for intuition only, no code reuse: D-NeRF (arXiv:2011.13961) shows how time is handled for dynamic scenes.
- Define and implement the evaluation protocol (section 7): held-out frame, PSNR, SSIM, temporal smoothness check, compression ratio.
- Implement `reconstruct_volume` for arbitrary `t`.

**Deliverables:** `src/inr4d/models/temporal.py`, `src/inr4d/evaluation/*`, `tests/test_evaluation.py`, comparison table A vs. B.

**Must understand:** what interpolation between time frames means, what PSNR/SSIM measure, why held-out evaluation matters (fitting training frames proves little).

**Done when:** the network reconstructs a frame it never saw, and the numbers are compared against the baseline.

### Role 5 - Baselines, Visualisation & Report Lead

**Mission:** make the results convincing and the report coherent.

**Tasks**
- Implement baselines: trilinear spatial interpolation and linear interpolation between neighbouring time frames.
- Produce figures: side-by-side slices (ground truth / baseline / SIREN / error map), GIF or video of the reconstructed cardiac cycle including intermediate times that do not exist in the data.
- Maintain `docs/REFERENCES.md`, `docs/DECISIONS.md`, `docs/CONTRIBUTIONS.md`.
- Assemble the final report and presentation; write the model-justification section with Role 2.

**Deliverables:** `src/inr4d/viz/*`, baseline code in `src/inr4d/evaluation/baselines.py`, figures in `results/figures/`, report draft.

**Must understand:** what the baselines do and why they are fair comparisons, how to read the metrics, how to tell a clear scientific story (problem -> idea -> method -> result -> limits).

**Done when:** a person who never saw the code understands the problem and result from the figures alone.

### Project Lead hat (part-time, on top of one role)

Owns the repo settings, milestones and issues, runs the weekly sync (15-20 min), reviews and merges PRs, resolves interface disputes, keeps `PROJECT_PLAN.md` current, coordinates with the instructor. The lead does not have to be the strongest coder, but has to be the most organised person.

## 5. Pipeline by stages

| # | Stage | Owner(s) | Output | Definition of done | Week |
|---|-------|----------|--------|--------------------|------|
| 0 | Setup & alignment | Lead, all | Repo, environment, roles agreed, papers assigned | Everyone can run `make test`; each member has read their paper | W1 |
| 1 | Data pipeline | R1 | `CoordinateDataset` + exploration notebook | Contract satisfied, tests green | W1-W2 |
| 2a | SIREN + 2D sanity check | R2 | `SIREN`, `FourierMLP`, 2D notebook | 2D fit works; SIREN sharper than ReLU MLP | W1-W2 |
| 2b | Baselines | R5 | Trilinear + temporal-linear interpolation | Baselines run on the dataset | W2 |
| 3 | Static 3D fit (**L1**) | R3 + R2 | Trained model, config, curves | PSNR/SSIM on a subset of voxels reported vs. baseline | W2-W3 |
| **G** | **Gate: go / no-go for 4D** | Lead | Decision in `DECISIONS.md` | L1 works -> proceed; otherwise fallback rule | end W3 |
| 4 | 4D extension (**L2**) | R4 (+R3 for training) | Time encoding A and B implemented | Both variants train without divergence | W3-W5 |
| 5 | Evaluation | R4 + R5 | Results table | Held-out-frame PSNR/SSIM, compression ratio, ablations on `omega_0`, width, depth | W4-W5 |
| 6 | Visualisation | R5 | Figures, GIF/video | Figures regenerate from a script | W5 |
| 7 | Report & presentation | R5 + all | Final report, slides, demo | Every member can explain their module | W6 |
| 8 | Freeze & handover | Lead | Tag `v1.0`, clean README | Fresh clone -> reproduce the main table from documented commands | W6 |

Stage 2a and 2b, and Stage 1, run in parallel. Nobody has to wait for anybody in W1-W2.

## 6. Milestones

- **M1 (end W2):** dataset works, SIREN works on 2D, baselines run. Tag `v0.1-setup`.
- **M2 (end W3):** static 3D result vs. baseline. Tag `v0.2-static`. Gate decision recorded.
- **M3 (end W5):** 4D result with held-out-frame evaluation. Tag `v0.3-4d`.
- **M4 (end W6):** report and presentation. Tag `v1.0`.

## 7. Evaluation protocol (fixed, so results are comparable)

1. **Primary subject:** one ACDC patient chosen once (record the ID in `configs/default.yaml`), all main tables use it.
2. **Static (L1):** fit one frame. Compare against trilinear interpolation on **held-out voxels/slices** (for example, drop every second slice, reconstruct it) - reconstructing the training voxels alone proves only memorisation.
3. **Temporal (L2):** train with one middle frame removed from the data, reconstruct it at its time `t`, compare with linear interpolation of the two neighbouring frames.
4. **Metrics:** PSNR, SSIM (on normalised intensities, state the range), plus parameter count vs. number of stored voxels (compression ratio).
5. **Robustness (L3):** repeat with 3 different random seeds and report mean and spread; optionally repeat on 2-3 additional patients.
6. **Ablations:** `omega_0`, width, depth, SIREN vs. Fourier-feature MLP, time encoding A vs. B.

## 8. Git workflow and originality practices

- `main` is protected: no direct pushes; every change comes via pull request with at least one review.
- Branch naming: `feat/<role>-<topic>`, for example `feat/data-coordinate-dataset`. Small commits, imperative messages ("Add physical-extent normalisation").
- One GitHub issue per stage task; link PRs to issues.
- **Do not commit data or checkpoints** (`.gitignore` covers this).
- Keep a real commit history from day one: commit as you work, not one big upload at the end.
- **Decision log:** every non-trivial design choice goes into `docs/DECISIONS.md` (context, options, decision, why). It is the best proof that the work is original, and it feeds the report.
- **AI assistants are allowed if the course allows it:** mark generated blocks with a short comment (`# AI-assisted: <what it does>`), read and understand every line, and never paste code from other repositories. Cite every paper you rely on in `docs/REFERENCES.md`.
- **Personal accountability:** each member fills in their section of `docs/CONTRIBUTIONS.md` (what I implemented, what I can explain). Expect oral questions about your own module.

## 9. Risks and fallbacks

| Risk | Mitigation |
|------|------------|
| Team has little deep-learning background | Roles 1 and 5 need only careful Python. Lead and the strongest person take Roles 2 and 4. Weekly 20-minute "explain your module" session. |
| Fewer than 5 active members | Merge R3 into R2 and R5 into R1. Cut L3 entirely. |
| 4D does not work by W5 | Deliver L1 + clean ablations + a documented 4D attempt (negative results with analysis are acceptable). |
| No GPU | Overfitting one patient with a 256x3 network is feasible on a modest GPU or free cloud notebooks; reduce points per step, or fit a cropped region around the heart. |
| Dataset access delay | Use a public 3D volume or a 2D-plus-time toy (image sequence) to keep R2-R4 unblocked until ACDC is available. |
| Someone disappears | Every module has a reviewer who has read the code (rule in section 8). |

## 10. Team rhythm

- Weekly sync, 15-20 minutes: what is done, what is blocked, what is next.
- Mid-week async check-in in the team chat.
- Demo at each milestone: run something, show a figure.
