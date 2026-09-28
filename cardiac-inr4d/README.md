# cardiac-inr4d

Implicit neural representation (SIREN) of a cardiac cine-MRI cycle: a small network that maps `(x, y, z, t)` to image intensity, so a 4D scan becomes a continuous, compact function.

Course project, team of 5. Longer-term motivation: continuous 4D cardiac representations as a building block for spatiotemporal alignment of multimodal medical data.

## Start here

- **Roles, interfaces, pipeline by stages:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
- Git workflow and rules: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Design decisions log: [`docs/DECISIONS.md`](docs/DECISIONS.md)
- Papers we rely on: [`docs/REFERENCES.md`](docs/REFERENCES.md)
- Who did what: [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md)
- Getting the data: [`data/README.md`](data/README.md)

## Quick start

```bash
git clone <repo-url> && cd cardiac-inr4d
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make test
```

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
results/        figures and tables (small files only)
data/           local data, NOT committed
docs/           plan, decisions, references, contributions
```

## Status

See milestones M1-M4 in the plan. Update this section as tags are released.
