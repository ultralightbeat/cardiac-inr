# Contributing

1. `main` is protected. Work on a branch: `feat/<role>-<topic>` (e.g. `feat/model-siren-layer`).
2. Small, frequent commits with imperative messages ("Add SIREN first-layer init").
3. Open a pull request early; at least one teammate reviews before merge. Link the issue.
4. Run `make lint test` before pushing.
5. Never commit data, checkpoints or large files (see `.gitignore`).
6. Non-trivial design choice? Add an entry to `docs/DECISIONS.md` in the same PR.
7. If AI helped write a block, mark it: `# AI-assisted: <what it does>` and make sure you can explain it.
8. Interface changes (shapes, function signatures in `docs/PROJECT_PLAN.md`, section 3) need a PR that updates the plan and a decision entry.
