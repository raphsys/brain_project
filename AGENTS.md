# Repository Guidelines

## Project Structure & Module Organization
- `src/brain_project/` is the main package. Core modules live under `src/brain_project/modules/` (perception, grouping, textures, invariance, proto-objects, recognition).
- Training and evaluation entry points live in `src/brain_project/train/` and `src/brain_project/eval/`.
- Shared helpers are in `src/brain_project/utils/`.
- Local datasets live in `data/` (e.g., CIFAR downloads), and experiment artifacts/checkpoints go in `runs/`.
- Exploratory work belongs in `notebooks/`. The `scripts/` directory is currently empty.

## Build, Test, and Development Commands
There is no build system or task runner configured; run modules directly with Python and the `src/` layout on `PYTHONPATH`.
- `PYTHONPATH=src python -m brain_project.train.train_m1` runs the M1 perception sanity check and downloads CIFAR-10 into `data/`.
- `PYTHONPATH=src python -m brain_project.train.train_m3` runs the M3 trainer.
- `PYTHONPATH=src python -m brain_project.eval.linear_probe` launches the linear probe evaluation.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation, type hints where helpful, and standard PEP 8 naming (snake_case for functions/variables, PascalCase for classes).
- Keep module files in `snake_case.py` and package names in `snake_case/`.
- No formatter or linter is configured in-repo; match the existing style in `src/brain_project/`.

## Testing Guidelines
- There is no `tests/` directory or test runner configured.
- If you add tests, prefer `pytest` with files named `test_*.py` under a new `tests/` directory, and document how to run them.

## Commit & Pull Request Guidelines
- The Git history has no commits yet, so there is no established commit message convention. Use concise, imperative summaries (e.g., "Add M3 trainer").
- For pull requests, include a short summary, note any dataset downloads or changes under `data/`, and avoid committing large artifacts from `runs/`.

## Data & Artifact Hygiene
- Treat `data/` and `runs/` as local, machine-specific outputs. If a file must be shared, document it explicitly and keep it minimal.
