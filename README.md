# Week 2 CoRS

This repository organizes the Week 2 CoRS project work across sampling, parsing, selection, evaluation, and perturbation. It provides the project layout and team workflow; the model and CoRS pipeline are not implemented here.

## Folders

- `sampling/`: sample selection and sampling notes.
- `parsing/`: input and output parsing work.
- `selection/`: selection methods and related notes.
- `eval/`: evaluation scripts, metrics, and reports. Tune on the dev split only.
- `perturb/`: perturbation methods and related notes.
- `configs/`: run configurations and frozen hyperparameters.
- `scripts/`: commands and scripts for repeatable runs.

## Team roles

Fill in each name as the team assigns roles.

| Role | Name |
| --- | --- |
| Sampling lead | |
| Parsing lead | |
| Selection lead | |
| Evaluation lead | |
| Perturbation lead | |

## Model outputs and run records

Store model outputs in the team’s shared Palmetto folder, not in this repository. Write one JSONL file per run. Each JSONL line represents one output and must include these fields:

`clip_id`, `sample_id`, `model`, `temperature`, `seed`, `reasoning_text`, `trajectory`, `log_prob` (when available), `runtime_s`, and `perturbation`.

Reuse cached outputs when the run configuration and inputs match. Use fixed seeds. Log every run’s configuration. Tune only on the dev split. Before running test evaluation, commit the frozen hyperparameters to `configs/` through a pull request.

## Changes and reviews

Make changes on a branch and open a pull request targeting `main`. Merge only after at least one teammate approves the pull request. Do not commit project changes directly to `main`.

Repository maintainers should protect `main` with required pull requests and at least one approving review. Require approval from someone other than the pull request author and dismiss stale approvals when new commits are pushed.
