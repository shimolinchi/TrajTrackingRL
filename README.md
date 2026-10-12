# TrajTrackingRL

PPO trajectory tracking for xArm7 + Wuji, with **incremental arm targets and absolute hand targets**.
This repository contains one baseline, extracted from the validated Tracking implementation.
It contains no GCRL, DAgger, base policy or residual-learning task.

The arm command at 60 Hz is `q_target[t] = clamp(q_target[t-1] + 1.5/60 * clamp(action[t], -1, 1))`.
The twenty hand commands map to absolute joint limits. Reset initializes the integrator from the actual joint position.
See [baseline settings](docs/baseline.md) and [architecture](docs/architecture.md).

## Install

Use Linux, Python 3.11 and a dedicated Isaac Sim 5.1 / Isaac Lab 2.3.2 environment.
The tested runtime uses Torch 2.7.0 + CUDA 12.8. Install Isaac through its own distribution first;
this package does not install or replace Isaac.

```bash
git clone --recurse-submodules git@github.com:shimolinchi/TrajTrackingRL.git
cd TrajTrackingRL
python -m pip install -e '.[browser,wandb,test]'
python -m playwright install chromium
cp configs/local.env.example configs/local.env
```

The two pinned upstream repositories and robot/object assets require separate access.
If the URDF importer stalls fetching its SDK dependency, run `bash scripts/setup_isaac_extensions.sh`
and set `TRAJTRACKING_ISAAC_EXTENSION_ROOT` as printed; see [provenance](docs/provenance.md).
FFmpeg and ffprobe must be on PATH for browser recording. Fill the paths in `configs/local.env`.

## Prepare and train

The dataset is external. [Data preparation](docs/data.md) describes its required schema,
original success labels, immutable caches and disjoint whole-trajectory split.

For a new raw dataset, edit the user block in `scripts/prepare.sh`, then run:

```bash
bash scripts/prepare.sh
```

Preparation produces a recipe and reports its path. Complete the explicit environment alignment
step described in the data guide. Put the generated recipe and contract paths in the training script.

For the original prepared baseline, provide its frozen selection/cache directory without changing the split.
Edit GPU IDs, paths and budget in `scripts/train.sh`, then run it without arguments:

```bash
bash scripts/train.sh
```

One process runs per selected GPU. The environment count is **per GPU**; the default is 8192.
Every run records source revisions, resolved configuration, physical GPU bindings, metrics and checkpoints.
W&B is optional. A complete checkpoint contains `complete.json`, `policy.pt` and all `rank_*.pt`.
Set `resume_checkpoint` to that directory and increase the update budget to resume with the same card count and recipe.

## Evaluate and view

Set `checkpoint` in `scripts/evaluate.sh` to `policy.pt` (or its directory), retain the training recipe,
and run `bash scripts/evaluate.sh`. Evaluation uses the frozen held-out partition and never optimizes the policy.
The live HTML viewer and saved state replay use actual simulation states. Saved HTML plays at 1x simulation time.
To view a saved run after evaluation exits, edit `case_directory` in `scripts/replay.sh` and run
`bash scripts/replay.sh`. Open the printed local URL; GPU/Isaac is not required for this state replay.
See [deployment](docs/deployment.md) for deterministic inference and action semantics.

Run CPU numerical and contract tests with `python -m pytest`.
The current acceptance results and their limits are recorded in [validation](docs/validation.md).
