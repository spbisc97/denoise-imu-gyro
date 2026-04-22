# Optimization Roadmap

This document captures the current optimization plan for the Localyte fork of
`denoise-imu-gyro`.

Date drafted: 2026-04-22
Planning branch: `optimization-roadmap`

## Goals

Primary goals:
- make the maintained paths reliable
- remove or isolate broken experimental code
- improve training correctness before chasing speed
- add enough automation that regressions are visible quickly
- keep the repo focused on gyro denoising / attitude estimation, not a general inertial platform

Non-goals for this phase:
- large architectural rewrite
- full multi-sensor fusion stack
- broad repo split
- premature optimization of code that is not yet trusted

## Priority Order

1. Correctness and trustworthiness
2. Reproducibility and testability
3. Training and preprocessing efficiency
4. Research-facing model ablations
5. Optional speedups and cleanup

## Workstreams

## 1. Stabilize Maintained Paths

Objective:
- define the supported surface area and make it consistently runnable

Tasks:
- fix or remove `main_with_trn.py`
- mark `main_with_rnn.py` and `main_without_acc.py` as experimental in docs
- refresh `README.md` to match the current fork instead of the original 2020 repo state
- add a support matrix for EuRoC, TUM-VI, KITTI, and experimental variants

Success criteria:
- every documented entrypoint either works or is explicitly labeled experimental/broken
- no import-time failures remain in advertised scripts

## 2. Fix Dataset / Training Correctness

Objective:
- remove hidden issues that distort training or first-run behavior

High-priority findings from repo inspection:
- `BaseDataset` computes normalization factors before subclass `read_data()` runs
- on a first run with no preprocessed data, normalization falls back to `0, 0`
- training window sampling appears too narrow: random start is drawn only from `[0, max_train_freq)`, which underuses each sequence

Tasks:
- reorder dataset initialization so preprocessing happens before normalization-factor loading
- ensure normalization factors are recomputed after predata generation when needed
- redesign train-window sampling to cover the full usable sequence length
- audit validation/test slicing so comparisons remain aligned after train sampling changes
- audit KITTI parser assumptions before treating KITTI as supported

Success criteria:
- first run from raw data produces valid normalization stats automatically
- training draws windows across the sequence rather than repeatedly near the start
- EuRoC and TUM-VI training behavior is deterministic and explainable

## 3. Add Minimal Automated Verification

Objective:
- catch regressions quickly without needing full datasets

Tasks:
- add smoke tests for maintained entrypoints
- add one import-level test for experimental entrypoints to catch missing symbols
- run smoke tests in CI on CPU
- include a lightweight lint or static check pass if it stays low-friction

Suggested first test set:
- `main_EUROC.py --mode smoke`
- `main_TUMVI.py --mode smoke`
- `main_KITTI.py --mode smoke`
- `main_with_rnn.py --mode smoke`
- `main_without_acc.py --mode smoke`

Success criteria:
- broken scripts fail in CI before merge
- repo can be validated on a fresh machine without datasets

## 4. Improve Training Workflow

Objective:
- make experiments easier to run, compare, and reproduce

Tasks:
- add seed control for Python, NumPy, and Torch
- log the effective config in a single canonical place per run
- standardize run naming for dataset, model, scheduler, and calibration mode
- expose train `shuffle` behavior explicitly and set sane defaults
- capture baseline vs learned-calibration comparisons in a structured summary file

Success criteria:
- repeated experiments are traceable
- config drift between runs is easy to detect

## 5. Efficiency Pass

Objective:
- improve runtime only after correctness is in place

Tasks:
- profile preprocessing time and identify avoidable repeated work
- avoid redundant dataset scans where possible
- consider optional mixed precision for training if numerically stable
- make DataLoader settings conditional on environment instead of one-size-fits-all
- reduce plotting overhead for non-interactive test runs

Success criteria:
- preprocessing and smoke/test runs are faster without changing results
- any training speedup is measured, not assumed

## 6. Research Cleanup

Objective:
- separate trustworthy baselines from speculative branches

Tasks:
- compare canonical CNN against calibrated-IMU baseline after correctness fixes
- decide whether `GyroNetWithCNNRNN` is worth keeping
- either implement a real transformer/TRN path or delete the dead entrypoint
- document which model classes are canonical, experimental, or deprecated

Success criteria:
- model inventory is small and understandable
- dead branches stop consuming maintenance time

## Proposed Execution Phases

## Phase 1: Repo Trust

Deliverables:
- fixed `main_with_trn.py` status
- refreshed README
- support matrix
- smoke-test automation scaffold

## Phase 2: Data and Training Correctness

Deliverables:
- dataset init fix
- normalization-factor fix
- train-window sampling fix
- validation of EuRoC / TUM-VI behavior

## Phase 3: Experiment Discipline

Deliverables:
- seed control
- better run metadata
- clearer baseline summaries

## Phase 4: Measured Speedups

Deliverables:
- profiling notes
- targeted runtime improvements
- optional mixed precision or loader tuning if justified

## Immediate Next Actions

Recommended next implementation order:
1. fix `main_with_trn.py` or remove it from supported paths
2. fix dataset first-run normalization behavior
3. fix training window sampling
4. add smoke tests
5. refresh `README.md`

## Decision Log

Current strategic decisions:
- keep this repository as the focused gyro-denoising / attitude-estimation repo
- do not split to a new repo yet
- prefer correctness fixes over model churn
- use `AGENT.md` as the local repo memory and treat this roadmap as an execution document
