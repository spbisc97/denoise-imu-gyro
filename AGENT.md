# denoise-imu-gyro (Localyte fork) — Repo Memory

This file is the canonical local memory for this repository.
Update it when behavior, entrypoints, workflows, scope, or known issues change.

The user has also enabled an AI memory MCP for future use. For now:
- `AGENT.md` is the source of truth for repo-local operational context.
- The memory MCP is reserved for future cross-repo or longer-lived structured memory.

## Scope

This repository is still one project:
- learned IMU gyroscope denoising / calibration
- open-loop attitude estimation
- EuRoC / TUM-VI as the main validated datasets

Current repo decision:
- Do not split into a new repository yet.
- Keep this repo focused on the 2020 paper lineage plus closely related calibration / denoising extensions.
- Create a new repo only if the work expands into a broader inertial odometry platform, multi-sensor fusion framework, or general benchmark suite.

## Current State

As last checked on 2026-04-22:
- current branch: `presentation/little-cheat`
- repo has local user changes outside this file (`.gitignore`, untracked `.codex/`, untracked `.gemini/`)
- repo is substantially ahead of `origin/master`; treat this fork as its own evolving code line, not a thin patch set

## Project Purpose

High-level pipeline:
1. Read raw dataset logs (IMU + ground truth).
2. Build preprocessed pickle files containing:
   - `us`: IMU measurements (`[T, 6]`, gyro + accel)
   - `xs`: target SO(3) orientation increment logs, optionally with a mask
3. Train a model to predict corrected gyro measurements.
4. Integrate corrected gyro to estimate attitude.
5. Export plots and OpenVINS-compatible results.

## Research Positioning

This repo implements and extends:
- M. Brossard, S. Bonnabel, A. Barrau, "Denoising IMU Gyroscopes With Deep Learning for Open-Loop Attitude Estimation," RA-L 2020

Field context from later literature:
- The 2024 survey "Deep Learning for Inertial Positioning" places this work in the sensor-level calibration / denoising branch.
- This is narrower than full learned inertial odometry methods such as AI-IMU Dead-Reckoning, TLIO, and newer hybrid inertial/VIO systems.
- Keep repo goals narrow unless the user explicitly decides to broaden scope.

## Canonical Code Paths

Main files worth trusting first:
- `main_EUROC.py`: main EuRoC entrypoint
- `main_TUMVI.py`: main TUM-VI entrypoint
- `src/entrypoint_utils.py`: shared CLI wiring for EuRoC / TUM-VI
- `src/dataset.py`: dataset parsing and preprocessing
- `src/networks.py`: canonical CNN model plus experimental variants
- `src/losses.py`: multi-rate SO(3) increment loss
- `src/learning.py`: train/test loops, plotting, OpenVINS export, calibrated-IMU baseline

Useful support files:
- `datasets_downloader.py`: dataset download / extract helper
- `environment.yml`: preferred environment definition
- `README.md`: user-facing project description, but currently behind the real repo state

## Validated Entry Points

Smoke-tested successfully on 2026-04-22:
- `main_EUROC.py --mode smoke`
- `main_TUMVI.py --mode smoke`
- `main_KITTI.py --mode smoke`
- `main_with_rnn.py --mode smoke`
- `main_without_acc.py --mode smoke`

Known broken entrypoint:
- `main_with_trn.py` fails at import time because it references `src.networks.GyroNetWithTRN`, which does not exist

Interpretation:
- EuRoC and TUM-VI are the main maintained paths.
- KITTI code exists and smoke-runs, but has not been treated as equally trustworthy.
- RNN and no-accelerometer variants are experimental.

## Model Notes

Canonical model:
- `src.networks.GyroNet`

Important model details:
- dilated causal-ish 1D CNN over IMU sequences
- learned static calibration parameters `gyro_Rot` and `gyro_bias`
- normalization factors injected from dataset statistics

Experimental models:
- `GyroNetWithoutAcc`
- `GyroNetWithRNN`
- `GyroNetWithCNNRNN`

Baseline model:
- `CalibratedIMUNet`
- static correction of the form `(I + dC) * omega + b`
- used as a calibrated-IMU comparison and as optional initialization for `GyroNet`

## Dataset Notes

Primary supported datasets:
- EuRoC
- TUM-VI

Dataset-specific behavior:
- EuRoC stores plain `xs` orientation increments
- TUM-VI stores `xs` plus a validity mask
- KITTI also stores masked targets, but the parser should be treated cautiously until dataset-backed validation is done

Known caution:
- `KITTiDataset` is less battle-tested than EuRoC / TUM-VI
- if working on KITTI, read the parser carefully before trusting results

## Environment

Preferred env:
- `environment.yml`
- conda env name: `denoise-imu-gyro-312`

Current assumptions:
- device-agnostic execution is supported
- CPU execution works, but training will be slower

Common command:
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --mode smoke`

## CLI / Workflow

All main entrypoints support:
- `--mode {auto,train,test,smoke}`
- `--data-dir <path>`
- `--address <last | path>`

EuRoC and TUM-VI also support:
- training window and batch params
- scheduler selection
- sequence overrides
- calibrated-IMU baseline toggles
- calibrated-IMU initialization / freezing
- `--no-show`

Calibrated-IMU workflow lives in `src/learning.py`:
- optional test-time comparison baseline
- optional train-time initialization of learned calibration params

## Outputs

Training outputs:
- `results/<DATASET>/<timestamp>/`
- includes weights, params, yaml dumps, plots, per-sequence outputs

TensorBoard:
- `results/runs/<DATASET>/...`

Export:
- OpenVINS-compatible text files under each run directory

## Known Issues

Known code/documentation issues:
- `main_with_trn.py` is broken
- `README.md` still mostly reflects the original 2020 repo rather than the current fork
- old branch-specific notes were previously stale; keep this file aligned with reality
- there is no real automated test suite yet, only smoke-style execution checks

Non-critical runtime note:
- in sandboxed environments, Matplotlib may warn about a non-writable config dir and fall back to `/tmp`; this is noisy but not a repo bug

## Near-Term Priorities

Recommended maintenance order:
1. Fix or remove `main_with_trn.py`
2. Refresh `README.md` to match the current fork
3. Add minimal automated smoke tests / CI
4. Clarify KITTI support status
5. Keep `AGENT.md` updated after meaningful repo changes

## Memory Policy

When updating memory:
- record scope decisions
- record which entrypoints are trusted or broken
- record dataset assumptions
- record behavior changes, not just code changes
- avoid branch-specific claims unless they are refreshed from `git` at the time of writing

What belongs here:
- repo-local facts needed by future agents
- stable run commands
- known pitfalls
- current strategy decisions

What can move to the AI memory MCP later:
- cross-repo research notes
- longer experiment history
- structured paper comparisons
- user preferences that are not specific to this repo

## Active Planning Docs

Current execution roadmap:
- `docs/optimization-roadmap.md`
