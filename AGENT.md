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
- Blackbird as the newly integrated aggressive-UAV dataset path

Current repo decision:
- Do not split into a new repository yet.
- Keep this repo focused on the 2020 paper lineage plus closely related calibration / denoising extensions.
- Create a new repo only if the work expands into a broader inertial odometry platform, multi-sensor fusion framework, or general benchmark suite.
- Primary technical objective: maximize gyro correction precision with CNN-based models on the maintained datasets.
- Deprioritize non-CNN model exploration unless the canonical CNN path is clearly saturated.

## Current State

As last checked on 2026-04-22:
- current branch: `optimization-roadmap`
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

## Current Research Priority

Primary interest:
- really precise gyro correction from CNNs

Interpretation for future work:
- optimize the canonical CNN path before spending time on RNN, CNN-RNN, transformer, or broader odometry work
- judge progress by corrected gyro quality and downstream attitude accuracy, not by model novelty alone
- prefer fixes that improve training signal quality, calibration quality, receptive-field usefulness, and evaluation discipline

## Canonical Code Paths

Main files worth trusting first:
- `main_EUROC.py`: main EuRoC entrypoint
- `main_TUMVI.py`: main TUM-VI entrypoint
- `main_BLACKBIRD.py`: Blackbird entrypoint (official CSV-export layout)
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
- `main_BLACKBIRD.py --mode smoke`

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

Model strategy:
- `GyroNet` is the main line
- experimental variants should not outrank CNN precision work unless they show clear gains in controlled comparisons

Baseline model:
- `CalibratedIMUNet`
- static correction of the form `(I + dC) * omega + b`
- used as a calibrated-IMU comparison and as optional initialization for `GyroNet`

## Dataset Notes

Primary supported datasets:
- EuRoC
- TUM-VI
- Blackbird

Dataset-specific behavior:
- EuRoC stores plain `xs` orientation increments
- EuRoC downloads are now hosted by ETH Research Collection as three large category ZIPs (`machine_hall.zip`, `vicon_room1.zip`, `vicon_room2.zip`) containing nested per-sequence ZIPs; `datasets_downloader.py --sources EUROC` handles this new layout and extracts into `data/EUROC/dataset/<sequence>/`
- TUM-VI stores `xs` plus a validity mask
- Blackbird uses official CSV exports under `<trajectory>/<yawMode>/<speed>/csv/`
- Blackbird ground truth is converted from `body_frame` to the IMU frame using the static body-to-IMU rotation from the official upstream conversion utilities
- KITTI also stores masked targets, but the parser should be treated cautiously until dataset-backed validation is done

Known caution:
- Blackbird integration is based on the official repo structure and message-conversion utilities, but it has not yet been dataset-backed validated locally because the official host was not reachable from this environment
- `KITTiDataset` is less battle-tested than EuRoC / TUM-VI
- if working on KITTI, read the parser carefully before trusting results

## Blackbird Integration Provenance

Upstream sources used for the Blackbird integration:
- official repo README: dataset naming, sequence organization, and general usage
- `fileTreeUtilities/sequenceDownloader.py`: exact folder layout and expected downloaded files per sequence
- `logConversionUtilities/bagToCsv.py`: official CSV export names and column ordering assumptions
- `logConversionUtilities/msgConverters.py`: topic mapping and the static `body_frame -> imu` rotation used to convert ground-truth attitudes into the IMU frame
- `ros_utilities/blackbird_dataset/launch/playback_sequence.launch`: confirms the per-sequence path convention rooted at `<datasetDir>/<flight>/`
- IJRR dataset paper: dataset characteristics and sensor rates

Implementation assumptions derived from those sources:
- raw data root for a sequence is `<data_dir>/<trajectory>/<yawMode>/<speed>/`
- preferred parser inputs are `csv/blackbird_slash_imu.csv` and `csv/blackbird_slash_state.csv`
- fallback ground-truth input is `groundTruthPoses.csv` if the state CSV is missing
- default Blackbird timing is `dt = 0.01` because the dataset IMU is `100 Hz`
- default `min_train_freq = 8` and `max_train_freq = 16` were chosen to keep the orientation-increment horizon roughly comparable to the existing `200 Hz` EuRoC / TUM-VI setup
- nested predata paths are required because Blackbird sequence names contain `/`

## Blackbird Validation Status

What has been validated locally:
- `conda run -n denoise-imu-gyro-312 python main_BLACKBIRD.py --mode smoke`
- parser path exercised against a synthetic on-disk fixture matching the official Blackbird folder and CSV layout
- shared preprocessing flow now computes normalization after dataset preprocessing and supports nested sequence names in cached predata paths

What remains unvalidated here:
- no end-to-end run against hosted real Blackbird files was possible from this environment
- `blackbird-dataset.mit.edu` was not reachable during integration work
- CSV column assumptions are source-aligned, but still deserve one real-sequence train/test pass before treating Blackbird results as fully trusted

## Blackbird Source Links

Primary references:
- `https://github.com/mit-aera/Blackbird-Dataset`
- `https://github.com/mit-aera/Blackbird-Dataset/blob/master/fileTreeUtilities/sequenceDownloader.py`
- `https://github.com/mit-aera/Blackbird-Dataset/blob/master/logConversionUtilities/bagToCsv.py`
- `https://github.com/mit-aera/Blackbird-Dataset/blob/master/logConversionUtilities/msgConverters.py`
- `https://github.com/mit-aera/Blackbird-Dataset/blob/master/ros_utilities/blackbird_dataset/launch/playback_sequence.launch`
- `https://doi.org/10.1177/0278364920908331`

## Environment

Preferred env:
- `environment.yml`
- conda env name: `denoise-imu-gyro-312`

Current assumptions:
- device-agnostic execution is supported
- CPU execution works, but training will be slower

Common command:
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --mode smoke`
- `conda run -n denoise-imu-gyro-312 python main_BLACKBIRD.py --mode smoke`

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
- calibrated-IMU initialization / freezing; by default training fits a static
  gyro calibration `(I + dC) * gyro + b` on the train split once, stores it
  in `data/static_calibrations/<DATASET>/gyro.yaml`, initializes `gyro_Rot` and
  `gyro_bias`, and freezes them so the CNN learns residuals
- `--no-show`

Calibrated-IMU workflow lives in `src/learning.py`:
- optional test-time comparison baseline
- default train-time initialization of learned calibration params; use
  `--no-calib-init` or `--no-calib-freeze` for ablations
- `--mode calibrate` recomputes and saves the persistent static calibration
- `--calib-source {static,fit,none}` selects persistent YAML, run-local fit,
  or no static initialization

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

Research optimization order:
1. improve dataset / target correctness for the CNN path
2. improve CNN training coverage and reproducibility
3. compare CNN against the calibrated-IMU baseline cleanly
4. only then consider new architectures

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
