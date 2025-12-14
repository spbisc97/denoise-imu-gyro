# denoise-imu-gyro (Localyte fork) — Agent Notes

This file is meant to help future coding agents quickly understand and work on this repository.
Keep this file updated when you change behavior, entrypoints, or workflows.

## What This Repo Does

Deep-learning-based denoising of IMU gyroscope signals for open-loop attitude estimation.

High-level pipeline:
1. Read raw dataset logs (IMU + ground truth).
2. Build preprocessed “predata” pickles containing:
   - `us`: IMU measurements (gyro+acc, shape `[T, 6]`)
   - `xs`: target SO(3) orientation increment logs (and optional mask)
3. Train a network to predict corrected gyro measurements.
4. Integrate predicted gyro to estimate attitude; export/plot results.

## Important Branches

This working tree is currently on the `calibration` branch (tracking `origin/calibration`), which is the most-modified branch vs `master`.

Notable additions on `calibration`:
- `datasets_downloader.py`: download/extract EuRoC / TUM-VI / KITTI (requires network access).
- Extra entrypoints: `main_KITTI.py`, `main_with_rnn.py`, `main_with_trn.py`, `main_without_acc.py`.
- Extra model variants in `src/networks.py`.
- `KITTiDataset` in `src/dataset.py`.
- Additional dependencies: `tensorboard`, `tqdm`, `requests`, `scipy`.

## Repo Layout

- `main_EUROC.py`, `main_TUMVI.py`, `main_KITTI.py`: dataset entrypoints (train/test/smoke).
- `src/dataset.py`: dataset parsing + preprocessing into pickles; provides PyTorch `Dataset`.
- `src/networks.py`: CNN-based `GyroNet` + experimental variants.
- `src/losses.py`: `GyroLoss` using multi-rate SO(3) increment errors.
- `src/learning.py`: training loop, evaluation, plotting, OpenVINS export.
- `src/lie_algebra.py`: SO(3) / quaternion utilities.
- `src/utils.py`: (pickle/yaml) IO helpers + batched linear algebra helpers.
- `metrics.py`: tiny sanity check for SO(3) vs quaternion angle.

## Environment / Dependencies

Primary env definition:
- `environment.yml` (conda, name includes Python suffix: `denoise-imu-gyro-312`)

Runtime notes:
- This codebase originally assumed CUDA everywhere; it has been updated to be device-agnostic.
- If you have no GPU, it should run on CPU (slower).

## How To Run

All main entrypoints support:
- `--mode {auto,train,test,smoke}`
- `--data-dir <path>` (raw dataset root)
- `--address <'last' | path>` (which run/weights to test)

Examples (no dataset required):
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --mode smoke`
- `conda run -n denoise-imu-gyro-312 python main_TUMVI.py --mode smoke`

With datasets present:
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --data-dir ./data/EUROC/dataset --mode train`
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --data-dir ./data/EUROC/dataset --mode test --address last`
- `conda run -n denoise-imu-gyro-312 python main_EUROC.py --mode test --no-calib-baseline` (disables the calibrated-IMU baseline)
- `conda run -n denoise-imu-gyro-312 python main_TUMVI.py --data-dir ./data/TUMVI/dataset --mode train`
- `conda run -n denoise-imu-gyro-312 python main_TUMVI.py --data-dir ./data/TUMVI/dataset --mode test --address last`
- `conda run -n denoise-imu-gyro-312 python main_TUMVI.py --mode test --no-calib-baseline` (disables the calibrated-IMU baseline)

Downloading datasets (network required):
- `conda run -n denoise-imu-gyro-312 python datasets_downloader.py --sources TUMVI`
  - Writes under `./data/<EUROC|TUMVI|KITTI>/{downloads,dataset}/`
  - If TUM-VI hosting returns TLS hostname mismatch, retry with `--insecure` (disables TLS verification).

## Outputs

- Training runs: `results/<DATASET>/<timestamp>/`
  - `weights.pt`, `net_params.*`, `train_params.*`, plots per sequence
- Tensorboard logs: `results/runs/<DATASET>/...`
- OpenVINS export: `results/<DATASET>/<run>/<sequence>.txt`

## Implementation Notes / Gotchas

- YAML dumping: training config contains class objects; `src/utils.py` sanitizes these to strings for YAML output.
- Paths: `src/dataset.py` creates `predata_dir` automatically; `src/learning.py` creates `res_dir` and `tb_dir`.
- Loss stability: `GyroLoss` can produce NaNs on arbitrary random inputs; smoke tests use small-magnitude synthetic inputs.
- Calibrated-IMU baseline: implemented as a static correction `(I + dC)ω + b` optimized by gradient descent (paper “calibrated IMU” comparison); enabled by default in `main_EUROC.py` and `main_TUMVI.py` (use `--no-calib-baseline` to disable).
- Two-stage training option: you can fit the calibrated-IMU baseline first and use it to initialize the model calibration params via `main_EUROC.py --calib-init` or `main_TUMVI.py --calib-init` (optionally `--calib-freeze`).

## Docs

- Paper reference: `docs/PaperReference/2002.10718v2_IMU_denoise.pdf`

## What Was Changed Recently (Keep Updated)

Recent agent work (Dec 2025):
- Device-agnostic execution (no hard `.cuda()` calls) across core modules.
- Safer YAML load/dump via `yaml.safe_*` and YAML sanitization for non-serializable objects.
- CLI-style `--mode/--data-dir/--address` support in all `main_*.py`.
- Added `environment.yml` (`denoise-imu-gyro-312`).
- Added `datasets_downloader.py` flags (`--sources`, retries/timeouts, optional `--insecure`).
- Added EuRoC/TUM-VI “calibrated IMU” GD baseline support (static correction) for comparison plots.
