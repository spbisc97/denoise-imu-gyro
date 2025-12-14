# denoise-imu-gyro — English Summary

This repository implements a deep-learning method to **denoise IMU gyroscope measurements** and improve **open-loop attitude (orientation) estimation** by integrating the corrected angular-rate signal. It is based on the paper *“Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude Estimation”* (Brossard et al., 2020) and includes extra utilities and baselines in this Localyte fork.

## What the repo does (end-to-end)

1. **Read a dataset** that provides IMU measurements plus ground-truth orientation (e.g., EuRoC MAV, TUM-VI; this fork also includes KITTI tooling).
2. **Preprocess** the raw logs into “predata” pickle files containing:
   - `us`: IMU measurements (gyro + accel), shape `[T, 6]`
   - `xs`: ground-truth **SO(3) orientation increment logs** (targets for training), shape `[T, 3]` (or `[T, 4]` when a mask is used)
3. **Train** a neural network that outputs **corrected gyro** (a denoised angular-rate sequence).
4. **Integrate** the corrected gyro in open loop to obtain an attitude trajectory, plot results, and optionally export to **OpenVINS** evaluation format.

## Core idea (model + loss)

### Model (gyro denoiser)

The default model (`src/networks.py:GyroNet`) is a **1D CNN with dilated convolutions** over a temporal window of IMU samples. It predicts gyro corrections while also learning a **simple static calibration** (misalignment + bias):

- `gyro_Rot` (a small 3×3 correction matrix)
- `gyro_bias` (3D bias vector)

The network output is used to produce corrected gyro measurements that are then integrated to estimate attitude.

This fork also contains experimental variants:
- `GyroNetWithoutAcc`: ignores accelerometer channels.
- `GyroNetWithRNN` / `GyroNetWithCNNRNN`: sequential alternatives (LSTM-based).
- `CalibratedIMUNet`: a **baseline** that only fits the static calibration `(I + dC)ω + b` by gradient descent (no CNN).

### Loss (SO(3) increment loss, multi-rate)

Training uses `src/losses.py:GyroLoss`, which compares **relative orientation increments** computed from:
- ground truth increments (`xs`) and
- predicted gyro increments integrated over multiple time scales (decimations).

This choice is important because:
- it matches how orientation is actually obtained (integration on SO(3)),
- it is naturally tied to **preintegration / increment** concepts, and
- it supports an **SO(3)-invariant** formulation (robust to global frame rotations).

The implementation uses a **Huber / SmoothL1**-style penalty for robustness.

## How to run (practical)

### 1) Create the conda environment

This repo includes a conda spec at `environment.yml`:

```bash
conda env create -f environment.yml
conda activate denoise-imu-gyro-312
```

### 2) Quick smoke test (no datasets required)

```bash
conda run -n denoise-imu-gyro-312 python main_EUROC.py --mode smoke
```

### 3) Download datasets (optional helper)

This fork provides `datasets_downloader.py` to fetch and extract datasets under `./data/<EUROC|TUMVI|KITTI>/{downloads,dataset}/`.

```bash
conda run -n denoise-imu-gyro-312 python datasets_downloader.py --sources EUROC
```

### 4) Train / test

Train (example, EuRoC):

```bash
conda run -n denoise-imu-gyro-312 python main_EUROC.py --data-dir ./data/EUROC/dataset --mode train
```

Test the last run:

```bash
conda run -n denoise-imu-gyro-312 python main_EUROC.py --data-dir ./data/EUROC/dataset --mode test --address last
```

## Outputs you should expect

- Runs are written under `results/<DATASET>/<timestamp>/`
  - `weights.pt`, `net_params.*`, `train_params.*`
  - plots and per-sequence exports
- Tensorboard logs under `results/runs/<DATASET>/...`
- OpenVINS export text files: `results/<DATASET>/<run>/<sequence>.txt`

## “What comes after” (typical extensions / research directions)

If you want to go beyond open-loop attitude estimation, the most common next steps are:

1. **Close the loop with a state estimator**: integrate the learned IMU correction inside a filter/smoother or factor graph (e.g., VIO/SLAM) instead of open-loop integration.
2. **Model time-varying bias explicitly**: learn bias evolution (instead of static `(I + dC)ω + b`) using sequential models and inject it as a factor/constraint.
3. **Handle domain shift**: robustness across sensors, motion regimes, temperatures, mounting, and sampling rates is often the main blocker in real deployments; domain adaptation and calibration-aware training matter.
4. **Extend to accelerometers** (and full inertial navigation): gyro denoising fixes attitude drift, but full position still needs careful handling of accelerometer errors, gravity alignment, and integration drift.

## References (starting points)

- M. Brossard, S. Bonnabel, A. Barrau, **“Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude Estimation”**, *IEEE RA-L*, 2020. DOI: `10.1109/LRA.2020.3003256`. arXiv: `2002.10718`.
- R. Buchanan, V. Agrawal, M. Camurri, F. Dellaert, M. Fallon, **“Deep IMU Bias Inference for Robust Visual-Inertial Odometry with Factor Graphs”**, arXiv: `2211.04517`, 2022.
- C. Chen, X. Pan, **“Deep Learning for Inertial Positioning: A Survey”**, *IEEE T-ITS*, 2024. DOI: `10.1109/TITS.2024.3381161`.
- F. Zheng, W. Li, C. Ding, X. Cui, **“ADNet: A Neural Network for Accelerometer Signals Denoising”**, *IJCNN*, 2024. DOI: `10.1109/IJCNN60899.2024.10650470`.
- J. D. Jurado, J. F. Raquet, **“A Common Framework for Inertial Sensor Error Modeling”** (Allan variance + noise terms consolidation). Public PDF: `https://silvjurado.com/wp-content/uploads/2024/05/A-Common-Framework-for-Inertial-Sensor-Error-Modeling.pdf`.
- P. Geneva et al., **OpenVINS** (evaluation toolbox used by the original repo): `https://github.com/rpng/open_vins/`.

