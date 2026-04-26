import argparse
import csv
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import src.dataset as ds
from src.lie_algebra import SO3
from src.utils import bmtm, pload


@dataclass(frozen=True)
class DatasetConfig:
    dataset_class: type
    default_data_dir: str
    predata_dir: str
    dt: float
    N: int
    min_train_freq: int
    max_train_freq: int
    default_sequences: tuple[str, ...] = ()


DATASET_CONFIGS = {
    "EUROC": DatasetConfig(
        dataset_class=ds.EUROCDataset,
        default_data_dir=os.path.join(BASE_DIR, "data", "EUROC", "dataset"),
        predata_dir=os.path.join(BASE_DIR, "data", "EUROC"),
        dt=0.005,
        N=32 * 500,
        min_train_freq=16,
        max_train_freq=32,
    ),
    "TUMVI": DatasetConfig(
        dataset_class=ds.TUMVIDataset,
        default_data_dir=os.path.join(BASE_DIR, "data", "TUMVI", "dataset"),
        predata_dir=os.path.join(BASE_DIR, "data", "TUMVI"),
        dt=0.005,
        N=32 * 500,
        min_train_freq=16,
        max_train_freq=32,
    ),
    "BLACKBIRD": DatasetConfig(
        dataset_class=ds.BLACKBIRDDataset,
        default_data_dir=os.path.join(BASE_DIR, "data", "BLACKBIRD", "dataset"),
        predata_dir=os.path.join(BASE_DIR, "data", "BLACKBIRD"),
        dt=0.01,
        N=16 * 500,
        min_train_freq=8,
        max_train_freq=16,
    ),
    "UZHFPV": DatasetConfig(
        dataset_class=ds.UZHFPVDataset,
        default_data_dir=os.path.join(BASE_DIR, "data", "UZHFPV", "dataset"),
        predata_dir=os.path.join(BASE_DIR, "data", "UZHFPV"),
        dt=0.002,
        N=8192,
        min_train_freq=16,
        max_train_freq=32,
        default_sequences=(
            "indoor_forward_3",
            "outdoor_forward_3",
            "outdoor_forward_5",
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create cleaner summary plots for a trained run.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--window-s", type=float, default=5.0, help="Rolling RMS window in seconds.")
    return parser.parse_args()


def discover_sequences(run_dir):
    sequences = []
    for root, _, files in os.walk(run_dir):
        if "results.p" not in files:
            continue
        rel_dir = os.path.relpath(root, run_dir)
        if rel_dir == ".":
            continue
        sequences.append(rel_dir)
    sequences.sort()
    return sequences


def build_dataset(config, data_dir, sequences):
    return config.dataset_class(
        data_dir=data_dir,
        predata_dir=config.predata_dir,
        train_seqs=sequences,
        val_seqs=sequences,
        test_seqs=sequences,
        mode="test",
        N=config.N,
        min_train_freq=config.min_train_freq,
        max_train_freq=config.max_train_freq,
        dt=config.dt,
    )


def as_tensor(value, *, dtype=None):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def integrate_gyro(gt_qs, gyro, dt):
    gt_qs = as_tensor(gt_qs, dtype=torch.float64)
    gyro = as_tensor(gyro, dtype=torch.float64)
    qs = SO3.qnorm(SO3.qexp(gyro * dt))
    rot0 = SO3.qnorm(gt_qs[:2])
    qs[0] = rot0[0]

    n_log2 = np.log2(qs.shape[0])
    for i in range(int(n_log2)):
        k = 2 ** i
        qs[k:] = SO3.qnorm(SO3.qmul(qs[:-k], qs[k:]))
    if int(n_log2) < n_log2:
        k = 2 ** int(n_log2)
        tail = qs[k:].shape[0]
        qs[k:] = SO3.qnorm(SO3.qmul(qs[:tail], qs[k:]))
    return SO3.from_quaternion(qs).float()


def calibration_correct(raw_gyro, calib):
    raw_gyro = as_tensor(raw_gyro)
    dC = as_tensor(calib["dC"], dtype=raw_gyro.dtype)
    b = as_tensor(calib["b"], dtype=raw_gyro.dtype)
    C = torch.eye(3, dtype=raw_gyro.dtype) + dC
    return (raw_gyro @ C.T) + b.view(1, 3)


def rotation_errors(gt_rots, est_rots):
    rel = SO3.normalize(bmtm(est_rots, gt_rots))
    err = 180.0 / np.pi * SO3.log(rel).cpu().numpy()
    rel_q = SO3.qnorm(SO3.to_quaternion(rel))
    angle = 2.0 * torch.atan2(rel_q[:, 1:].norm(dim=1), rel_q[:, 0].abs().clamp(min=1e-12))
    return err, (180.0 / np.pi) * angle.cpu().numpy()


def rolling_rms(values, window):
    window = max(int(window), 1)
    if values.size < window:
        window = values.size
    kernel = np.ones(window, dtype=np.float64) / window
    return np.sqrt(np.convolve(values ** 2, kernel, mode="same"))


def prepare_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def plot_rmse_summary(metrics, out_path, dataset_name):
    labels = [m["sequence"] for m in metrics]
    x = np.arange(len(labels))
    width = 0.25

    raw = [m["raw_rmse_deg"] for m in metrics]
    net = [m["net_rmse_deg"] for m in metrics if m["net_rmse_deg"] is not None]
    cal = [m["cal_rmse_deg"] for m in metrics if m["cal_rmse_deg"] is not None]
    has_net = len(net) == len(metrics)
    has_cal = len(cal) == len(metrics)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6), width_ratios=[1.25, 1.0])
    raw_offset = -width if has_net and has_cal else (-width / 2 if has_net or has_cal else 0)
    ax_full.bar(x + raw_offset, raw, width=width, color="#c44e52", label="Raw IMU")
    if has_net:
        ax_full.bar(x, net, width=width, color="#4c72b0", label="CNN corrected")
    if has_cal:
        cal_offset = width if has_net else width / 2
        ax_full.bar(x + cal_offset, [m["cal_rmse_deg"] for m in metrics], width=width, color="#55a868", label="Static calib")
    ax_full.set_title(f"{dataset_name} orientation RMSE")
    ax_full.set_ylabel("Geodesic RMSE (deg)")
    ax_full.set_xticks(x)
    ax_full.set_xticklabels(labels, rotation=25, ha="right")
    ax_full.legend(frameon=True)
    ax_full.grid(axis="y", alpha=0.35)

    zoom_width = 0.35 if has_net and has_cal else 0.5
    if has_net:
        ax_zoom.bar(x - (zoom_width / 2 if has_cal else 0), net, width=zoom_width, color="#4c72b0", label="CNN corrected")
    if has_cal:
        cal_offset = zoom_width / 2 if has_net else 0
        ax_zoom.bar(x + cal_offset, [m["cal_rmse_deg"] for m in metrics], width=zoom_width, color="#55a868", label="Static calib")
    ax_zoom.set_title("Zoom on corrected methods")
    ax_zoom.set_ylabel("Geodesic RMSE (deg)")
    ax_zoom.set_xticks(x)
    ax_zoom.set_xticklabels(labels, rotation=25, ha="right")
    ax_zoom.grid(axis="y", alpha=0.35)
    ax_zoom.legend(frameon=True)
    zoom_values = net + ([m["cal_rmse_deg"] for m in metrics] if has_cal else [])
    ax_zoom.set_ylim(0.0, max(zoom_values) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_improvement_summary(metrics, out_path):
    labels = [m["sequence"] for m in metrics]
    x = np.arange(len(labels))
    width = 0.32
    net_gain = [m["net_gain_pct"] for m in metrics if m["net_gain_pct"] is not None]
    cal_gain = [m["cal_gain_pct"] for m in metrics]
    has_net = len(net_gain) == len(metrics)
    has_cal = all(value is not None for value in cal_gain)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    if has_net:
        ax.bar(x - (width / 2 if has_cal else 0), net_gain, width=width, color="#4c72b0", label="CNN vs raw")
    if has_cal:
        cal_offset = width / 2 if has_net else 0
        ax.bar(x + cal_offset, cal_gain, width=width, color="#55a868", label="Static calib vs raw")
    ax.set_title("Relative improvement over raw IMU")
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.35)
    values = net_gain + ([v for v in cal_gain if v is not None] if has_cal else [])
    ax.set_ylim(max(0.0, min(values) - 2.0), 100.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hardest_sequence(detail, out_path, window_s, dt):
    t_min = detail["time_min"] - detail["time_min"][0]
    raw = rolling_rms(detail["raw_angle_deg"], int(window_s / dt))
    net = rolling_rms(detail["net_angle_deg"], int(window_s / dt)) if detail["net_angle_deg"] is not None else None
    cal = rolling_rms(detail["cal_angle_deg"], int(window_s / dt)) if detail["cal_angle_deg"] is not None else None

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(t_min, raw, color="#c44e52", linewidth=1.7, label="Raw IMU")
    if net is not None:
        ax.plot(t_min, net, color="#4c72b0", linewidth=2.0, label="CNN corrected")
    if cal is not None:
        ax.plot(t_min, cal, color="#55a868", linewidth=1.8, label="Static calib")
    ax.set_title(f"Rolling orientation error on hardest sequence: {detail['sequence']}")
    ax.set_xlabel("Time since sequence start (min)")
    ax.set_ylabel(f"{window_s:.1f}s rolling RMS error (deg)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correction_summary(metrics, out_path):
    if any(m["net_rmse_deg"] is None for m in metrics):
        return
    labels = [m["sequence"] for m in metrics]
    corr = np.array([[m["corr_rms_x_deg_s"], m["corr_rms_y_deg_s"], m["corr_rms_z_deg_s"]] for m in metrics])
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, corr[:, 0], width=width, color="#4c72b0", label="x")
    ax.bar(x, corr[:, 1], width=width, color="#dd8452", label="y")
    ax.bar(x + width, corr[:, 2], width=width, color="#55a868", label="z")
    ax.set_title("CNN gyro-correction RMS by sequence")
    ax.set_ylabel("Correction RMS (deg/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(title="Axis", frameon=True)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_metrics_csv(metrics, out_path):
    fieldnames = [
        "sequence",
        "raw_rmse_deg",
        "net_rmse_deg",
        "cal_rmse_deg",
        "net_gain_pct",
        "cal_gain_pct",
        "corr_rms_x_deg_s",
        "corr_rms_y_deg_s",
        "corr_rms_z_deg_s",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_md(metrics, detail, out_path):
    has_net = all(row["net_rmse_deg"] is not None for row in metrics)
    best = min(metrics, key=lambda row: row["net_rmse_deg"]) if has_net else None
    hardest = max(metrics, key=lambda row: row["raw_rmse_deg"])
    avg_raw = np.mean([row["raw_rmse_deg"] for row in metrics])
    avg_net = np.mean([row["net_rmse_deg"] for row in metrics]) if has_net else None
    gains = [row["net_gain_pct"] for row in metrics if row["net_gain_pct"] is not None]
    cal_values = [row["cal_rmse_deg"] for row in metrics if row["cal_rmse_deg"] is not None]
    cal_beats_net = (
        sum(
            1
            for row in metrics
            if row["cal_rmse_deg"] is not None and row["cal_rmse_deg"] < row["net_rmse_deg"]
        )
        if has_net
        else 0
    )

    lines = [
        "# Run Summary",
        "",
        f"- Average raw IMU RMSE: `{avg_raw:.3f} deg`",
        f"- Hardest raw sequence: `{hardest['sequence']}` at `{hardest['raw_rmse_deg']:.3f} deg`",
        f"- Hardest-sequence report plot: `{detail['sequence']}`",
        "",
    ]
    if has_net:
        lines.insert(3, f"- Average CNN-corrected RMSE: `{avg_net:.3f} deg`")
        lines.insert(4, f"- Mean CNN RMSE reduction: `{np.mean(gains):.1f}%`")
        lines.insert(5, f"- Best CNN-corrected sequence: `{best['sequence']}` at `{best['net_rmse_deg']:.3f} deg`")
    if cal_values:
        insert_at = 6 if has_net else 3
        lines.insert(insert_at, f"- Average static-calibration RMSE: `{np.mean(cal_values):.3f} deg`")
        lines.insert(insert_at + 1, f"- Mean static-calibration RMSE reduction: `{np.mean([row['cal_gain_pct'] for row in metrics if row['cal_gain_pct'] is not None]):.1f}%`")
        if has_net:
            lines.insert(insert_at + 2, f"- Static calibration beats the CNN on `{cal_beats_net}/{len(metrics)}` test sequences")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    data_dir = args.data_dir or config.default_data_dir
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)

    prepare_style()
    sequences = discover_sequences(run_dir)
    has_net_results = bool(sequences)
    if not sequences:
        sequences = list(config.default_sequences)
    if not sequences:
        raise FileNotFoundError(f"No sequence results found under {run_dir!r}")
    dataset = build_dataset(config, data_dir, sequences)
    calib_path = os.path.join(run_dir, "calibrated_imu.p")
    calib = pload(calib_path) if os.path.isfile(calib_path) else None

    metrics = []
    details = []
    for index, sequence in enumerate(dataset.sequences):
        raw_us, _ = dataset[index]
        gt = dataset.load_gt(index)
        net_path = os.path.join(run_dir, sequence, "results.p")
        net_us = pload(net_path)["hat_xs"] if os.path.isfile(net_path) else None

        n = net_us.shape[0] if net_us is not None else min(raw_us.shape[0], gt["qs"].shape[0])
        gt_qs = as_tensor(gt["qs"][:n])
        gt_rots = SO3.from_quaternion(gt_qs).float()
        raw_rots = integrate_gyro(gt_qs, raw_us[:n, :3], config.dt)

        raw_err_vec, raw_angle = rotation_errors(gt_rots, raw_rots)
        net_err_vec = None
        net_angle = None
        net_rmse = None
        net_gain = None
        corr_rms = [None, None, None]
        if net_us is not None:
            net_rots = integrate_gyro(gt_qs, net_us[:n, :3], config.dt)
            net_err_vec, net_angle = rotation_errors(gt_rots, net_rots)
            net_rmse = float(np.sqrt(np.mean(net_angle ** 2)))
            corr = as_numpy(as_tensor(raw_us[:n, :3]) - as_tensor(net_us[:n, :3])) * (180.0 / np.pi)
            corr_rms = np.sqrt(np.mean(corr ** 2, axis=0))

        cal_angle = None
        cal_rmse = None
        cal_gain = None
        if calib is not None:
            cal_us = calibration_correct(raw_us[:n, :3], calib)
            cal_rots = integrate_gyro(gt_qs, cal_us, config.dt)
            _, cal_angle = rotation_errors(gt_rots, cal_rots)
            cal_rmse = float(np.sqrt(np.mean(cal_angle ** 2)))
            cal_gain = 100.0 * (1.0 - (cal_rmse / float(np.sqrt(np.mean(raw_angle ** 2)))))

        raw_rmse = float(np.sqrt(np.mean(raw_angle ** 2)))
        if net_rmse is not None:
            net_gain = 100.0 * (1.0 - (net_rmse / raw_rmse))

        metrics.append(
            {
                "sequence": sequence,
                "raw_rmse_deg": raw_rmse,
                "net_rmse_deg": net_rmse,
                "cal_rmse_deg": cal_rmse,
                "net_gain_pct": net_gain,
                "cal_gain_pct": cal_gain,
                "corr_rms_x_deg_s": None if corr_rms[0] is None else float(corr_rms[0]),
                "corr_rms_y_deg_s": None if corr_rms[1] is None else float(corr_rms[1]),
                "corr_rms_z_deg_s": None if corr_rms[2] is None else float(corr_rms[2]),
            }
        )
        details.append(
            {
                "sequence": sequence,
                "time_min": as_numpy(gt["ts"][:n]) / 60.0,
                "raw_angle_deg": raw_angle,
                "net_angle_deg": net_angle,
                "cal_angle_deg": cal_angle,
                "raw_err_vec_deg": raw_err_vec,
                "net_err_vec_deg": net_err_vec,
            }
        )

    hardest = max(metrics, key=lambda row: row["raw_rmse_deg"])["sequence"]
    hardest_detail = next(item for item in details if item["sequence"] == hardest)

    plot_rmse_summary(metrics, os.path.join(out_dir, "orientation_rmse_summary.png"), args.dataset)
    plot_improvement_summary(metrics, os.path.join(out_dir, "orientation_improvement_summary.png"))
    plot_hardest_sequence(
        hardest_detail,
        os.path.join(out_dir, "hardest_sequence_rolling_error.png"),
        args.window_s,
        config.dt,
    )
    plot_correction_summary(metrics, os.path.join(out_dir, "gyro_correction_rms_summary.png"))
    write_metrics_csv(metrics, os.path.join(out_dir, "metrics_summary.csv"))
    write_summary_md(metrics, hardest_detail, os.path.join(out_dir, "summary.md"))

    print(f"report_ok {out_dir}")
    for row in metrics:
        if has_net_results:
            print(
                "sequence={sequence} raw_rmse_deg={raw_rmse_deg:.3f} "
                "net_rmse_deg={net_rmse_deg:.3f} net_gain_pct={net_gain_pct:.1f}".format(**row)
            )
        else:
            print(
                "sequence={sequence} raw_rmse_deg={raw_rmse_deg:.3f} "
                "cal_rmse_deg={cal_rmse_deg:.3f} cal_gain_pct={cal_gain_pct:.1f}".format(**row)
            )


if __name__ == "__main__":
    main()
