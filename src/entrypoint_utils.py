from __future__ import annotations

import argparse
from typing import Any, MutableMapping

import torch


def add_common_entrypoint_args(
    parser: argparse.ArgumentParser,
    *,
    dataset_params: MutableMapping[str, Any],
    train_params: MutableMapping[str, Any],
    default_data_dir: str,
    default_address: str,
    dataset_label: str,
) -> None:
    parser.add_argument("--mode", choices=["auto", "train", "test", "smoke"], default="auto")
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--address", default=default_address, help="Weights/run to test: 'last' or a path")

    parser.add_argument(
        "--calib-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include 'calibrated IMU' (static GD) baseline on test runs.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call matplotlib's plt.show() (useful for non-interactive runs).",
    )

    parser.add_argument("--epochs", type=int, default=int(train_params["n_epochs"]))
    parser.add_argument("--freq-val", type=int, default=int(train_params["freq_val"]))
    parser.add_argument("--batch-size", type=int, default=int(train_params["dataloader"]["batch_size"]))
    parser.add_argument("--N", type=int, default=int(dataset_params["N"]), help="Training window length (samples)")
    parser.add_argument(
        "--train-windows-per-seq",
        type=int,
        default=int(dataset_params.get("train_windows_per_seq", 16)),
        help="Number of random windows to draw per training sequence in each epoch.",
    )
    parser.add_argument("--lr", type=float, default=float(train_params["optimizer"]["lr"]))
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(train_params["optimizer"].get("weight_decay", 0.0)),
    )
    parser.add_argument(
        "--calib-lr-scale",
        type=float,
        default=float(train_params["optimizer"].get("calib_lr_scale", 1.0)),
    )
    parser.add_argument(
        "--calib-weight-decay",
        type=float,
        default=float(train_params["optimizer"].get("calib_weight_decay", 0.0)),
    )
    parser.add_argument(
        "--norm-weight-decay",
        type=float,
        default=float(train_params["optimizer"].get("norm_weight_decay", 0.0)),
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=bool(train_params["dataloader"].get("shuffle", True)),
        help="Shuffle training windows each epoch.",
    )

    scheduler_defaults = dict(train_params.get("scheduler", {}))
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "warm_restarts"],
        default="cosine",
        help="LR scheduler to use (default: cosine).",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=float(scheduler_defaults.get("eta_min", 1e-3)),
        help="Scheduler minimum LR (eta_min).",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="CosineAnnealingLR: T_max (defaults to --epochs).",
    )
    parser.add_argument(
        "--t0",
        type=int,
        default=int(scheduler_defaults.get("T_0", 600)),
        help="CosineAnnealingWarmRestarts: T_0 (initial period).",
    )
    parser.add_argument(
        "--t-mult",
        type=int,
        default=int(scheduler_defaults.get("T_mult", 2)),
        help="CosineAnnealingWarmRestarts: T_mult (period multiplier).",
    )

    parser.add_argument(
        "--train-seqs",
        default=",".join(dataset_params["train_seqs"]),
        help=f"Comma-separated {dataset_label} train sequences",
    )
    parser.add_argument(
        "--val-seqs",
        default=",".join(dataset_params["val_seqs"]),
        help=f"Comma-separated {dataset_label} val sequences",
    )
    parser.add_argument(
        "--test-seqs",
        default=",".join(dataset_params["test_seqs"]),
        help=f"Comma-separated {dataset_label} test sequences",
    )

    parser.add_argument("--calib-steps", type=int, default=300, help="Steps for calibrated-IMU GD baseline")
    parser.add_argument("--calib-lr", type=float, default=5e-2, help="LR for calibrated-IMU GD baseline")
    parser.add_argument(
        "--calib-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before training, fit the calibrated-IMU baseline and use it to initialize the model calibration params."
        ),
    )
    parser.add_argument(
        "--calib-freeze",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze calibration params (gyro_Rot/bias) after calib-init during training.",
    )


def apply_common_entrypoint_overrides(
    args: argparse.Namespace,
    *,
    dataset_params: MutableMapping[str, Any],
    train_params: MutableMapping[str, Any],
) -> None:
    dataset_params["data_dir"] = args.data_dir
    dataset_params["N"] = int(args.N)
    dataset_params["train_windows_per_seq"] = int(args.train_windows_per_seq)
    dataset_params["train_seqs"] = [s for s in args.train_seqs.split(",") if s]
    dataset_params["val_seqs"] = [s for s in args.val_seqs.split(",") if s]
    dataset_params["test_seqs"] = [s for s in args.test_seqs.split(",") if s]

    train_params["n_epochs"] = int(args.epochs)
    train_params["freq_val"] = int(args.freq_val)
    train_params["dataloader"]["batch_size"] = int(args.batch_size)
    train_params["dataloader"]["shuffle"] = bool(args.shuffle)
    train_params["optimizer"]["lr"] = float(args.lr)
    train_params["optimizer"]["weight_decay"] = float(args.weight_decay)
    train_params["optimizer"]["calib_lr_scale"] = float(args.calib_lr_scale)
    train_params["optimizer"]["calib_weight_decay"] = float(args.calib_weight_decay)
    train_params["optimizer"]["norm_weight_decay"] = float(args.norm_weight_decay)

    eta_min = float(args.eta_min)
    if args.scheduler == "cosine":
        train_params["scheduler_class"] = torch.optim.lr_scheduler.CosineAnnealingLR
        t_max = int(args.t_max) if args.t_max is not None else int(train_params["n_epochs"])
        train_params["scheduler"] = {
            "T_max": t_max,
            "eta_min": eta_min,
        }
    else:
        train_params["scheduler_class"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        train_params["scheduler"] = {
            "T_0": int(args.t0),
            "T_mult": int(args.t_mult),
            "eta_min": eta_min,
        }


def apply_calib_args_for_train(process: Any, args: argparse.Namespace) -> None:
    process.init_from_calib_baseline = bool(args.calib_init)
    process.freeze_calib_params = bool(args.calib_freeze)
    process.calib_baseline_steps = int(args.calib_steps)
    process.calib_baseline_lr = float(args.calib_lr)


def apply_calib_args_for_test(process: Any, args: argparse.Namespace) -> None:
    process.enable_calibrated_imu_baseline = bool(args.calib_baseline)
    process.calib_baseline_steps = int(args.calib_steps)
    process.calib_baseline_lr = float(args.calib_lr)
    process.show_plots = not bool(args.no_show)
