from __future__ import annotations

import argparse
from typing import Any, MutableMapping


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
    dataset_params["train_seqs"] = [s for s in args.train_seqs.split(",") if s]
    dataset_params["val_seqs"] = [s for s in args.val_seqs.split(",") if s]
    dataset_params["test_seqs"] = [s for s in args.test_seqs.split(",") if s]

    train_params["n_epochs"] = int(args.epochs)
    train_params["freq_val"] = int(args.freq_val)
    train_params["dataloader"]["batch_size"] = int(args.batch_size)


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

