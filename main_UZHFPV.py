import os
import argparse
import sys
import torch
# torch.backends.cudnn.enabled = False
import src.learning as lr
import src.entrypoint_utils as ep
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = "./data/UZHFPV/dataset"
address = "last"

net_class = sn.GyroNet
net_params = {
    "in_dim": 6,
    "out_dim": 3,
    "c0": 16,
    "dropout": 0.1,
    "ks": [7, 7, 7, 7],
    "ds": [4, 4, 4],
    "momentum": 0.1,
    "gyro_std": [1 * np.pi / 180, 2 * np.pi / 180, 5 * np.pi / 180],
}

dataset_class = ds.UZHFPVDataset
dataset_params = {
    "data_dir": data_dir,
    "predata_dir": os.path.join(base_dir, "data/UZHFPV"),
    "train_seqs": [
        "indoor_forward_3",
        "outdoor_forward_3",
    ],
    "val_seqs": [
        "outdoor_forward_5",
    ],
    "test_seqs": [
        "indoor_forward_3",
        "outdoor_forward_3",
        "outdoor_forward_5",
    ],
    "N": 32 * 500,
    "min_train_freq": 16,
    "max_train_freq": 32,
    "train_windows_per_seq": 16,
    "dt": 0.002,
}

train_params = {
    "optimizer_class": torch.optim.Adam,
    "optimizer": {
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "calib_lr_scale": 0.25,
        "calib_weight_decay": 0.0,
        "norm_weight_decay": 0.0,
        "amsgrad": False,
    },
    "loss_class": sl.GyroLoss,
    "loss": {
        "min_N": int(np.log2(dataset_params["min_train_freq"])),
        "max_N": int(np.log2(dataset_params["max_train_freq"])),
        "w": 1e6,
        "target": "rotation matrix",
        "huber": 0.005,
        "dt": dataset_params["dt"],
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler": {
        "T_max": 1800,
        "eta_min": 1e-3,
    },
    "dataloader": {
        "batch_size": 10,
        "pin_memory": False,
        "num_workers": 0,
        "shuffle": True,
    },
    "freq_val": 600,
    "n_epochs": 1800,
    "res_dir": os.path.join(base_dir, "results/UZHFPV"),
    "tb_dir": os.path.join(base_dir, "results/runs/UZHFPV"),
}


def _smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net_class(**net_params).to(device)
    net.set_normalized_factors(torch.zeros(net_params["in_dim"]), torch.ones(net_params["in_dim"]))
    us = (1e-3 * torch.randn(2, 256, net_params["in_dim"], device=device)).float()
    hat_xs = net(us)
    Loss = train_params["loss_class"]
    criterion = Loss(**train_params["loss"]).to(device)
    xs = (1e-3 * torch.randn(2, 256, 3, device=device)).float()
    loss = criterion(xs, hat_xs)
    print("smoke_ok", hat_xs.shape, float(loss.detach().cpu()), "device:", device)


def main():
    global data_dir, address

    parser = argparse.ArgumentParser()
    ep.add_common_entrypoint_args(
        parser,
        dataset_params=dataset_params,
        train_params=train_params,
        default_data_dir=data_dir,
        default_address=address,
        dataset_label="UZHFPV",
        default_static_calib_path=os.path.join(base_dir, "data/static_calibrations/UZHFPV/gyro.yaml"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    address = args.address
    ep.apply_common_entrypoint_overrides(args, dataset_params=dataset_params, train_params=train_params)

    if args.mode == "smoke":
        _smoke_test()
        return 0

    if not os.path.isdir(data_dir):
        print(f"Missing dataset directory: {data_dir!r}", file=sys.stderr)
        print("Run: python datasets_downloader.py --sources UZHFPV --max-workers 2", file=sys.stderr)
        return 2

    res_dir = train_params["res_dir"]
    has_runs = os.path.isdir(res_dir) and len(os.listdir(res_dir)) > 0
    mode = args.mode
    if mode == "auto":
        mode = "test" if has_runs else "train"

    if mode == "train":
        learning_process = lr.GyroLearningBasedProcessing(
            train_params["res_dir"],
            train_params["tb_dir"],
            net_class,
            net_params,
            address=None,
            dt=train_params["loss"]["dt"],
        )
        ep.apply_calib_args_for_train(learning_process, args)
        if args.calib_source == "none":
            learning_process.init_from_calib_baseline = False
        learning_process.train(dataset_class, dataset_params, train_params)
        return 0

    if mode == "calibrate":
        learning_process = lr.GyroLearningBasedProcessing(
            train_params["res_dir"],
            train_params["tb_dir"],
            net_class,
            net_params,
            address=None,
            dt=train_params["loss"]["dt"],
        )
        ep.apply_calib_args_for_train(learning_process, args)
        learning_process.compute_static_calibration(dataset_class, dataset_params, train_params)
        return 0

    learning_process = lr.GyroLearningBasedProcessing(
        train_params["res_dir"],
        train_params["tb_dir"],
        net_class,
        net_params,
        address=address,
        dt=train_params["loss"]["dt"],
    )
    ep.apply_calib_args_for_test(learning_process, args)
    learning_process.test(dataset_class, dataset_params, ["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
