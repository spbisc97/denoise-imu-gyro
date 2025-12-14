import os
import argparse
import sys
import torch
import src.learning as lr
import src.entrypoint_utils as ep
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = './data/TUMVI/dataset'
# test a given network
# address = os.path.join(base_dir, 'results/TUMVI/2020_02_18_16_26_33')
# or test the last trained network
address = 'last'
################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [0.2*np.pi/180, 0.2*np.pi/180, 0.2*np.pi/180],
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.TUMVIDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': os.path.join(base_dir, 'data/TUMVI'),
    # set train, val and test sequence
    'train_seqs': [
        'dataset-room1_512_16',
        'dataset-room3_512_16',
        'dataset-room5_512_16',
        ],
    'val_seqs': [
        'dataset-room2_512_16',
        'dataset-room4_512_16',
        'dataset-room6_512_16',
        ],
    'test_seqs': [
        'dataset-room2_512_16',
        'dataset-room4_512_16',
        'dataset-room6_512_16'
        ],
    # size of trajectory during training
    'N': 32 * 500, # should be integer * 'max_train_freq'
    'min_train_freq': 16,
    'max_train_freq': 32,
}
################################################################################
# Training parameters
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])),
        'max_N': int(np.log2(dataset_params['max_train_freq'])),
        'w':  1e6,
        'target': 'rotation matrix mask',
        'huber': 0.005,
        'dt': 0.005,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingLR,
    'scheduler': {
        'T_max': 1800,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 10,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 600,
    # total number of epochs
    'n_epochs': 1800,
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/TUMVI"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/TUMVI"),
}


def _smoke_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net_class(**net_params).to(device)
    net.set_normalized_factors(torch.zeros(net_params['in_dim']), torch.ones(net_params['in_dim']))
    us = (1e-3 * torch.randn(2, 256, net_params['in_dim'], device=device)).float()
    hat_xs = net(us)
    Loss = train_params['loss_class']
    criterion = Loss(**train_params['loss']).to(device)
    target = train_params['loss']['target']
    x_dim = 4 if 'mask' in target else 3
    xs = (1e-3 * torch.randn(2, 256, x_dim, device=device)).float()
    if x_dim == 4:
        xs[:, :, 3] = 1.0
    loss = criterion(xs, hat_xs)
    print('smoke_ok', hat_xs.shape, float(loss.detach().cpu()))


def main():
    global data_dir, address

    parser = argparse.ArgumentParser()
    ep.add_common_entrypoint_args(
        parser,
        dataset_params=dataset_params,
        train_params=train_params,
        default_data_dir=data_dir,
        default_address=address,
        dataset_label="TUM-VI",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    address = args.address
    ep.apply_common_entrypoint_overrides(args, dataset_params=dataset_params, train_params=train_params)

    if args.mode == 'smoke':
        _smoke_test()
        return 0

    if not os.path.isdir(data_dir):
        print(f"Missing dataset directory: {data_dir!r}", file=sys.stderr)
        print("Download/extract datasets into ./data/ (see datasets_downloader.py), or pass --data-dir.", file=sys.stderr)
        return 2

    res_dir = train_params['res_dir']
    has_runs = os.path.isdir(res_dir) and len(os.listdir(res_dir)) > 0
    mode = args.mode
    if mode == 'auto':
        mode = 'test' if has_runs else 'train'

    if mode == 'train':
        learning_process = lr.GyroLearningBasedProcessing(
            train_params['res_dir'],
            train_params['tb_dir'],
            net_class,
            net_params,
            address=None,
            dt=train_params['loss']['dt'],
        )
        ep.apply_calib_args_for_train(learning_process, args)
        learning_process.train(dataset_class, dataset_params, train_params)
        return 0

    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        address=address,
        dt=train_params['loss']['dt'],
    )
    ep.apply_calib_args_for_test(learning_process, args)
    learning_process.test(dataset_class, dataset_params, ['test'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
################################################################################
