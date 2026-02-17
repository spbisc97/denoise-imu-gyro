
import os
import argparse
import sys
import torch
import optuna
import src.learning as lr
import src.entrypoint_utils as ep
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np
import time

# Set base directory for data and results
base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = './data/EUROC/dataset'

# Global variable to store best loss for return
best_trial_loss = float('inf')
EPOCHS = 300


def objective(trial):
    global best_trial_loss
    
    # Suggest hyperparameters
    lr_value = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 0.9)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    # loss_w = trial.suggest_float('loss_w', 1e4, 1e8, log=True) # Let's keep loss_w simpler for now or it might diverge too easily

    # Network parameters (incorporating trial suggestions)
    net_class = sn.GyroNet
    net_params = {
        'in_dim': 6,
        'out_dim': 3,
        'c0': 16,
        'dropout': dropout,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': momentum,
        'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
    }

    # Dataset parameters (same as main_EUROC.py)
    dataset_class = ds.EUROCDataset
    dataset_params = {
        'data_dir': data_dir,
        'predata_dir': os.path.join(base_dir, 'data/EUROC'),
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
            ],
        'val_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult',
            ],
        'test_seqs': [],
        'N': 32 * 500, 
        'min_train_freq': 16,
        'max_train_freq': 32,
    }

    # Training parameters
    train_params = {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {
            'lr': lr_value,
            'weight_decay': weight_decay,
            'amsgrad': False,
        },
        'loss_class': sl.GyroLoss,
        'loss': {
            'min_N': int(np.log2(dataset_params['min_train_freq'])),
            'max_N': int(np.log2(dataset_params['max_train_freq'])),
            'w':  1e6, # Fixed for now
            'target': 'rotation matrix',
            'huber': 0.005,
            'dt': 0.005,
        },
        'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingLR,
        'scheduler': {
            'T_max': 300, 
            'eta_min': 1e-3,
        },
        'dataloader': {
            'batch_size': 10,
            'pin_memory': False,
            'num_workers': 0,
            'shuffle': False,
        },
        'freq_val': max(1, EPOCHS // 3), 
        'n_epochs': EPOCHS, 
        # Unique directory for each trial to avoid conflicts
        'res_dir': os.path.join(base_dir, f"results/EUROC_optuna/trial_{trial.number}_{int(time.time())}"),
        'tb_dir': os.path.join(base_dir, f"results/runs/EUROC_optuna/trial_{trial.number}_{int(time.time())}"),
    }

    # Initialize learning process
    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        address=None,
        dt=train_params['loss']['dt'],
    )
    
    # Start training and get validation loss
    best_val_loss = learning_process.train(dataset_class, dataset_params, train_params)
    
    # Ensure it's a python float
    if hasattr(best_val_loss, 'item'):
        val = best_val_loss.item()
    else:
        val = best_val_loss
        
    return val


def main():
    parser = argparse.ArgumentParser(description='Optuna optimization for EUROC')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs per trial')
    args = parser.parse_args()

    if not os.path.isdir(data_dir):
        print(f"Missing dataset directory: {data_dir!r}", file=sys.stderr)
        return 2

    # Update global config or pass it to objective?
    # Since objective doesn't take extra args easily without functools.partial or class, 
    # we can use a global or a wrapper. 
    # Let's use a wrapper or simply set the default in the objective if not passed.
    # Actually, let's use a class or update the objective to use a global config dictionary.
    
    global EPOCHS
    EPOCHS = args.epochs

    storage_name = "sqlite:///optuna_study.db"
    study_name = "euroc_optimization"
    
    # Create study
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    
    # Run optimization
    print(f"Starting optimization with {args.trials} trials and {EPOCHS} epochs...")
    study.optimize(objective, n_trials=args.trials) 

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return 0

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
