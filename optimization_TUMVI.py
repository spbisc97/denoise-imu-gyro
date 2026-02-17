
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
import yaml

# Set base directory for data and results
base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = './data/TUMVI/dataset'

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
    # loss_w = trial.suggest_float('loss_w', 1e4, 1e8, log=True) 

    # Network parameters (TUMVI specific)
    net_class = sn.GyroNet
    net_params = {
        'in_dim': 6,
        'out_dim': 3,
        'c0': 16,
        'dropout': dropout,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': momentum,
        'gyro_std': [0.2*np.pi/180, 0.2*np.pi/180, 0.2*np.pi/180], # TUMVI specific
    }

    # Dataset parameters (TUMVI specific)
    dataset_class = ds.TUMVIDataset
    dataset_params = {
        'data_dir': data_dir,
        'predata_dir': os.path.join(base_dir, 'data/TUMVI'),
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
            'w':  1e6,
            'target': 'rotation matrix mask', # TUMVI specific
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
        'res_dir': os.path.join(base_dir, f"results/TUMVI_optuna/trial_{trial.number}_{int(time.time())}"),
        'tb_dir': os.path.join(base_dir, f"results/runs/TUMVI_optuna/trial_{trial.number}_{int(time.time())}"),
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
    parser = argparse.ArgumentParser(description='Optuna optimization for TUMVI')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs per trial')
    args = parser.parse_args()

    if not os.path.isdir(data_dir):
        print(f"Missing dataset directory: {data_dir!r}", file=sys.stderr)
        return 2
    
    global EPOCHS
    EPOCHS = args.epochs

    storage_name = "sqlite:///optuna_study_tumvi.db"
    study_name = "tumvi_optimization"
    
    # Create study
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    
    # Run optimization
    print(f"Starting TUMVI optimization with {args.trials} trials and {EPOCHS} epochs...")
    study.optimize(objective, n_trials=args.trials) 

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params to YAML
    best_params_path = os.path.join(base_dir, "best_params_TUMVI.yaml")
    with open(best_params_path, 'w') as f:
        yaml.dump(trial.params, f)
    print(f"Best parameters saved to {best_params_path}")

    return 0

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
