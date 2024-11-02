import argparse
from typing import List, Tuple, Dict, Any, Optional
import yaml

from models.benchmarks import FNO3D
from era5.datasets import ERA5_6Hour
from common.training import CheckpointLoader
from workers.trainer import BenchmarkTrainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train .

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    global_latitude: Tuple[float, float] = tuple(config['dataset']['global_latitude'])
    global_longitude: Tuple[float, float] = tuple(config['dataset']['global_longitude'])
    global_resolution: Tuple[int, int]  = tuple(config['dataset']['global_resolution'])
    local_latitude: Tuple[float, float] = tuple(config['dataset']['local_latitude'])
    local_longitude: Tuple[float, float] = tuple(config['dataset']['local_longitude'])
    train_fromyear: str                 = int(config['dataset']['train_fromyear'])
    train_toyear: str                   = int(config['dataset']['train_toyear'])
    val_fromyear: str                   = int(config['dataset']['val_fromyear'])
    val_toyear: str                     = int(config['dataset']['val_toyear'])
    indays: int                         = int(config['dataset']['indays'])
    outdays: int                        = int(config['dataset']['outdays'])

    embedding_dim: int                  = int(config['fno3d']['embedding_dim'])
    n_tmodes: int                       = int(config['fno3d']['n_tmodes'])
    n_hmodes: int                       = int(config['fno3d']['n_hmodes'])
    n_wmodes: int                       = int(config['fno3d']['n_wmodes'])
    n_layers: int                       = int(config['fno3d']['n_layers'])

    from_checkpoint: Optional[str]      = config['training_fno3d']['from_checkpoint']
    train_batch_size: int               = int(config['training_fno3d']['train_batch_size'])
    val_batch_size: int                 = int(config['training_fno3d']['val_batch_size'])
    learning_rate: float                = float(config['training_fno3d']['learning_rate'])
    n_epochs: int                       = int(config['training_fno3d']['n_epochs'])
    patience: int                       = int(config['training_fno3d']['patience'])
    tolerance: int                      = float(config['training_fno3d']['tolerance'])
    save_frequency: int                 = int(config['training_fno3d']['save_frequency'])

    # Instatiate the training datasets
    train_dataset = ERA5_6Hour(
        fromyear=train_fromyear,
        toyear=train_toyear,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        indays=indays,
        outdays=outdays,
    )
    val_dataset = ERA5_6Hour(
        fromyear=val_fromyear,
        toyear=val_toyear,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        indays=indays,
        outdays=outdays,
    )

    # Load FNO3D
    if from_checkpoint is not None:
        print(f'Training from {from_checkpoint}')
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        net: FNO3D = checkpoint_loader.load(scope=globals())
    else:
        net = FNO3D(
            in_channels=train_dataset.in_channels, 
            out_channels=train_dataset.out_channels,
            embedding_dim=embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=train_dataset.out_timesteps,
            n_tmodes=n_tmodes, n_hmodes=n_hmodes, n_wmodes=n_wmodes,
            n_layers=n_layers,
        )
    
    trainer = BenchmarkTrainer(
        net=net, 
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        learning_rate=learning_rate,
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=f'.checkpoints',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train FNO3D')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    main(config)



