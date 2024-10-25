import argparse
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import ERA5_6Hour
from common.training import CheckpointLoader
from workers.trainer import LocalOperatorTrainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train Global Operator.

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

    from_checkpoint: Optional[str]      = config['local_architecture']['from_checkpoint']
    global_checkpoint: str              = str(config['local_architecture']['global_checkpoint'])
    n_hmodes: int                       = str(config['local_architecture']['n_hmodes'])
    n_wmodes: int                       = str(config['local_architecture']['n_wmodes'])
    n_layers: int                       = int(config['local_architecture']['n_layers'])
    n_attention_heads: int              = int(config['local_architecture']['n_attention_heads'])

    noise_level: float                  = float(config['training']['noise_level'])
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Load global operator
    global_loader = CheckpointLoader(checkpoint_path=global_checkpoint)
    global_operator: GlobalOperator
    global_operator, _ = global_loader.load(scope=globals())

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

    # Load local operator
    if from_checkpoint is not None:
        local_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        local_operator: LocalOperator = local_loader.load(scope=globals())[0]   # ignore optimizer
    else:
        local_operator = LocalOperator(
            in_channels=global_operator.in_channels, 
            out_channels=global_operator.out_channels,
            embedding_dim=global_operator.embedding_dim,
            in_timesteps=global_operator.in_timesteps, 
            out_timesteps=global_operator.out_timesteps,
            n_hmodes=n_hmodes,
            n_wmodes=n_wmodes,
            n_layers=n_layers,
            spatial_resolution=train_dataset.local_resolution,
            n_attention_heads=n_attention_heads,
        )

    local_optimizer = Adam(params=local_operator.parameters(), lr=learning_rate)
    
    # Load local trainer
    trainer = LocalOperatorTrainer(
        local_operator=local_operator,
        global_operator=global_operator, 
        optimizer=local_optimizer,
        noise_level=noise_level,
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
        device=torch.device('cuda'),
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=f'.checkpoints/local',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train the Local Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)



