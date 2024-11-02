import argparse
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import ERA5_6Hour
from common.training import CheckpointLoader
from workers.trainer import MainTrainer


def main(config: Dict[str, Any]) -> None:
    """
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

    global_embedding_dim: int           = int(config['global_architecture']['embedding_dim'])
    global_n_tmodes: int                = int(config['global_architecture']['n_tmodes'])
    global_n_hmodes: int                = int(config['global_architecture']['n_hmodes'])
    global_n_wmodes: int                = int(config['global_architecture']['n_wmodes'])
    global_n_layers: int                = int(config['global_architecture']['n_layers'])

    local_embedding_dim: int            = int(config['local_architecture']['embedding_dim'])
    local_n_tmodes: int                 = int(config['local_architecture']['n_tmodes'])
    local_n_hmodes: int                 = int(config['local_architecture']['n_hmodes'])
    local_n_wmodes: int                 = int(config['local_architecture']['n_wmodes'])
    local_n_layers: int                 = int(config['local_architecture']['n_layers'])
    local_downsampling: float           = float(config['local_architecture']['local_downsampling'])
    n_attention_heads: int              = int(config['local_architecture']['n_attention_heads'])
    global_patch_size: Tuple[int]       = tuple(config['local_architecture']['global_patch_size'])

    local_checkpoint: Optional[str]     = config['training']['local_checkpoint']
    global_checkpoint: Optional[str]    = config['training']['global_checkpoint']
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    save_frequency: int                 = int(config['training']['save_frequency'])
    freeze_global: bool                 = bool(config['training']['freeze_global'])
    freeze_local: bool                  = bool(config['training']['freeze_local'])

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

    # Load global operator
    if global_checkpoint is not None:
        print(f'Training from {global_checkpoint}')
        global_loader = CheckpointLoader(checkpoint_path=global_checkpoint)
        global_operator: GlobalOperator = global_loader.load(scope=globals())
    else:
        global_operator = GlobalOperator(
            in_channels=train_dataset.in_channels, 
            out_channels=train_dataset.out_channels,
            embedding_dim=global_embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=train_dataset.out_timesteps,
            n_tmodes=global_n_tmodes, n_hmodes=global_n_hmodes, n_wmodes=global_n_wmodes,
            n_layers=global_n_layers,
            spatial_resolution=train_dataset.global_resolution,
        )

    # Load local operator
    if local_checkpoint is not None:
        print(f'Training from {local_checkpoint}')
        local_loader = CheckpointLoader(checkpoint_path=local_checkpoint)
        local_operator: LocalOperator = local_loader.load(scope=globals())
    else:
        local_operator = LocalOperator(
            in_channels=train_dataset.in_channels, 
            out_channels=train_dataset.out_channels,
            local_embedding_dim=local_embedding_dim,
            global_embedding_dim=global_embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=train_dataset.out_timesteps,
            n_tmodes=local_n_tmodes, n_hmodes=local_n_hmodes, n_wmodes=local_n_wmodes,
            n_layers=local_n_layers,
            n_attention_heads=n_attention_heads,
            global_patch_size=global_patch_size,
            spatial_resolution=(
                int(train_dataset.local_resolution[0] / local_downsampling), 
                int(train_dataset.local_resolution[1] / local_downsampling),
            )
        )

    if freeze_global:
        print('Freezed global operator')
        for param in global_operator.parameters():
            param.requires_grad = False
    
    if freeze_local:
        print('Freezed local operator')
        for param in local_operator.parameters():
            param.requires_grad = False
    
    trainer = MainTrainer(
        local_operator=local_operator,
        global_operator=global_operator, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
        learning_rate=learning_rate,
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=f'.checkpoints',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train operators')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    main(config)



