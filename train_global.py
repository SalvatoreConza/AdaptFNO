import argparse
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

from models.windnet import GlobalOperator
from era5.wind.datasets import Wind2dERA5

from common.training import CheckpointLoader
from workers.train import GlobalOperatorTrainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train Global Operator in WindNet model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataroot: str                       = str(config['dataset']['root'])
    pressure_level: str                 = int(config['dataset']['pressure_level'])
    latitude: Tuple[float, float]       = tuple(config['dataset']['latitude'])
    longitude: Tuple[float, float]      = tuple(config['dataset']['longitude'])
    fromdate: str                       = str(config['dataset']['fromdate'])
    todate: str                         = str(config['dataset']['todate'])
    bundle_size: int                    = int(config['dataset']['bundle_size'])
    window_size: int                    = int(config['dataset']['window_size'])
    resolution: Tuple[int, int]         = tuple(config['dataset']['resolution'])
    to_float16: bool                    = bool(config['dataset']['to_float16'])
    split: Tuple[float, float]          = tuple(config['dataset']['split'])

    u_dim: int                          = int(config['architecture']['u_dim'])
    width: int                          = int(config['architecture']['width'])
    depth: int                          = int(config['architecture']['depth'])
    x_modes: int                        = int(config['architecture']['x_modes'])
    y_modes: int                        = int(config['architecture']['y_modes'])
    from_checkpoint: Optional[str]      = config['architecture']['from_checkpoint']
    
    lambda_: float                      = float(config['training']['lambda'])
    noise_level: float                  = float(config['training']['noise_level'])
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    checkpoint_path: Optional[str]      = config['training']['checkpoint_path']
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Initialize the training datasets
    full_dataset = Wind2dERA5(
        dataroot=dataroot,
        pressure_level=pressure_level,
        latitude=latitude,
        longitude=longitude,
        fromdate=fromdate,
        todate=todate,
        bundle_size=bundle_size,
        window_size=window_size,
        resolution=resolution,
        to_float16=to_float16
    )
    train_dataset, val_dataset = random_split(
        dataset=full_dataset,
        lengths=split,
    )

    # Load model
    if from_checkpoint is not None:
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        net: nn.Module; optimizer: Optimizer
        net, optimizer = checkpoint_loader.load(scope=globals())
    else:
        net: nn.Module = GlobalOperator(
            in_timesteps=full_dataset.in_timesteps,
            out_timesteps=full_dataset.out_timesteps,
            u_dim=u_dim, 
            width=width, depth=depth,
            x_modes=x_modes, y_modes=y_modes,
        )
        optimizer: Optimizer = Adam(params=net.parameters(), lr=learning_rate)
    
    trainer = GlobalOperatorTrainer(
        model=net, optimizer=optimizer,
        spectral_regularization_coef=lambda_,
        noise_level=noise_level,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        device=device,
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=checkpoint_path,
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train the Global Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)




