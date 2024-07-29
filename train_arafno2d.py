import argparse
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from torch.optim import Optimizer, Adam

from models.arafno2d import AutoRegressiveAdaptiveFNO2d
from datasets import AutoRegressiveDiffReact2d
from utils.training import CheckpointLoader
from workers import Trainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train the AFNO2d model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataset_path: str                   = str(config['dataset']['path'])
    window_size: int                    = int(config['dataset']['window_size'])
    resolution: Optional[List[int,int]] = config['dataset']['resolution']
    seen_samples: int                   = int(config['dataset']['seen_samples'])
    dataset_split: Tuple[float, float]  = tuple(config['dataset']['split'])

    u_dim: int                          = int(config['architecture']['u_dim'])
    depth: int                          = int(config['architecture']['depth'])
    width: int                          = int(config['architecture']['width'])
    x_modes: int                        = int(config['architecture']['x_modes'])
    y_modes: int                        = int(config['architecture']['y_modes'])
    from_checkpoint: Optional[str]      = config['architecture']['from_checkpoint']
    
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    learning_rate: float                = float(config['training']['learning_rate'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    checkpoint_path: Optional[str]      = config['training']['checkpoint_path']
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Initialize the training datasets
    full_dataset = AutoRegressiveDiffReact2d(
        dataroot=dataset_path,
        window_size=window_size,
        resolution=tuple(resolution),
    )
    subset = Subset(dataset=full_dataset, indices=list(range(seen_samples)))
    train_dataset, val_dataset = random_split(
        dataset=subset,
        lengths=dataset_split,
    )

    # Load model
    if from_checkpoint is not None:
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        net: nn.Module; optimizer: Optimizer
        net, optimizer = checkpoint_loader.load()
    else:
        net: nn.Module = AutoRegressiveAdaptiveFNO2d(
            window_size=window_size, u_dim=u_dim, 
            width=width, depth=depth,
            x_modes=x_modes, y_modes=y_modes,
        )
        optimizer: Optimizer = Adam(params=net.parameters(), lr=learning_rate)
    
    trainer = Trainer(
        model=net, optimizer=optimizer,
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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train the ARAFNO2d model.')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)

