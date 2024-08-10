import argparse
from typing import Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from torch.optim import Adam

from legacy.models.afno2d import AdaptiveFNO2d
from legacy.datasets.pdebench import OneShotDiffReact2d
import processes


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train the AFNO2d model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataset_path: str                   = str(config['dataset']['path'])
    input_step: int                     = int(config['dataset']['input_step'])
    target_step: int                    = int(config['dataset']['target_step'])
    resolution: Optional[Tuple[int,int]]= config['dataset']['resolution']
    n_samples: int                      = int(config['dataset']['n_samples'])
    dataset_split: Tuple[float, float]  = tuple(config['dataset']['split'])
    u_dim: int                          = int(config['architecture']['u_dim'])
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
    checkpoint_path: str                = str(config['training']['checkpoint_path'])

    # Initialize the training datasets
    full_dataset = OneShotDiffReact2d(
        dataroot=dataset_path,
        input_step=input_step,
        target_step=target_step,
        resolution=tuple(resolution),
    )
    subset = Subset(dataset=full_dataset, indices=list(range(n_samples)))
    train_dataset, val_dataset = random_split(
        dataset=subset,
        lengths=dataset_split,
    )

    # Load model
    if from_checkpoint is not None:
        net: nn.Module = torch.load(from_checkpoint)
    else:
        net: nn.Module = AdaptiveFNO2d(
            u_dim=u_dim, width=width, 
            x_modes=x_modes, y_modes=y_modes
        )
    
    net: nn.Module = net.to(device=device)
    net: nn.Module = processes.train(
        model=net, 
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=Adam(params=net.parameters(), lr=learning_rate),
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        n_epochs=n_epochs,
        patience=patience,
        tolerance=tolerance,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train the Adaptive FNO2d model.')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    
    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)

