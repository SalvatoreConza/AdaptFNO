import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset, DataLoader
from torch.optim import Adam

from models import AdaptiveFNO2d
from datasets import OneShotDiffReact2d
import processes

def main(config: Dict[str, Any]) -> None:
    """
    Main function to predict using the AFNO2d model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataset_path: str                   = str(config['dataset']['path'])
    input_step: int                     = int(config['dataset']['input_step'])
    target_step: int                    = int(config['dataset']['target_step'])
    resolution: Optional[Tuple[int,int]]= config['dataset']['resolution']
    from_sample: int                    = int(config['dataset']['from_sample'])
    to_sample: int                      = int(config['dataset']['to_sample'])
    from_checkpoint: str                = str(config['architecture']['from_checkpoint'])

    # Initialize the training datasets
    full_dataset = OneShotDiffReact2d(
        dataroot=dataset_path,
        input_step=input_step,
        target_step=target_step,
        resolution=tuple(resolution),
    )
    subset = Subset(dataset=full_dataset, indices=list(range(from_sample, to_sample)))
    dataloader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

    # Load model
    net: nn.Module = torch.load(from_checkpoint)
    net: nn.Module = net.to(device=device)

    # Predict
    processes.predict(model=net, dataloader=dataloader)


if __name__ == "__main__":

    # Initialize the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Predict physical fields using Adaptive FNO2d model.')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)

