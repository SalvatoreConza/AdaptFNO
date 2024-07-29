import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.optim import Optimizer, Adam

from models.arafno2d import AutoRegressiveAdaptiveFNO2d
from datasets import MultiStepDiffReact2d
from utils.training import CheckpointLoader
from workers import Predictor


def main(config: Dict[str, Any]) -> None:
    """
    Main function to predict using the AFNO2d model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataset_path: str                   = str(config['dataset']['path'])
    input_timesteps: List[int]          = config['dataset']['input_timesteps']
    target_timestep: int                = int(config['dataset']['target_timestep'])
    resolution: Optional[List[int, int]]= config['dataset']['resolution']
    from_sample: int                    = int(config['dataset']['from_sample'])
    to_sample: int                      = int(config['dataset']['to_sample'])
    from_checkpoint: str                = str(config['architecture']['from_checkpoint'])

    # Initialize the training datasets
    full_dataset = MultiStepDiffReact2d(
        dataroot=dataset_path,
        input_timesteps=input_timesteps,
        target_timestep=target_timestep,
        resolution=tuple(resolution),
    )
    test_dataset = Subset(dataset=full_dataset, indices=list(range(from_sample, to_sample)))

    # Load model
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net, _ = checkpoint_loader.load(scope=globals())

    # Predict
    predictor = Predictor(model=net, device=device)
    predictor.predict(dataset=test_dataset)


if __name__ == "__main__":

    # Initialize the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Predict physical fields using ARAFNO2d model.')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)

