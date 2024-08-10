import argparse
from typing import List, Tuple, Dict, Any, Optional
import yaml

import torch
from torch.optim import Adam

from legacy.datasets.pdebench import MultiStepDiffReact2d
from common.training import CheckpointLoader
from legacy.workers import Predictor
from models.arafno2d import AutoRegressiveAdaptiveFNO2d


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
    test_dataset = MultiStepDiffReact2d(
        dataroot=dataset_path,
        input_timesteps=input_timesteps,
        target_timestep=target_timestep,
        from_sample=from_sample,
        to_sample=to_sample,
        resolution=tuple(resolution) if resolution else None,
    )

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

