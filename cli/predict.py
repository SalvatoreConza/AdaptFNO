import argparse
from typing import Tuple, Dict, Any, Optional, List
import yaml

import torch
from torch.optim import Adam

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import ERA5_6Hour_Prediction
from common.training import CheckpointLoader
from workers.predictor import MainPredictor


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
    ondate: str                         = str(config['predict']['ondate'])
    indays: int                         = int(config['dataset']['indays'])
    outdays: int                        = int(config['dataset']['outdays'])

    local_checkpoint: str               = str(config['predict']['local_checkpoint'])
    global_checkpoint: str              = str(config['predict']['global_checkpoint'])
    plot_resolution: Optional[List[int, int]] = config['predict']['plot_resolution']

    # Initialize the global operator from global checkpoint
    print(f'Conditioning on {global_checkpoint}')
    global_loader = CheckpointLoader(checkpoint_path=global_checkpoint)
    global_operator: GlobalOperator = global_loader.load(scope=globals())

    # Initialize the local operator from local checkpoint
    print(f'Predicting with {local_checkpoint}')
    local_loader = CheckpointLoader(checkpoint_path=local_checkpoint)
    local_operator: LocalOperator = local_loader.load(scope=globals())

    # Initialize the predictor
    local_predictor = MainPredictor(global_operator=global_operator, local_operator=local_operator)

    # Initialize the test dataset
    dataset = ERA5_6Hour_Prediction(
        ondate=ondate,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        indays=indays,
        outdays=outdays,
    )
    
    local_predictor.predict(dataset=dataset, plot_resolution=plot_resolution)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    main(config)


