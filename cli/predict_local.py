import argparse
from typing import Tuple, Dict, Any, Optional, List
import yaml

import torch
from torch.optim import Adam

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import ERA5_6Hour_Prediction
from common.training import CheckpointLoader
from workers.predictor import LocalOperatorPredictor


def main(config: Dict[str, Any]) -> None:
    """
    Main function to inference Global Operator in WindNet model.

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

    from_checkpoint: str                = str(config['predict']['from_local_checkpoint'])
    global_checkpoint: str              = str(config['local_architecture']['global_checkpoint'])
    plot_resolution: Optional[List[int, int]] = config['predict']['local_plot_resolution']

    # Initialize the global operator from global checkpoint
    global_loader = CheckpointLoader(checkpoint_path=global_checkpoint)
    global_operator: GlobalOperator
    global_operator, _ = global_loader.load(scope=globals())

    # Initialize the local operator from local checkpoint
    local_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    local_operator: LocalOperator
    local_operator, _ = local_loader.load(scope=globals())

    # Initialize the predictor
    local_predictor = LocalOperatorPredictor(
        global_operator=global_operator,
        local_operator=local_operator,
        device=torch.device('cuda'),
    )

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

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Inference the Global Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)


