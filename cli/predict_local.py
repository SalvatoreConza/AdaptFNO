import argparse
from typing import Tuple, Dict, Any, Optional, List
import yaml

import torch
from torch.optim import Adam

from models.operators import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5
from common.training import CheckpointLoader
from workers.predictor import GlobalOperatorPredictor, LocalOperatorPredictor


def main(config: Dict[str, Any]) -> None:
    """
    Main function to inference Global Operator in WindNet model.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataroot: str                       = str(config['dataset']['root'])
    pressure_level: str                 = int(config['dataset']['pressure_level'])
    global_latitude: Tuple[float, float] = tuple(config['dataset']['global_latitude'])
    global_longitude: Tuple[float, float] = tuple(config['dataset']['global_longitude'])
    global_resolution: Tuple[int, int]  = tuple(config['dataset']['global_resolution'])
    local_latitude: Tuple[float, float] = tuple(config['dataset']['local_latitude'])
    local_longitude: Tuple[float, float] = tuple(config['dataset']['local_longitude'])
    local_resolution: Tuple[int, int]   = tuple(config['dataset']['local_resolution'])
    fromdate: str                       = str(config['dataset']['fromdate'])
    todate: str                         = str(config['dataset']['todate'])
    time_resolution: int                = int(config['dataset']['time_resolution'])
    bundle_size: int                    = int(config['dataset']['bundle_size'])

    from_checkpoint: str                = str(config['local_architecture']['from_checkpoint'])
    global_checkpoint: str              = str(config['global_architecture']['from_checkpoint'])
    plot_resolution: Optional[List[int, int]] = config['local_plotting']['plot_resolution']

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
        device=device,
    )

    # Initialize the test dataset
    dataset = Wind2dERA5(
        dataroot=dataroot,
        pressure_level=pressure_level,
        fromdate=fromdate,
        todate=todate,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        local_resolution=local_resolution,
        time_resolution=time_resolution,
        bundle_size=bundle_size,
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


