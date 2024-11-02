import argparse
from typing import Tuple, Dict, Any, Optional, List
import yaml

import torch
from torch.optim import Adam

from models.benchmarks import FNO3D
from era5.datasets import ERA5_6Hour_Prediction
from common.training import CheckpointLoader
from workers.predictor import BenchmarkPredictor


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
    ondate: str                         = str(config['predict_fno3d']['ondate'])
    indays: int                         = int(config['dataset']['indays'])
    outdays: int                        = int(config['dataset']['outdays'])

    from_checkpoint: str               = str(config['predict_fno3d']['from_checkpoint'])
    plot_resolution: Optional[List[int, int]] = config['predict_fno3d']['plot_resolution']

    # Initialize the model
    print(f'Predicting with {from_checkpoint}')
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net: FNO3D = checkpoint_loader.load(scope=globals())

    # Initialize the predictor
    local_predictor = BenchmarkPredictor(net=net)

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


