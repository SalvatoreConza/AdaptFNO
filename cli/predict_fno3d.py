import argparse
from typing import Tuple, Dict, Any, Optional, List
import yaml

import torch
from torch.optim import Adam

from models.benchmarks import FNO2D
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
    net: FNO2D = checkpoint_loader.load(scope=globals())

    # Initialize the predictor
    local_predictor = BenchmarkPredictor(net=net)

    dfs = []
    for ondate in ['20240110', '20240113', '20240116', '20240119', '20240122', '20240125', '20240128', 
        '20240131', '20240203', '20240206', '20240209', '20240212', '20240215', '20240218', 
        '20240221', '20240224', '20240227', '20240301', '20240304', '20240307', '20240310', 
        '20240313', '20240316', '20240319', '20240322', '20240325', '20240328', '20240331', 
        '20240403', '20240406', '20240409', '20240412', '20240415', '20240418', '20240421', 
        '20240424', '20240427', '20240430', '20240503', '20240506', '20240509', '20240512', 
        '20240515', '20240518', '20240521', '20240524', '20240527', '20240530', '20240602', 
        '20240605', '20240608', '20240611', '20240614', '20240617', '20240620', '20240623', 
        '20240626', '20240629', '20240702', '20240705', '20240708', '20240711', '20240714', 
        '20240717', '20240720', '20240723', '20240726', '20240729', '20240801', '20240804', 
        '20240807', '20240810', '20240813', '20240816', '20240819', '20240822', '20240825', 
        '20240828', '20240831', '20240903', '20240906', '20240909', '20240912', '20240915'
    ]:

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
        
        df = local_predictor.predict(dataset=dataset, plot_resolution=plot_resolution)

        dfs.append(df)

    import pandas as pd
    a = pd.concat(dfs, axis=0, ignore_index=True)
    a.to_json('evaluate.json', index=False, orient='records')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    main(config)


