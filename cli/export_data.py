import argparse
from typing import Tuple, Dict, Any
import yaml

from era5.precompute import TensorWriter


def main(config: Dict[str, Any], year: int) -> None:
    """
    Main function to write grib dataset to pytorch tensor files.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
        year (int): Year of data to export
    """

    # Parse CLI arguments:
    global_latitude: Tuple[float, float]    = config['dataset']['global_latitude']
    global_longitude: Tuple[float, float]   = config['dataset']['global_longitude']
    global_resolution: Tuple[float]         = config['dataset']['global_resolution']
    local_latitude: Tuple[float, float]     = config['dataset']['local_latitude']
    local_longitude: Tuple[float, float]    = config['dataset']['local_longitude']
    indays: int                             = int(config['dataset']['indays'])
    outdays: int                            = int(config['dataset']['outdays'])

    # Instatiate the training datasets
    train_dataset = TensorWriter(
        year=year,
        global_latitude=tuple(global_latitude) if global_latitude else None,
        global_longitude=tuple(global_longitude) if global_longitude else None,
        global_resolution=tuple(global_resolution) if global_resolution else None,
        local_latitude=tuple(local_latitude) if local_latitude else None,
        local_longitude=tuple(local_longitude) if local_longitude else None,
        indays=indays,
        outdays=outdays,
    )
    # Write tensor files
    train_dataset.write2disk()


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Write grib dataset to pytorch tensor files')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    parser.add_argument('--year', type=int, required=True, help='Year of data to export')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config, args.year)
