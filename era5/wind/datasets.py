import os
import math
from typing import Tuple, List, Optional
import datetime as dt
from functools import lru_cache

import xarray as xr

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Wind2dERA5(Dataset):

    def __init__(
        self,
        dataroot: str,
        pressure_level: int,
        latitude: Tuple[float, float],
        longitude: Tuple[float, float],
        fromdate: str,
        todate: str,
        bundle_size: int,
        window_size: int,
        resolution: Tuple[int, int],
        to_float16: bool,
    ):
        
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.dataroot: str = dataroot
        self.pressure_level: int = pressure_level
        self.latitude: Tuple[int, int] = latitude
        self.longitude: Tuple[int, int] = longitude
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')
        self.bundle_size: int = bundle_size
        self.window_size: int = window_size
        self.resolution: Tuple[int, int] = resolution
        self.to_float16: bool = to_float16

        if 24 % self.bundle_size != 0:
            raise ValueError(f'bundle_size must be a divisor of 24, got {self.bundle_size}')

        self.datafolder: str = f'{dataroot}/{pressure_level}'
        self.filenames: List[str] = sorted([
            name for name in os.listdir(self.datafolder)
            if name.endswith('.grib')
            and self.fromdate <= dt.datetime.strptime(name.replace('.grib',''), '%Y%m%d') <= self.todate
        ])
        self.in_timesteps: int = self.bundle_size * self.window_size
        self.out_timesteps: int = self.bundle_size
        self.total_timesteps: int = len(self.filenames) * 24
        self.n_bundles: int = math.ceil(self.total_timesteps / self.bundle_size)
        self.raw_indices: List[Tuple[int, int]] = [(t // 24, t % 24) for t in range(len(self.filenames) * 24)]

    def __getitem__(self, bundle_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if bundle_idx >= len(self):
            raise IndexError
        input_slice, output_slice = self._compute_temporal_slices(bundle_idx=bundle_idx)
        # Get indices
        input_indices: List[int] = self.raw_indices[input_slice]
        output_indices: List[int] = self.raw_indices[output_slice]
        
        inputs: List[torch.Tensor] = [
            self._to_tensor(
                filename=self.filenames[day_index], 
                latitude=self.latitude, 
                longitude=self.longitude,
            )[hour_index]
            for day_index, hour_index in input_indices
        ]
        input: torch.Tensor = torch.stack(tensors=inputs, dim=0)
        del inputs

        outputs: List[torch.Tensor] = [
            self._to_tensor(
                filename=self.filenames[day_index], 
                latitude=self.latitude, 
                longitude=self.longitude
            )[hour_index]
            for day_index, hour_index in output_indices
        ]
        output: torch.Tensor = torch.stack(tensors=outputs, dim=0)
        del outputs
        return input, output

    def __len__(self) -> int:
        return self.n_bundles - self.window_size

    @lru_cache(maxsize=8)
    def _to_tensor(
        self, 
        filename: str, 
        latitude: Tuple[float, float], 
        longitude: Tuple[float, float]
    ) -> torch.Tensor:
        dataset: xr.Dataset = xr.open_dataset(f'{self.datafolder}/{filename}', engine='cfgrib')
        dataset: xr.Dataset = dataset.sel(
            latitude=slice(*latitude), longitude=slice(*longitude)
        )
        data: torch.Tensor = torch.tensor(data=dataset.to_dataarray().values)
        assert data.ndim == 4
        # Convert to shape (timesteps, 2, *self.resolution)
        data: torch.Tensor = data.permute(1, 0, 2, 3)
        # Transform resolution
        data: torch.Tensor = F.interpolate(
            input=data, size=self.resolution, mode='bicubic',
        )
        if self.to_float16: 
            data: torch.Tensor = data.to(dtype=torch.half)
        
        return data
        
    def _compute_temporal_slices(self, bundle_idx: int) -> Tuple[slice, slice]:
        left_idx: int = bundle_idx * self.bundle_size
        mid_idx: int = left_idx + self.in_timesteps
        right_idx: int = mid_idx + self.out_timesteps
        input_slice = slice(left_idx, mid_idx, 1)
        output_slice = slice(mid_idx, right_idx, 1)
        return input_slice, output_slice


if __name__ == '__main__':
    self = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        latitude=(10, -10),
        longitude=(160, 200),
        fromdate='20230101',
        todate='20230102',
        bundle_size=6,
        window_size=2,
        resolution=(64, 64),
        to_float16=True,
    )

    