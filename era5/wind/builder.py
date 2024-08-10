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
        global_latitude: Tuple[float, float],
        global_longitude: Tuple[float, float],
        local_latitude: Tuple[float, float],
        local_longitude: Tuple[float, float],
        from_date: str,
        to_date: str,
        bundle_size: int,
        window_size: int = 1,
        resolution: Optional[Tuple[int, int]] = None,
        to_float16: bool = False,
    ):
        
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.dataroot: str = dataroot
        self.pressure_level: int = pressure_level
        self.global_latitude: Tuple[int, int] = global_latitude
        self.global_longitude: Tuple[int, int] = global_longitude
        self.local_latitude: Tuple[int, int] = local_latitude
        self.local_longitude: Tuple[int, int] = local_longitude
        self.from_date: dt.datetime = dt.datetime.strptime(from_date, '%Y%m%d')
        self.to_date: dt.datetime = dt.datetime.strptime(to_date, '%Y%m%d')
        self.bundle_size: int = bundle_size
        self.window_size: int = window_size
        self.resolution: Optional[Tuple[int, int]] = resolution
        self.to_float16: bool = to_float16

        if 24 % self.bundle_size != 0:
            raise ValueError(f'The bundle_size must be a divisor of 24, got {self.bundle_size}')

        self.datafolder: str = f'{dataroot}/{pressure_level}'
        self.filenames: List[str] = sorted([
            name for name in os.listdir(self.datafolder)
            if name.endswith('.grib')
            and self.from_date <= dt.datetime.strptime(name.replace('.grib',''), '%Y%m%d') <= self.to_date
        ])
        self.timesteps: int = len(self.filenames) * 24
        self.n_bundles: int = math.ceil(self.timesteps / self.bundle_size)
        self.raw_indices: List[Tuple[int, int]] = [(t // 24, t % 24) for t in range(len(self.filenames) * 24)]

    def __getitem__(self, bundle_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bundle_idx >= len(self):
            raise IndexError
        input_slice, output_slice = self._compute_temporal_slices(bundle_idx=bundle_idx)
        # Get indices
        input_indices: List[int] = self.raw_indices[input_slice]
        output_indices: List[int] = self.raw_indices[output_slice]
        
        # Get global input
        global_inputs: List[torch.Tensor] = [
            self._to_tensor(
                filename=self.filenames[day_index], 
                latitude=self.global_latitude, 
                longitude=self.global_longitude,
            )[hour_index]
            for day_index, hour_index in input_indices
        ]
        global_input: torch.Tensor = torch.stack(tensors=global_inputs, dim=0)
        del global_inputs

        # Get local input
        local_inputs: List[torch.Tensor] = [
            self._to_tensor(
                filename=self.filenames[day_index], 
                latitude=self.local_latitude, 
                longitude=self.local_longitude,
            )[hour_index]
            for day_index, hour_index in input_indices
        ]
        local_input: torch.Tensor = torch.stack(tensors=local_inputs, dim=0)
        del local_inputs

        # Get local output
        local_outputs: List[torch.Tensor] = [
            self._to_tensor(
                filename=self.filenames[day_index], 
                latitude=self.local_latitude, 
                longitude=self.local_longitude
            )[hour_index]
            for day_index, hour_index in output_indices
        ]
        local_output: torch.Tensor = torch.stack(tensors=local_outputs, dim=0)
        del local_outputs
        return global_input, local_input, local_output

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

        if self.to_float16: 
            data: torch.Tensor = data.to(dtype=torch.half)
        
        data: torch.Tensor = F.interpolate(
            input=data, size=self.resolution, mode='bicubic', align_corners=False,
        )
        return data.permute(1, 0, 2, 3) # (timesteps, u_dim, latitude, longitude)
        
    def _compute_temporal_slices(self, bundle_idx: int) -> Tuple[slice, slice]:
        left_idx: int = bundle_idx * self.bundle_size
        mid_idx: int = left_idx + self.window_size * self.bundle_size
        right_idx: int = mid_idx + self.bundle_size
        input_slice = slice(left_idx, mid_idx, 1)
        output_slice = slice(mid_idx, right_idx, 1)
        return input_slice, output_slice


if __name__ == '__main__':
    self = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        global_latitude=(90, -90),
        global_longitude=(0, 360),
        local_latitude=(10, -10),
        local_longitude=(160, 200),
        from_date='20230101',
        to_date='20230101',
        bundle_size=6,
        window_size=2,
        resolution=(64, 64),
        to_float16=True,
    )

