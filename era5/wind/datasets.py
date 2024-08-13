import os
import math
from typing import Tuple, List, Optional
import datetime as dt
import asyncio
from concurrent.futures import ProcessPoolExecutor

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
        fromdate: str,
        todate: str,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[int, int] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        local_resolution: Tuple[int, int] | None,
        bundle_size: int,
        window_size: int,
    ):
        
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.dataroot: str = dataroot
        self.pressure_level: int = pressure_level
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')
        self.local_latitude: Tuple[int, int] | None = local_latitude
        self.local_longitude: Tuple[int, int] | None = local_longitude
        self.local_resolution: Tuple[int, int] | None = local_resolution
        self.global_latitude: Tuple[int, int] | None = global_latitude
        self.global_longitude: Tuple[int, int] | None = global_longitude
        self.global_resolution: Tuple[int, int] | None = global_resolution
        self.bundle_size: int = bundle_size
        self.window_size: int = window_size

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
        
        self.has_global: bool = all([global_latitude, global_longitude, global_resolution])
        self.has_local: bool = all([local_latitude, local_longitude, local_resolution])
        assert self.has_global or self.has_local, 'either global or local must be specified'

        if self.has_global:
            self.global_tensors: List[torch.Tensor] = asyncio.run(
                self._load_tensors(self.global_latitude, self.global_longitude, self.global_resolution)
            )

        if self.has_local:
            self.local_tensors: List[torch.Tensor] = asyncio.run(
                self._load_tensors(self.local_latitude, self.local_longitude, self.local_resolution)
            )

    def __getitem__(self, bundle_idx: int) -> Tuple[torch.Tensor, ...]:
        if bundle_idx >= len(self):
            raise IndexError
        
        input_slice, output_slice = self._compute_temporal_slices(bundle_idx=bundle_idx)
        sample: Tuple[torch.Tensor, ...] = tuple()  # (global_input, global_output, local_input, local_output) 
        if self.has_global:
            global_input = self._stack_tensors(tensors=self.global_tensors, time_slice=input_slice)
            global_output = self._stack_tensors(tensors=self.global_tensors, time_slice=output_slice)
            sample += (global_input, global_output)

        if self.has_local:
            local_input = self._stack_tensors(tensors=self.local_tensors, time_slice=input_slice)
            local_output = self._stack_tensors(tensors=self.local_tensors, time_slice=output_slice)
            sample += (local_input, local_output)

        return sample   

    def __len__(self) -> int:
        return self.n_bundles - self.window_size

    async def _load_tensors(
        self,
        latitude: Tuple[float, float], 
        longitude: Tuple[float, float],
        resolution: Tuple[int, int],
    ) -> List[torch.Tensor]:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=4) as pool:
            tasks: List[asyncio.Future] = [
                loop.run_in_executor(
                    pool, self._compute_tensor, filename, latitude, longitude, resolution,
                )
                for filename in self.filenames
            ]
            return await asyncio.gather(*tasks)

    def _compute_tensor(
        self, 
        filename: str, 
        latitude: Tuple[float, float], 
        longitude: Tuple[float, float],
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        dataset: xr.Dataset = xr.open_dataset(f'{self.datafolder}/{filename}', engine='cfgrib')
        dataset: xr.Dataset = dataset.sel(
            latitude=slice(*latitude), longitude=slice(*longitude)
        )
        data: torch.Tensor = torch.tensor(data=dataset.to_dataarray().values)
        # Convert to shape (timesteps, 2, *resolution)
        data: torch.Tensor = data.permute(1, 0, 2, 3)
        # Transform resolution
        data: torch.Tensor = F.interpolate(
            input=data, size=resolution, mode='bicubic',
        )
        return data

    def _compute_temporal_slices(self, bundle_idx: int) -> Tuple[slice, slice]:
        left_idx: int = bundle_idx * self.bundle_size
        mid_idx: int = left_idx + self.in_timesteps
        right_idx: int = mid_idx + self.out_timesteps
        input_slice = slice(left_idx, mid_idx, 1)
        output_slice = slice(mid_idx, right_idx, 1)
        return input_slice, output_slice
    
    def _stack_tensors(self, tensors: List[torch.Tensor], time_slice: slice) -> torch.Tensor:
        indices: List[Tuple[int, int]] = self.raw_indices[time_slice]
        return torch.stack([tensors[day][hour] for day, hour in indices], dim=0)


if __name__ == '__main__':
    dataset = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        fromdate='20230101',
        todate='20240731',
        bundle_size=6,
        window_size=2,
        global_latitude=(45, -45),
        global_longitude=(0, 180),
        global_resolution=(128, 128),
        local_latitude=None,
        local_longitude=None,
        local_resolution=None,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch, (input, output) in enumerate(dataloader, start=1):
        print(f'Load batch {batch}/{len(dataloader)}')
        print(input.shape)
        print(output.shape)
        print('--------')
