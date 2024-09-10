import os
from typing import Tuple, List

from functools import lru_cache
import datetime as dt

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from common.functional import hash_params


class ERA5_6Hour(Dataset):

    def __init__(
        self,
        fromdate: str,
        todate: str,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[int, int] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        indays: int,
        outdays: int,
    ):
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')
        self.global_latitude: Tuple[float, float] | None = global_latitude
        self.global_longitude: Tuple[float, float] | None = global_longitude
        self.local_latitude: Tuple[float, float] | None = local_latitude
        self.local_longitude: Tuple[float, float] | None = local_longitude
        self.indays: int = indays
        self.outdays: int = outdays
        
        self.in_channels: int = 20
        self.out_channels: int = 2

        self.time_resolution: int = 6
        self.timesteps_per_day: int = 24 // self.time_resolution
        self.in_timesteps: int = self.timesteps_per_day * indays
        self.out_timesteps: int = self.timesteps_per_day * outdays
        
        # Check global/local dataset
        self.has_global: bool = all([global_latitude, global_longitude])
        self.has_local: bool = all([local_latitude, local_longitude])
        assert self.has_global or self.has_local, 'either global or local must be specified'

        # Get tensor directories
        self.tensor_root: str = os.path.join(
            'tensors', 
            hash_params(
                fromdate=fromdate, todate=todate, 
                global_latitude=global_latitude, global_longitude=global_longitude,
                local_latitude=local_latitude, local_longitude=local_longitude,
                indays=indays, outdays=outdays,
            )
        )
        assert os.path.isdir(self.tensor_root), 'Data tensors are not prepared'
        if self.has_global:
            self.global_input_directory: str = os.path.join(self.tensor_root, 'global', 'input')
            self.global_output_directory: str = os.path.join(self.tensor_root, 'global', 'output')
            assert os.path.isdir(self.global_input_directory)
            assert os.path.isdir(self.global_output_directory)
            assert len(os.listdir(self.global_input_directory)) == len(os.listdir(self.global_output_directory))

        if self.has_local:
            self.local_input_directory: str = os.path.join(self.tensor_root, 'local', 'input')
            self.local_output_directory: str = os.path.join(self.tensor_root, 'local', 'output')
            assert os.path.isdir(self.local_input_directory)
            assert os.path.isdir(self.local_output_directory)
            assert len(os.listdir(self.local_input_directory)) == len(os.listdir(self.local_output_directory))

        # Compute resolution
        if self.has_global:
            if global_resolution is None:
                self.global_resolution: Tuple[int, int] = (
                    (self.global_latitude[0] - self.global_latitude[1]) * 4 + 1,
                    (self.global_longitude[1] - self.global_longitude[0]) * 4 + 1
                )
            else:
                self.global_resolution: Tuple[int, int] = global_resolution
        else:
            self.global_resolution = None

        if self.has_local:
            self.local_resolution: Tuple[int, int] = (
                    (self.local_latitude[0] - self.local_latitude[1]) * 4 + 1,
                    (self.local_longitude[1] - self.local_longitude[0]) * 4 + 1
                )
        else:
            self.local_resolution = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample: Tuple[torch.Tensor, ...] = tuple()

        if self.has_global:
            global_input: torch.Tensor = torch.load(os.path.join(self.global_input_directory, f"GI{idx}.pt"))
            global_output: torch.Tensor = torch.load(os.path.join(self.global_output_directory, f"GO{idx}.pt"))
            # Resize
            if (self.global_resolution is not None) and (self.global_resolution != tuple(global_input.shape[-2:])):
                global_input = F.interpolate(input=global_input, size=self.global_resolution, mode='bicubic')

            if (self.global_resolution is not None) and (self.global_resolution != tuple(global_output.shape[-2:])):
                global_output = F.interpolate(input=global_output, size=self.global_resolution, mode='bicubic')
            
            sample += (global_input, global_output)

        if self.has_local:
            local_input: torch.Tensor = torch.load(os.path.join(self.local_input_directory, f"LI{idx}.pt"))
            local_output: torch.Tensor = torch.load(os.path.join(self.local_output_directory, f"LO{idx}.pt"))
            sample += (local_input, local_output)

        return sample

    def __len__(self) -> int:
        if hasattr(self, 'global_input_directory'):
            return len(os.listdir(self.global_input_directory))
        else:
            return len(os.listdir(self.locall_input_directory))

if __name__ == '__main__':

    self = ERA5_6Hour(
        fromdate='20230101',
        todate='20231231',
        global_latitude=(45, -45),
        global_longitude=(60, 150),
        global_resolution=None,
        local_latitude=(30, -10),
        local_longitude=(90, 130),
        indays=3,
        outdays=1,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=self, batch_size=32, num_workers=0)



