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
        fromyear: int,
        toyear: int,
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
        self.fromyear: int = fromyear
        self.toyear: int = toyear
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

        if self.has_global:
            global_hash: str = hash_params(
                global_latitude=global_latitude, global_longitude=global_longitude, 
                global_resolution=global_resolution,
                indays=indays, outdays=outdays,
            )
            self.global_input_directory: str = os.path.join('tensors', 'globals', global_hash, 'input')
            self.global_output_directory: str = os.path.join('tensors', 'globals', global_hash, 'output')
            assert os.path.isdir(self.global_input_directory), 'Data tensors are not prepared'
            assert os.path.isdir(self.global_output_directory), 'Data tensors are not prepared'

            self.global_input_filenames: List[str] = self._directory2filenames(directory=self.global_input_directory)
            self.global_output_filenames: List[str] = self._directory2filenames(directory=self.global_output_directory)
            assert len(self.global_input_filenames) == len(self.global_output_filenames)

        if self.has_local:
            local_hash: str = hash_params(
                local_latitude=local_latitude, local_longitude=local_longitude,
                indays=indays, outdays=outdays,
            )
            self.local_input_directory: str = os.path.join('tensors', 'locals', local_hash, 'input')
            self.local_output_directory: str = os.path.join('tensors', 'locals', local_hash, 'output')
            assert os.path.isdir(self.local_input_directory), 'Data tensors are not prepared'
            assert os.path.isdir(self.local_output_directory), 'Data tensors are not prepared'
            
            self.local_input_filenames: List[str] = self._directory2filenames(directory=self.local_input_directory)
            self.local_output_filenames: List[str] = self._directory2filenames(directory=self.local_output_directory)
            assert len(self.local_input_filenames) == len(self.local_output_filenames)

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
            global_input: torch.Tensor = torch.load(os.path.join(self.global_input_directory, self.global_input_filenames[idx]))
            global_output: torch.Tensor = torch.load(os.path.join(self.global_output_directory, self.global_output_filenames[idx]))
            sample += (global_input, global_output)

        if self.has_local:
            local_input: torch.Tensor = torch.load(os.path.join(self.local_input_directory, self.local_input_filenames[idx]))
            local_output: torch.Tensor = torch.load(os.path.join(self.local_output_directory, self.local_output_filenames[idx]))
            sample += (local_input, local_output)

        return sample

    def __len__(self) -> int:
        if self.has_global:
            return len(self.global_input_filenames)
        else:
            return len(self.local_input_filenames)

    def _directory2filenames(self, directory: str) -> List[str]:
        return sorted([
            fname 
            for fname in os.listdir(path=directory)
                if self.fromyear <= int(fname[2:6]) <= self.toyear
                    and fname.endswith('.pt')
        ])


class ERA5_6Hour_Inference(ERA5_6Hour):

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
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')

        super().__init__(
            fromyear=fromdate.year,
            toyear=todate.year,
            global_latitude=global_latitude,
            global_longitude=global_longitude,
            global_resolution=global_resolution,
            local_latitude=local_latitude,
            local_longitude=local_longitude,
            indays=indays,
            outdays=outdays,
        )

    # override
    def _directory2filenames(self, directory: str) -> List[str]:
        filenames: List[str] = []
        for fname in os.listdir(path=directory):
            min_date, max_date = self._filename2datetimes(filename=fname)
            if self.fromdate <= min_date and max_date <= self.todate:
                filenames.append(fname)
        
        return sorted(filenames)

    @staticmethod
    def _filename2datetimes(filename: str) -> Tuple[dt.datetime, dt.datetime]:
        components: List[str] = filename.split('__')
        year = int(components[0][-4:])
        min_month = int(components[1].split('_')[0][:2])
        min_day = int(components[1].split('_')[0][2:4])
        max_month = int(components[2].split('_')[-1][:2])
        max_day = int(components[2].split('_')[-1][2:4])
        return (
            dt.datetime(year=year, month=min_month, day=min_day), 
            dt.datetime(year=year, month=max_month, day=max_day)
        )


if __name__ == '__main__':

    self = ERA5_6Hour(
        fromyear=2020,
        toyear=2022,
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



