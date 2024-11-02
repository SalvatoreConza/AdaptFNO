import os
import re
from typing import Tuple, List

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
        
        global_hash: str = hash_params(
            global_latitude=global_latitude, global_longitude=global_longitude, 
            global_resolution=global_resolution,
            indays=indays, outdays=outdays,
        )
        self.global_input_directory: str = os.path.join('tensors', 'globals', global_hash, 'input')
        assert os.path.isdir(self.global_input_directory), 'Data tensors are not prepared'

        self.global_input_filenames: List[str] = self._directory2filenames(directory=self.global_input_directory)
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
        if global_resolution is None:
            self.global_resolution: Tuple[int, int] = (
                (self.global_latitude[0] - self.global_latitude[1]) * 4 + 1,
                (self.global_longitude[1] - self.global_longitude[0]) * 4 + 1
            )
        else:
            self.global_resolution: Tuple[int, int] = global_resolution

        self.local_resolution: Tuple[int, int] = (
            (self.local_latitude[0] - self.local_latitude[1]) * 4 + 1,
            (self.local_longitude[1] - self.local_longitude[0]) * 4 + 1
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        global_input: torch.Tensor = torch.load(os.path.join(self.global_input_directory, self.global_input_filenames[idx]))
        local_input: torch.Tensor = torch.load(os.path.join(self.local_input_directory, self.local_input_filenames[idx]))
        local_output: torch.Tensor = torch.load(os.path.join(self.local_output_directory, self.local_output_filenames[idx]))
        return global_input, local_input, local_output

    def __len__(self) -> int:
        return len(self.local_input_filenames)

    def _directory2filenames(self, directory: str) -> List[str]:
        return sorted([
            fname 
            for fname in os.listdir(path=directory)
                if self.fromyear <= int(fname[2:6]) <= self.toyear
                    and fname.endswith('.pt')
        ])


class ERA5_6Hour_Prediction(ERA5_6Hour):

    def __init__(
        self,
        ondate: str,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[int, int] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        indays: int,
        outdays: int,
    ):
        self.ondate: dt.datetime = dt.datetime.strptime(ondate, '%Y%m%d')

        super().__init__(
            fromyear=self.ondate.year,
            toyear=self.ondate.year,
            global_latitude=global_latitude,
            global_longitude=global_longitude,
            global_resolution=global_resolution,
            local_latitude=local_latitude,
            local_longitude=local_longitude,
            indays=indays,
            outdays=outdays,
        )
        assert len(self) == 1, 'Prediction dataset must have a single sample'

    # override
    def _directory2filenames(self, directory: str) -> List[str]:
        # Get only one file that has the last input date matching with self.ondate
        filename: str = next(
            filter(
                lambda fname: self._checkfilename(fname),
                os.listdir(path=directory),
            )
        )
        return [filename]

    def _checkfilename(self, filename: str) -> bool:
        year = str(self.ondate.year)
        month = str(self.ondate.month)
        day = str(self.ondate.day)
        components: List[str] = filename.split('__')
        if year in components[0] and f'{month.zfill(2)}{day.zfill(2)}' in components[1].split('_')[-1]:
            return True
        return False
    
    def compute_out_timestamps(self) -> List[dt.datetime]:
        filename: str = self.local_input_filenames[0]
        components: List[str] = filename.split('__')
        year = int(components[0][-4:])
        out_datestrings: List[str] = components[-1].replace('.pt','').split('_')
        
        out_timestamps: List[dt.datetime] = []
        for datestring in out_datestrings:
            for hour in [0, 6, 12, 18]:
                out_timestamps.append(
                    dt.datetime(year=year, month=int(datestring[:2]), day=int(datestring[2:4]), hour=hour, minute=0, second=0)
                )
        
        return out_timestamps


if __name__ == '__main__':

    self = ERA5_6Hour(
        fromyear=2020,
        toyear=2022,
        global_latitude=(45, -45),
        global_longitude=(60, 150),
        global_resolution=(128, 128),
        local_latitude=None,
        local_longitude=None,
        indays=3,
        outdays=1,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=self, batch_size=32)



