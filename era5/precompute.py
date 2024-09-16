import os
import datetime as dt
from typing import Tuple, List

import xarray as xr
import torch
import torch.nn.functional as F

from common.functional import hash_params


class TensorWriter:

    def __init__(
        self,
        year: int,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[float, float] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        indays: int,
        outdays: int,
    ):
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        self.year: int = year
        self.global_latitude: Tuple[float, float] | None = global_latitude
        self.global_longitude: Tuple[float, float] | None = global_longitude
        self.global_resolution: Tuple[float, float] | None = global_resolution
        self.local_latitude: Tuple[float, float] | None = local_latitude
        self.local_longitude: Tuple[float, float] | None = local_longitude
        self.indays: int = indays
        self.outdays: int = outdays

        self.in_channels: int = 20
        self.out_channels: int = 2

        self.time_resolution: int = 6   # fixed
        self.timesteps_per_day: int = 24 // self.time_resolution
        self.in_timesteps: int = self.timesteps_per_day * indays
        self.out_timesteps: int = self.timesteps_per_day * outdays
        
        self.grib_path: str = f'./data/{year}.grib'
        assert os.path.isfile(self.grib_path), 'Data file is not available'
        self.dataset: xr.Dataset = xr.open_dataset(self.grib_path, engine='cfgrib').sortby('time')
        self.total_timesteps: int = len(self.dataset.time)

        self.slices: List[Tuple[slice, slice]] = [
            (slice(s, s + self.in_timesteps), slice(s + self.in_timesteps, s + self.in_timesteps + self.out_timesteps))
            for s in range(0, self.total_timesteps - self.in_timesteps + 1, self.out_timesteps)
                if s + self.in_timesteps + self.out_timesteps <= self.total_timesteps   # drop incomplete slice
        ]

        # Check global/local dataset
        self.has_global: bool = all([global_latitude, global_longitude])
        self.has_local: bool = all([local_latitude, local_longitude])
        assert self.has_global or self.has_local, 'either global or local must be specified'

        # Precompute sub-dataset for faster indexing (xr.Dataset does lazy loading)
        if self.has_global:
            self.global_subsets = self._prepare_subsets(self.dataset, self.global_latitude, self.global_longitude)

        if self.has_local:
            self.local_subsets = self._prepare_subsets(self.dataset, self.local_latitude, self.local_longitude)

        # Precompute datestrings
        self.datestrings: List[str] = self._prepare_datestrings(
            subsets=self.global_subsets if self.has_global else self.local_subsets
        )

        if self.has_global:
            global_hash: str = hash_params(
                global_latitude=global_latitude, global_longitude=global_longitude, 
                global_resolution=global_resolution,
                indays=indays, outdays=outdays,
            )
            self.global_input_directory: str = os.path.join('tensors', 'globals', global_hash, 'input')
            self.global_output_directory: str = os.path.join('tensors', 'globals', global_hash, 'output')
            os.makedirs(self.global_input_directory, exist_ok=True)
            os.makedirs(self.global_output_directory, exist_ok=True)

        if self.has_local:
            local_hash: str = hash_params(
                local_latitude=local_latitude, local_longitude=local_longitude,
                indays=indays, outdays=outdays,
            )
            self.local_input_directory: str = os.path.join('tensors', 'locals', local_hash, 'input')
            self.local_output_directory: str = os.path.join('tensors', 'locals', local_hash, 'output')
            os.makedirs(self.local_input_directory, exist_ok=True)
            os.makedirs(self.local_output_directory, exist_ok=True)

    def write2disk(self) -> None:
        for idx in range(len(self)):
            print(f"Y{self.year}: writing sample {idx + 1}/{len(self)}")
            suffix: str = '__' + self.datestrings[idx]

            if self.has_global:
                global_input_filepath: str = os.path.join(self.global_input_directory, f'GI{self.year}{suffix}.pt')
                global_output_filepath: str = os.path.join(self.global_output_directory, f'GO{self.year}{suffix}.pt')
                if (not self._check_existence(global_input_filepath)) or (not self._check_existence(global_output_filepath)):
                    global_input, global_output = self._to_tensor(idx=idx, is_global=True)
                    self._clean_save(tensor=global_input, filepath=global_input_filepath)
                    self._clean_save(tensor=global_output, filepath=global_output_filepath)
                
            if self.has_local:
                local_input_filepath: str = os.path.join(self.local_input_directory, f'LI{self.year}{suffix}.pt')
                local_output_filepath: str = os.path.join(self.local_output_directory, f'LO{self.year}{suffix}.pt')
                if (not self._check_existence(local_input_filepath)) or (not self._check_existence(local_output_filepath)):
                    local_input, local_output = self._to_tensor(idx=idx, is_global=False)
                    self._clean_save(tensor=local_input, filepath=local_input_filepath)
                    self._clean_save(tensor=local_output, filepath=local_output_filepath)

    @staticmethod
    def _check_existence(filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        
        file_directory: str = os.path.dirname(filepath)
        file_sizes: List[int] = [
            os.path.getsize(os.path.join(file_directory, fname)) 
            for fname in os.listdir(file_directory)
        ]
        expected_file_size: int = max(set(file_sizes), key=file_sizes.count)  # most occurence
        return os.path.getsize(filepath) == expected_file_size

    @staticmethod
    def _clean_save(tensor: torch.Tensor, filepath=str):
        try:
            torch.save(obj=tensor, f=filepath)
        except KeyboardInterrupt:
            if os.path.exists(filepath):
                # Clean up partial file
                os.remove(filepath)                

    def _to_tensor(self, idx: int, is_global: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_global:
            surface_subset, pressure_subset, output_subset = self.global_subsets[idx]
        else:
            surface_subset, pressure_subset, output_subset = self.local_subsets[idx]

        # Process surface tensor
        surface_level_tensor: torch.Tensor = torch.tensor(surface_subset.to_dataarray().values)
        surface_level_tensor = surface_level_tensor.permute(1, 0, 2, 3)
        # Process pressure tensor
        pressure_level_tensor: torch.Tensor = torch.tensor(pressure_subset.to_dataarray().values)
        pressure_level_tensor = pressure_level_tensor.permute(1, 2, 0, 3, 4).flatten(start_dim=1, end_dim=2)
        # Merge to input tensor
        in_tensor: torch.Tensor = torch.cat(tensors=[surface_level_tensor, pressure_level_tensor], dim=1)
        assert in_tensor.ndim == 4 and in_tensor.shape == (
            self.in_timesteps, self.in_channels, in_tensor.shape[2], in_tensor.shape[3]
        )
        # Resize
        if is_global and self.global_resolution is not None:
            in_tensor = F.interpolate(input=in_tensor, size=self.global_resolution, mode='nearest')

        # Process output tensor
        out_tensor: torch.Tensor = torch.tensor(output_subset.to_dataarray().values)
        out_tensor = out_tensor.permute(1, 0, 2, 3)
        assert out_tensor.ndim == 4 and out_tensor.shape == (
            self.out_timesteps, self.out_channels, out_tensor.shape[2], out_tensor.shape[3]
        )
        # Resize
        if is_global and self.global_resolution is not None:
            out_tensor = F.interpolate(input=out_tensor, size=self.global_resolution, mode='nearest')

        return in_tensor, out_tensor

    def _prepare_subsets(
        self, 
        dataset: xr.Dataset, 
        latitude: Tuple[float, float], 
        longitude: Tuple[float, float]
    ) -> List[Tuple[xr.Dataset, xr.Dataset, xr.Dataset]]:
        selected_dataset: xr.Dataset = dataset.sel(latitude=slice(*latitude), longitude=slice(*longitude))

        # Prepare subsets which contain (input subset, output subset)
        subsets: List[Tuple[xr.Dataset, xr.Dataset]] = [
            (
                selected_dataset.isel(time=input_slice), 
                selected_dataset.isel(time=output_slice)
            )
            for input_slice, output_slice in self.slices
        ]

        # Prepare subsets which contain (surface subset, pressure subset, output subset)
        subsets: List[Tuple[xr.Dataset, xr.Dataset, xr.Dataset]] = [
            (
                input_subset[['u10', 'v10', 't2m', 'msl', 'sp']],
                input_subset.sel(isobaricInhPa=[1000, 850, 500])[['z', 'r', 't', 'u', 'v']],
                output_subset[['u10', 'v10']],
            )
            for input_subset, output_subset in subsets
        ]
        return subsets

    def _prepare_datestrings(self, subsets: List[Tuple[xr.Dataset, xr.Dataset, xr.Dataset]]) -> List[str]:
        return [
            self._numpytimes2string(input_subset.time.values) + '__' + self._numpytimes2string(output_subset.time.values) 
            for _, input_subset, output_subset in subsets
        ]

    @staticmethod
    def _numpytimes2string(timestamps: list) -> str:
        datestrings: List[str] = []
        for t in timestamps:
            datestring: str = t.astype('datetime64[s]').astype(dt.datetime).strftime('%m%d')
            if datestring not in datestrings:
                datestrings.append(datestring)
        
        return '_'.join(datestrings)

    def __len__(self):
        return len(self.slices)
    