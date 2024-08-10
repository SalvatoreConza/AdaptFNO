from typing import Tuple, List, Dict, Optional

import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class OneShotDiffReact2d(Dataset):

    def __init__(
        self, 
        dataroot: str, 
        input_step: int = 0, 
        target_step: int = -1, 
        resolution: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.dataroot: str = dataroot
        self.input_step: int = input_step
        self.target_step: int = target_step
        self.resolution: Optional[Tuple[int, int]] = resolution
        self.file = h5py.File(name=dataroot, mode='r')
        self.n_samples = len(self.file.keys())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key: str = f'{str(idx).zfill(4)}/data'
        data: np.ndarray = np.array(self.file[key])
        data: torch.Tensor = torch.tensor(data).permute(0, 3, 1, 2)
        
        if self.resolution is not None:
            data: torch.Tensor = self.resize_tensor(tensor=data)
        
        input: torch.Tensor = data[self.input_step]
        target: torch.Tensor = data[self.target_step]
        return input, target

    def __len__(self) -> int:
        return self.n_samples
    
    def __del__(self):
        self.file.close()

    def resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize a 4D tensor to the given size (height, width)."""
        return F.interpolate(tensor, size=self.resolution, mode='bilinear', align_corners=False)


class AutoRegressiveDiffReact2d(Dataset):

    def __init__(
        self,
        dataroot: str,
        window_size: int,
        from_sample: int,
        to_sample: int,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.dataroot: str = dataroot
        self.window_size: int = window_size
        self.from_sample: int = from_sample
        self.to_sample: int = to_sample
        self.resolution: Optional[Tuple[int, int]] = resolution
        
        self.file = h5py.File(name=dataroot, mode='r')
        self.n_samples = to_sample - from_sample + 1

        self.indices: List[Tuple[int, int]] = []
        for sample_index in range(from_sample, to_sample + 1):
            for target_timestep in range(self.window_size, 101):
                self.indices.append((sample_index, target_timestep))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_index, target_timestep = self.indices[idx]
        # find sample index and target timestep
        key: str = f'{str(sample_index).zfill(4)}/data'
        input_timesteps = slice(target_timestep - self.window_size, target_timestep, 1)
        # get full array
        data: np.ndarray = np.array(self.file[key])
        # get the input tensor
        input: torch.Tensor = torch.tensor(data=data[input_timesteps]).permute(0, 3, 1, 2)
        # get the target tensor
        target: torch.Tensor = torch.tensor(data=data[[target_timestep]]).permute(0, 3, 1, 2)
        # resize tensor        
        if self.resolution is not None:
            input = self.resize_tensor(input)
            target = self.resize_tensor(target)

        return input, target

    def __len__(self) -> int:
        return len(self.indices)

    def __del__(self):
        self.file.close()

    def resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize a 4D tensor to the given size (height, width)."""
        return F.interpolate(tensor, size=self.resolution, mode='bilinear', align_corners=False)


class MultiStepDiffReact2d(Dataset):

    def __init__(
        self,
        dataroot: str,
        input_timesteps: List[int],
        target_timestep: int,
        from_sample: int,
        to_sample: int,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert all(input_timesteps[i] == input_timesteps[i - 1] + 1 for i in range(1, len(input_timesteps)))
        assert target_timestep > input_timesteps[-1] - 1, "target_timestep must greater than input's last timestep"

        self.dataroot: str = dataroot
        self.input_timesteps: List[int] = input_timesteps
        self.target_timestep: int = target_timestep
        self.n_prediction_steps: int = target_timestep - input_timesteps[-1]

        self.from_sample: int = from_sample
        self.to_sample: int = to_sample

        self.resolution: Optional[Tuple[int, int]] = resolution
        self.file = h5py.File(name=dataroot, mode='r')

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key: str = f'{str(self.from_sample + idx).zfill(4)}/data'
        # get full array
        data: np.ndarray = np.array(self.file[key])
        # get the input tensor
        input: torch.Tensor = torch.tensor(data=data[self.input_timesteps]).permute(0, 3, 1, 2)
        # get the target tensor
        target: torch.Tensor = torch.tensor(data=data[[self.target_timestep]]).permute(0, 3, 1, 2)
        # resize tensor        
        if self.resolution is not None:
            input = self.resize_tensor(input)
            target = self.resize_tensor(target)

        return input, target

    def __len__(self) -> int:
        return self.to_sample - self.from_sample + 1

    def __del__(self):
        self.file.close()

    def resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize a 4D tensor to the given size (height, width)."""
        return F.interpolate(tensor, size=self.resolution, mode='bilinear', align_corners=False)



# TEST
if __name__ == '__main__':

    self = AutoRegressiveDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
        window_size=5,
        from_sample=0,
        to_sample=979,
        resolution=None,
    )
    self = MultiStepDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
        input_timesteps=list(range(5)),
        target_timestep=100,
        from_sample=980,
        to_sample=999,
        resolution=None,
    )




