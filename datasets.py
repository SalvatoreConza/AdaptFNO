from typing import Tuple, Optional

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
        device: torch.device = 'cuda', 
    ):
        self.dataroot: str = dataroot
        self.input_step: int = input_step
        self.target_step: int = target_step
        self.resolution: Optional[Tuple[int, int]] = resolution
        self.device: torch.device = device
        self.file = h5py.File(name=dataroot, mode='r')
        self.num_samples = len(self.file.keys())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key: str = f'{str(idx).zfill(4)}/data'
        data: np.ndarray = np.array(self.file[key])
        data: torch.Tensor = torch.tensor(data, device=self.device).permute(0, 3, 1, 2)
        
        if self.resolution is not None:
            data: torch.Tensor = self.resize_tensor(tensor=data)
        
        input: torch.Tensor = data[self.input_step]
        target: torch.Tensor = data[self.target_step]
        return input, target

    def __len__(self) -> int:
        return self.num_samples
    
    def __del__(self):
        self.file.close()

    def resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize a 4D tensor to the given size (height, width)."""
        return F.interpolate(tensor, size=self.resolution, mode='bilinear', align_corners=False)


if __name__ == '__main__':
    dataset = OneShotDiffReact2d(dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    inputs, targets = next(iter(dataloader))





