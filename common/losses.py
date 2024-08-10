from typing import List, Tuple

import torch
import torch.nn as nn


class RegularizedPowerError(nn.Module):

    def __init__(self, lambda_: float, power: int = 2):
        super().__init__()
        self.power: int = power
        self.lambda_: float = lambda_

    def forward(
        self, 
        spectral_weights: List[torch.Tensor],
        prediction: torch.Tensor, 
        groundtruth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        power_error: torch.Tensor = (
            (prediction - groundtruth).abs() ** self.power
        ).mean()
        scaled_error: torch.Tensor = power_error ** (1 / self.power)
        weight_magnitude: torch.Tensor = torch.stack(
            tensors=spectral_weights, dim=0
        ).abs().mean()
        return scaled_error, power_error, weight_magnitude, power_error + self.lambda_ * weight_magnitude

