import torch
import torch.nn as nn


class MeanPowerError(nn.Module):

    def __init__(self, p: int):
        super().__init__()
        self.p: int = p

    def forward(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(prediction - groundtruth) ** self.p)

