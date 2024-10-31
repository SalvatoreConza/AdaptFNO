from typing import List, Tuple, Literal

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from common.functional import compute_velocity_field


class TemporalMSE(nn.Module):

    def __init__(self, n_timesteps: int, reduction: str):
        super().__init__()
        self.n_timesteps: int = n_timesteps
        self.reduction: str = reduction
        self.loss_function = nn.MSELoss(reduction='none').cuda()
        # self.temporal_weights: nn.Parameter = self._linearly_decayed_weights()
        self.temporal_weights: nn.Parameter = self._exponentially_decayed_weights(decay_rate=0.2).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape
        assert input.ndim == 5
        weighted_loss: torch.Tensor = self.loss_function(input=input, target=target) * self.temporal_weights
        if self.reduction == 'mean':
            weighted_loss = weighted_loss.mean()
        elif self.reduction == 'sum':
            weighted_loss = weighted_loss.sum()
        
        return weighted_loss

    def _linearly_decayed_weights(self) -> nn.Parameter:
        temporal_weights: torch.Tensor = torch.arange(
            start=self.n_timesteps, end=0, step=-1, 
            requires_grad=False,
        )
        temporal_weights = temporal_weights / temporal_weights.sum() * self.n_timesteps # weight_sum == n_timesteps, not 1
        return nn.Parameter(temporal_weights.reshape(1, self.n_timesteps, 1, 1, 1), requires_grad=False)

    def _exponentially_decayed_weights(self, decay_rate: float) -> nn.Parameter:
        # higher decay_rate leads to faster decay
        temporal_weights: torch.Tensor = torch.exp(-decay_rate * torch.arange(self.n_timesteps))
        temporal_weights = temporal_weights / temporal_weights.sum() * self.n_timesteps # weight_sum == n_timesteps, not 1
        return nn.Parameter(data=temporal_weights.reshape(1, self.n_timesteps, 1, 1, 1), requires_grad=False)
