from typing import List, Tuple, Literal

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from common.functional import compute_velocity_field


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


class VGG16Loss(nn.Module):

    def __init__(self, reduction: Literal['sum', 'mean']):
        super().__init__()
        self.reduction: str = reduction
        self.feature_extractor: nn.Module = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.ndim == target.ndim == 5
        assert input.shape == target.shape
        batch_size, timesteps, u_dim, x_res, y_res = input.shape
        input = compute_velocity_field(input=input, dim=2)
        target = compute_velocity_field(input=target, dim=2)
        input = input.reshape(batch_size * timesteps, 1, x_res, y_res)
        target = target.reshape(batch_size * timesteps, 1, x_res, y_res)
        input = input.expand(batch_size * timesteps, 3, x_res, y_res)
        target = target.expand(batch_size * timesteps, 3, x_res, y_res)
        prediction_features: torch.Tensor = self.feature_extractor(input=input)
        groundtruth_features: torch.Tensor = self.feature_extractor(input=target)
        total_loss: torch.Tensor = self.loss_function(input=prediction_features, target=groundtruth_features)
        if self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / batch_size


class TemporalMSE(nn.Module):

    def __init__(self, n_timesteps: int, reduction: str):
        super().__init__()
        self.n_timesteps: int = n_timesteps
        self.reduction: str = reduction
        self.loss_function = nn.MSELoss(reduction='none')
        # self.temporal_weights: nn.Parameter = self._linearly_decayed_weights()
        self.temporal_weights: nn.Parameter = self._exponentially_decayed_weights(decay_rate=0.2)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape
        assert input.ndim == 5
        weighted_loss: torch.Tensor = (
            self.loss_function(input=input, target=target) * self.temporal_weights.to(device=input.device)
        )
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
