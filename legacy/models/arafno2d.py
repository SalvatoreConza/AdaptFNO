
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AdaptiveSpectralConv2d, 
    FeatureNormalization, 
    LiftingLayer, 
    LocalLinearTransformation, 
    TemporalAggregateLayer,
)


class AutoRegressiveAdaptiveFNO2d(nn.Module):

    def __init__(
        self, 
        window_size: int,
        u_dim: int,
        width: int, 
        depth: int,
        x_modes: int, 
        y_modes: int,
    ):
        super().__init__()

        self.window_size: int = window_size
        self.u_dim: int = u_dim
        self.width: int = width
        self.depth: int = depth
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        assert width > u_dim, '`width` should be greater than `u_dim` for the model to uplift the input dim'

        self.P = LiftingLayer(in_features=u_dim, out_features=self.width)
        self.Q = LiftingLayer(in_features=self.width, out_features=u_dim)
        self.Wt = TemporalAggregateLayer(in_timesteps=window_size, out_timesteps=1)

        self.spectral_convolutions = nn.ModuleList(modules=[])
        self.local_linear_transformations = nn.ModuleList(modules=[])
        self.feature_normalizations = nn.ModuleList(modules=[])
        
        for _ in range(depth):
            self.spectral_convolutions.append(
                AdaptiveSpectralConv2d(
                    t_dim=self.window_size, u_dim=self.width, 
                    x_modes=self.x_modes, y_modes=self.y_modes
                )
            )
            self.local_linear_transformations.append(
                LocalLinearTransformation(t_dim=window_size, u_dim=width)
            )
            self.feature_normalizations.append(
                FeatureNormalization(normalized_shape=(self.window_size, self.width), dims=(1, 2))
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input dim = [batch_size, window_size, u_dim, x_res, y_res)
        batch_size: int = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]
        assert self.window_size == input.shape[1]
        assert self.u_dim == input.shape[2]

        # UpLifting
        lifted_input: torch.Tensor = self.P(input)
        assert lifted_input.shape == (batch_size, self.window_size, self.width, x_res, y_res)

        # Fourier Layers
        fourier_output: torch.Tensor = lifted_input
        for i in range(self.depth):
            # Apply spectral convolution
            spectral_conv = self.spectral_convolutions[i]
            out1: torch.Tensor = spectral_conv(fourier_output)
            # Apply local linear transformation
            local_linear_tranformation = self.local_linear_transformations[i]
            out2: torch.Tensor = local_linear_tranformation(fourier_output)
            # Connection
            assert out1.shape == out2.shape == (batch_size, self.window_size, self.width, x_res, y_res), (
                f'both out1 and out2 must have the same shape as '
                f'(batch_size, self.window_size, self.width, self.x_res, self.y_res), '
                f'got out1.shape = {out1.shape} and out2.shape = {out2.shape}'
            )
            fourier_output: torch.Tensor = out1 + out2

            # Normalize over temporal and width axes
            feature_normalization = self.feature_normalizations[i]
            fourier_output: torch.Tensor = feature_normalization(fourier_output)
            # Apply non-linearity
            fourier_output: torch.Tensor = F.gelu(fourier_output)

        # Apply temporal weights
        assert fourier_output.shape == (batch_size, self.window_size, self.width, x_res, y_res)
        fourier_output: torch.Tensor = self.Wt(fourier_output)
        assert fourier_output.shape == (batch_size, 1, self.width, x_res, y_res)
        # Projection
        projected_output: torch.Tensor = F.gelu(self.Q(fourier_output))
        assert projected_output.shape == (batch_size, 1, self.u_dim, x_res, y_res)
        return projected_output


# TEST
if __name__ == '__main__':
    from legacy.datasets.pdebench import AutoRegressiveDiffReact2d
    from torch.utils.data import DataLoader
    dataset = AutoRegressiveDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
        window_size=10
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    net = AdaptiveSpectralConv2d(t_dim=10, u_dim=2, x_modes=64, y_modes=64)

    x = next(iter(dataloader))[0]
    y = net(x)
    print(x.shape)
    print(y.shape)

