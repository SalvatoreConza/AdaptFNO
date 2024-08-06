from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLinearTransformation(nn.Module):

    """
    Extended version of Layer R in paper: https://arxiv.org/pdf/2010.08895
    """

    def __init__(self, t_dim: int, u_dim: int, x_modes: int, y_modes: int):
        super().__init__()
        self.t_dim: int = t_dim
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        weights = torch.empty(t_dim, u_dim, u_dim, x_modes, y_modes, dtype=torch.cfloat)
        nn.init.normal_(weights, mean=0.0, std=0.01)
        self.weights = nn.Parameter(data=weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, t_dim, u_dim, x_modes, y_modes = input.shape
        output: torch.Tensor = torch.einsum("btixy,tioxy->btoxy", input, self.weights)
        assert output.shape == input.shape == (batch_size, self.t_dim, self.u_dim, self.x_modes, self.y_modes)
        return output


class SpectralAggregateLayer(nn.Module):

    def __init__(self, u_dim: int, x_modes: int, y_modes: int):
        super().__init__()
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        weights: torch.Tensor = torch.empty(u_dim, x_modes, y_modes, dtype=torch.float)
        nn.init.normal_(weights, mean=0.0, std=1.0 / 0.01)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-3:] == (self.u_dim, self.x_modes, self.y_modes)
        # self.weights is broadcasted to input.shape
        output: torch.Tensor = input * self.weights
        assert output.shape == input.shape
        return output


class TemporalAggregateLayer(nn.Module):

    def __init__(self, t_dim: int):
        super().__init__()
        self.t_dim: int = t_dim
        weights = torch.rand(1, t_dim, 1, 1, 1, dtype=torch.float)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[1] == self.t_dim
        batch_size, t_dim, u_dim, x_modes, y_modes = input.shape
        # self.weights is broadcasted to input shape
        output: torch.Tensor = input * self.weights / self.weights.sum()
        output: torch.Tensor = output.sum(dim=1, keepdim=True)
        assert output.shape == (batch_size, 1, u_dim, x_modes, y_modes)
        return output


class AutoRegressiveAdaptiveSpectralConv2d(nn.Module):

    def __init__(
        self, 
        window_size: int,
        u_dim: int,
        x_modes: int,
        y_modes: int,
    ):
        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        All the process is conducted with adaptive x_modes and y_modes
        """
        super().__init__()

        self.window_size: int = window_size
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.R = FrequencyLinearTransformation(t_dim=window_size, u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)
        self.Ws = SpectralAggregateLayer(u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, 'Expected input of shape: (batch_size, window_size, self.u_dim, x_modes, y_modes)'
        assert input.shape[1] == self.window_size, 'input.shape[1] must match self.window_size'
        assert input.shape[2] == self.u_dim, 'input.shape[2] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]

        assert x_res >= self.x_modes, f'x_res={input.shape[2]} must greater or equal to self.x_modes={self.x_modes}'
        assert y_res >= self.y_modes, f'y_res={input.shape[3]} must greater or equal to self.y_modes={self.y_modes}'

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.fft2(input)
        assert out_fft.shape == (batch_size, self.window_size, self.u_dim, x_res, y_res)
        
        # Truncate max x_modes, y_modes
        out_fft: torch.Tensor = out_fft[:, :, :, :self.x_modes, :self.y_modes]

        # Apply spectral weights
        out_fft: torch.Tensor = self.Ws(out_fft)
        assert out_fft.shape == (batch_size, self.window_size, self.u_dim, self.x_modes, self.y_modes)

        # Linear transformation
        out_linear: torch.Tensor = self.R(input=out_fft)
        assert out_linear.shape == (batch_size, self.window_size, self.u_dim, self.x_modes, self.y_modes)

        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_res, y_res))
        assert out_ifft.shape == (batch_size, self.window_size, self.u_dim, x_res, y_res)
        return out_ifft


class LocalLinearTransformation(nn.Module):

    def __init__(self, window_size: int, u_dim: int):
        super().__init__()
        self.window_size: int = window_size
        self.u_dim: int = u_dim
        self.linear_transformation: nn.Module = nn.Conv2d(
            in_channels=window_size * u_dim, out_channels=window_size * u_dim, kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]
        assert input.shape == (batch_size, self.window_size, self.u_dim, x_res, y_res)
        input: torch.Tensor = input.reshape(batch_size, self.window_size * self.u_dim, x_res, y_res)
        output: torch.Tensor = self.linear_transformation(input=input)
        output: torch.Tensor = output.reshape(batch_size, self.window_size, self.u_dim, x_res, y_res)
        return output


class LiftingLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weights = nn.Parameter(
            data=torch.rand(in_features, out_features) / (in_features * out_features)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, window_size, u_dim, x_res, y_res = input.shape
        assert self.in_features == u_dim
        output: torch.Tensor = torch.einsum('btixy,io->btoxy', input, self.weights)
        assert output.shape == (batch_size, window_size, self.out_features, x_res, y_res)
        return output


class LayerNormOnDims(nn.Module):
    def __init__(self, normalized_shape , dims):
        super(LayerNormOnDims, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dims = dims

    def forward(self, x):
        # x is of shape (batch_size, self.window_size, self.width, self.x_res, self.x_res)
        original_shape = x.shape
        
        # Move the dimensions to be normalized to the end
        permute_order = [d for d in range(len(original_shape)) if d not in self.dims] + list(self.dims)
        x_permuted = x.permute(*permute_order)
        
        # Apply LayerNorm on the specified dimensions
        x_normalized = self.layer_norm(x_permuted)
        
        # Permute back to the original shape
        inverse_permute_order = [permute_order.index(i) for i in range(len(original_shape))]
        x_out = x_normalized.permute(*inverse_permute_order)
        
        return x_out


class FeatureNormalization(nn.Module):

    def __init__(self, normalized_shape: Tuple[int, ...], dims: Tuple[int, ...]):
        super().__init__()
        assert len(normalized_shape) == len(dims)

        self.dims: Tuple[int, ...] = dims
        self.normalized_shape: Tuple[int, ...] = normalized_shape
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim >= len(self.dims)
        original_shape = input.shape
        # Move normalized dimensions to the end
        permute_order: List[int] = [d for d in range(len(original_shape)) if d not in self.dims] + list(self.dims)
        input: torch.Tensor = input.permute(*permute_order)
        # Apply LayerNorm on the last dimensions
        output: torch.Tensor = self.layer_norm(input)
        # Permute back to the original shape
        inverse_permute_order: List[int] = [permute_order.index(i) for i in range(len(original_shape))]
        output: torch.Tensor = output.permute(*inverse_permute_order)
        assert output.shape == original_shape
        return output


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
        self.Wt = TemporalAggregateLayer(t_dim=window_size)

        self.spectral_convolutions = nn.ModuleList(modules=[])
        self.local_linear_transformations = nn.ModuleList(modules=[])
        self.feature_normalizations = nn.ModuleList(modules=[])
        
        for _ in range(depth):
            self.spectral_convolutions.append(
                AutoRegressiveAdaptiveSpectralConv2d(
                    window_size=self.window_size, u_dim=self.width, 
                    x_modes=self.x_modes, y_modes=self.y_modes
                )
            )
            self.local_linear_transformations.append(
                LocalLinearTransformation(window_size=window_size, u_dim=width)
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
    from datasets import AutoRegressiveDiffReact2d
    from torch.utils.data import DataLoader
    dataset = AutoRegressiveDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
        window_size=10
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    net = AutoRegressiveAdaptiveSpectralConv2d(window_size=10, u_dim=2, x_modes=64, y_modes=64)

    x = next(iter(dataloader))[0]
    y = net(x)
    print(x.shape)
    print(y.shape)

