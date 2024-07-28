from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLinearTransformation(nn.Module):

    """
    Extended version of Layer R in paper: https://arxiv.org/pdf/2010.08895
    """

    def __init__(self, t_dim: int, u_dim: int, x_modes: int, y_modes: int):
        super().__init__()
        scale: float = t_dim * u_dim * u_dim * x_modes * y_modes
        weights = torch.empty(t_dim, u_dim, u_dim, x_modes, y_modes, dtype=torch.cfloat)
        nn.init.normal_(weights, mean=0.0, std=1.0 / scale)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btixy,tioxy->btoxy", input, self.weights)


class SpectralAggregateLayer(nn.Module):

    def __init__(self, x_modes: int, y_modes: int):
        super().__init__()
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        weights = torch.rand(x_modes, y_modes, dtype=torch.float)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-2:] == (self.x_modes, self.y_modes)
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
        super().__init__()

        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        All the process is conducted with adaptive x_modes and y_modes

        Parameters
        ---
        u_dim (int): number of physical fields
        """

        self.window_size: int = window_size
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.R = FrequencyLinearTransformation(t_dim=window_size, u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)
        self.Ws = SpectralAggregateLayer(x_modes=x_modes, y_modes=y_modes)
        self.Wt = TemporalAggregateLayer(t_dim=window_size)

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

        # Linear transformation
        out_linear: torch.Tensor = self.R(input=out_fft)
        assert out_linear.shape == (batch_size, self.window_size, self.u_dim, self.x_modes, self.y_modes)

        # Apply spectral weights
        out_linear: torch.Tensor = self.Ws(out_linear)
        assert out_linear.shape == (batch_size, self.window_size, self.u_dim, self.x_modes, self.y_modes)

        # Apply temporal weights
        out_linear: torch.Tensor = self.Wt(out_linear)
        assert out_linear.shape == (batch_size, 1, self.u_dim, self.x_modes, self.y_modes)

        # Inverse Fourier transform
        # out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_res, y_res))
        out_ifft: torch.Tensor = torch.fft.ifft2(out_linear, s=(x_res, y_res))
        assert out_ifft.shape == (batch_size, 1, self.u_dim, x_res, y_res)
        return out_ifft


class LiftingLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weights = nn.Parameter(
            data=torch.rand(in_features, out_features) / (in_features * out_features)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, window_size, u_dim, x_res, y_res = input.shape
        assert u_dim == self.in_features
        return torch.einsum('btixy,io->btoxy', input, self.weights)


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

        self.spectral_convolutions: nn.ModuleList = []
        self.linear_transformations: nn.ModuleList = []
        for _ in range(depth):
            self.spectral_convolutions.append(
                AutoRegressiveAdaptiveSpectralConv2d(
                    window_size=self.window_size, u_dim=self.u_dim, 
                    x_modes=self.x_modes, y_modes=self.y_modes
                )
            )
            self.linear_transformations.append(
                nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1)    
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
        output: torch.Tensor = lifted_input
        for i in range(self.depth):
            spectral_conv = self.spectral_convolutions[i]
            local_linear_tranformation = self.linear_transformations[i]
            out1: torch.Tensor = spectral_conv(output)
            out2: torch.Tensor = local_linear_tranformation(output)

            assert out1.shape == out2.shape, (
                f'both out1 and out2 must have the same shape as '
                f'(batch_size, self.window_size, self.width, self.x_res, self.x_res) ' 
                f'= {(batch_size, self.window_size, self.width, self.x_modes, self.y_modes)}'
            )
            output: torch.Tensor = out1 + out2
            output: torch.Tensor = F.gelu(output)

        # Projection
        projected_output: torch.Tensor = F.gelu(self.Q(output))
        assert projected_output.shape == input.shape

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

