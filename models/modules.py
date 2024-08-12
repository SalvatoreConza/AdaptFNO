from typing import List, Tuple
import torch
import torch.nn as nn


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
        nn.init.kaiming_normal_(weights)
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
        nn.init.kaiming_normal_(weights)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-3:] == (self.u_dim, self.x_modes, self.y_modes)
        # self.weights is broadcasted to input.shape
        output: torch.Tensor = input * self.weights
        assert output.shape == input.shape
        return output


class TemporalAggregateLayer(nn.Module):

    def __init__(self, in_timesteps: int, out_timesteps: int):
        super().__init__()
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        weights = torch.rand(in_timesteps, out_timesteps, dtype=torch.float)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[1] == self.in_timesteps
        batch_size, in_timesteps, u_dim, x_modes, y_modes = input.shape
        output: torch.Tensor = torch.einsum("biuxy,io->bouxy", input, self.weights)
        weight_sum: torch.Tensor = self.weights.sum(dim=0).reshape(1, self.out_timesteps, 1, 1, 1)
        output: torch.Tensor = output / weight_sum  # broadcasted to input shape
        assert output.shape == (batch_size, self.out_timesteps, u_dim, x_modes, y_modes)
        return output


class AdaptiveSpectralConv2d(nn.Module):

    def __init__(
        self,
        t_dim: int,
        u_dim: int,
        x_modes: int,
        y_modes: int,
    ):
        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        All the process is conducted with adaptive x_modes and y_modes
        """
        super().__init__()

        self.t_dim: int = t_dim
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.R = FrequencyLinearTransformation(t_dim=t_dim, u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)
        self.Ws = SpectralAggregateLayer(u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, 'Expected input of shape: (batch_size, t_dim, self.u_dim, x_modes, y_modes)'
        assert input.shape[1] == self.t_dim, 'input.shape[1] must match self.t_dim'
        assert input.shape[2] == self.u_dim, 'input.shape[2] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]

        assert x_res >= self.x_modes, f'x_res={input.shape[2]} must greater or equal to self.x_modes={self.x_modes}'
        assert y_res >= self.y_modes, f'y_res={input.shape[3]} must greater or equal to self.y_modes={self.y_modes}'

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.fft2(input)
        assert out_fft.shape == (batch_size, self.t_dim, self.u_dim, x_res, y_res)

        # Truncate max x_modes, y_modes
        out_fft: torch.Tensor = out_fft[:, :, :, :self.x_modes, :self.y_modes]

        # Apply spectral weights
        out_fft: torch.Tensor = self.Ws(out_fft)
        assert out_fft.shape == (batch_size, self.t_dim, self.u_dim, self.x_modes, self.y_modes)

        # Linear transformation
        out_linear: torch.Tensor = self.R(input=out_fft)
        assert out_linear.shape == (batch_size, self.t_dim, self.u_dim, self.x_modes, self.y_modes)

        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_res, y_res))
        assert out_ifft.shape == (batch_size, self.t_dim, self.u_dim, x_res, y_res)
        return out_ifft


class LocalLinearTransformation(nn.Module):

    def __init__(self, t_dim: int, u_dim: int):
        super().__init__()
        self.t_dim: int = t_dim
        self.u_dim: int = u_dim
        self.linear_transformation: nn.Module = nn.Conv2d(
            in_channels=t_dim * u_dim, out_channels=t_dim * u_dim, kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]
        assert input.shape == (batch_size, self.t_dim, self.u_dim, x_res, y_res)
        input: torch.Tensor = input.reshape(batch_size, self.t_dim * self.u_dim, x_res, y_res)
        output: torch.Tensor = self.linear_transformation(input=input)
        output: torch.Tensor = output.reshape(batch_size, self.t_dim, self.u_dim, x_res, y_res)
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
        batch_size, t_dim, u_dim, x_res, y_res = input.shape
        assert self.in_features == u_dim
        output: torch.Tensor = torch.einsum('btixy,io->btoxy', input, self.weights)
        assert output.shape == (batch_size, t_dim, self.out_features, x_res, y_res)
        return output


class LayerNormOnDims(nn.Module):
    def __init__(self, normalized_shape , dims):
        super(LayerNormOnDims, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dims = dims

    def forward(self, x):
        # x is of shape (batch_size, self.t_dim, self.width, self.x_res, self.x_res)
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
    

