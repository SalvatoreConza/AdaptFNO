import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpectralConv2d(nn.Module):

    def __init__(
        self, 
        u_dim: int,
        x_modes: int,
        y_modes: int,
    ):
        super().__init__()

        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        All the process is conducted with adaptive x_modes and y_modes
        """

        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.scale: float = 1 / (u_dim * u_dim)
        self.weights_R = nn.Parameter(
            data=self.scale * torch.rand(u_dim, u_dim, self.x_modes, self.y_modes, dtype=torch.cfloat)
        )

    def R(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, self.weights_R)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4, 'Expected input of shape: (batch_size, self.u_dim, x_modes, y_modes)'
        assert input.shape[1] == self.u_dim, 'input.shape[1] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_res: int = input.shape[2]
        y_res: int = input.shape[3]

        assert x_res >= self.x_modes, f'x_res={input.shape[2]} must greater or equal to self.x_modes={self.x_modes}'
        assert y_res >= self.y_modes, f'y_res={input.shape[3]} must greater or equal to self.y_modes={self.y_modes}'

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.fft2(input)
        assert out_fft.shape == (batch_size, self.u_dim, x_res, y_res)
        
        # Truncate max x_modes, y_modes
        out_fft: torch.Tensor = out_fft[:, :, :self.x_modes, :self.y_modes]

        # Linear transformation
        out_linear: torch.Tensor= self.R(input=out_fft)
        assert out_fft.shape == (batch_size, self.u_dim, self.x_modes, self.y_modes)

        # Apply spectral weights
        spectral_weights: torch.Tensor = self.compute_spectral_weights(coeffs=self.weights_R)
        assert spectral_weights.shape == (self.x_modes, self.y_modes)
        # Note: `spectral_weights` will be broadcasted to (batch_size, self.u_dim, self.x_modes, self.y_modes)
        out_linear: torch.Tensor = spectral_weights * out_linear
        assert out_linear.shape == (batch_size, self.u_dim, self.x_modes, self.y_modes)

        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_res, y_res))
        return out_ifft

    def compute_spectral_weights(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.norm(coeffs, p='fro', dim=(0, 1))


class AdaptiveFNO2d(nn.Module):

    def __init__(
        self, 
        u_dim: int,
        width: int, 
        x_modes: int, 
        y_modes: int,
    ):
        super().__init__()

        self.u_dim: int = u_dim
        self.width: int = width
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        assert width > u_dim, '`width` should be greater than `u_dim` for the model to uplift the input dim'

        self.P = nn.Linear(in_features=u_dim, out_features=self.width)
        self.Q = nn.Linear(in_features=self.width, out_features=u_dim)

        # Fourier Layer 0
        self.spectral_conv0 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_modes=self.x_modes,
            y_modes=self.y_modes,
        )
        self.W0 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width,
            kernel_size=1,
        )
        # Fourier Layer 1
        self.spectral_conv1 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_modes=self.x_modes,
            y_modes=self.y_modes,
        )
        self.W1 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width,
            kernel_size=1,
        )
        # Fourier Layer 2
        self.spectral_conv2 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_modes=self.x_modes,
            y_modes=self.y_modes,
        )
        self.W2 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width,
            kernel_size=1,
        )
        # Fourier Layer 3
        self.spectral_conv3 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_modes=self.x_modes,
            y_modes=self.y_modes,
        )
        self.W3 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width,
            kernel_size=1,
        )
        # Fourier Layer 4
        self.spectral_conv4 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_modes=self.x_modes,
            y_modes=self.y_modes,
        )
        self.W4 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width,
            kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input dim = [batch_size, u_dim, x_res, y_res)
        batch_size: int = input.shape[0]
        u_dim: int = input.shape[1]
        x_res: int = input.shape[2]
        y_res: int = input.shape[3]
        
        # Uplifting
        input: torch.Tensor = input.permute(0, 2, 3, 1)
        lifted_input: torch.Tensor = self.P(input)
        lifted_input: torch.Tensor = lifted_input.permute(0, 3, 1, 2)
        assert lifted_input.shape == (batch_size, self.width, x_res, y_res)
        
        # Block 0
        out1: torch.Tensor = self.spectral_conv0(lifted_input)
        out2: torch.Tensor = self.W0(lifted_input)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, self.x_modes, self.y_modes) ' 
            f'= {(batch_size, self.width, self.x_modes, self.y_modes)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 1
        out1: torch.Tensor = self.spectral_conv1(out)
        out2: torch.Tensor = self.W1(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, self.x_modes, self.y_modes) ' 
            f'= {(batch_size, self.width, self.x_modes, self.y_modes)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 2
        out1: torch.Tensor = self.spectral_conv2(out)
        out2: torch.Tensor = self.W2(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, self.x_modes, self.y_modes) ' 
            f'= {(batch_size, self.width, self.x_modes, self.y_modes)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 3
        out1: torch.Tensor = self.spectral_conv3(out)
        out2: torch.Tensor = self.W3(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, self.x_modes, self.y_modes) ' 
            f'= {(batch_size, self.width, self.x_modes, self.y_modes)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 4
        out1: torch.Tensor = self.spectral_conv4(out)
        out2: torch.Tensor = self.W4(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, self.x_modes, self.y_modes) ' 
            f'= {(batch_size, self.width, self.x_modes, self.y_modes)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        out: torch.Tensor = out.permute(0, 2, 3, 1)
        projected_output: torch.Tensor = self.Q(out)
        projected_output: torch.Tensor = F.gelu(projected_output)
        projected_output: torch.Tensor = projected_output.permute(0, 3, 1, 2)
        return projected_output


