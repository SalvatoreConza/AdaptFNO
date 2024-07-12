from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reference:

[1] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). 
    Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

"""


class SpectralConv2d(nn.Module):

    def __init__(
        self, 
        u_dim: int,
        x_modes: int, 
        y_modes: int,
    ):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        Parameters
        ---
        u_dim (int): number of physical fields
        x_modes (int): number of Fourier waves in first spatial dimension (x)
        y_modes (int): number of Fourier waves in second spatial dimension (y)
        """

        self.u_dim: int = u_dim

        # expected input spatial dimension: (M, N)
        self.x_modes: int = x_modes   # must be less than M
        self.y_modes: int = y_modes   # must be less than floor(N/2) + 1 due to the use of torch.rfft2

        self.scale: float = 1 / (u_dim * u_dim)
        self.weights_R = nn.Parameter(
            data=self.scale * torch.rand(u_dim, u_dim, x_modes, y_modes, dtype=torch.cfloat)
        )

    def R(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, self.weights_R)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4, 'Expected input of shape: (batch_size, self.u_dim, x_dim, y_dim)'
        assert input.shape[1] == self.u_dim, 'input.shape[1] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_dim: int = input.shape[2]
        y_dim: int = input.shape[3]
        out_y_dim: int = y_dim // 2 + 1

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.rfft2(input)
        assert out_fft.shape == (batch_size, self.u_dim, x_dim, out_y_dim)

        # Linear transformation
        out_linear: torch.Tensor= self.R(
            input=out_fft[:, :, :self.x_modes, :self.y_modes],
        )
        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_dim, y_dim))
        return out_ifft


class FNO2d(nn.Module):

    def __init__(
        self, 
        u_dim: int, 
        x_modes: int, 
        y_modes: int, 
        width: int, 
    ):
        super().__init__()

        """
        Parameters:
        ---

        u_dim (int): dim of the physic field
        x_modes (int): number of Fourier waves in the first dimension
        y_modes (int): number of Fourier waves in the second dimension
        width (int): 

        """

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.P .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batch_size, x, y, c)
        output: the solution of the next timestep
        output shape: (batch_size, x, y, c)
        """

        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        self.width: int = width

        assert width > u_dim, '`width` should be greater than `u_dim` for the model to uplift the input dim'

        self.P = nn.Linear(in_features=u_dim, out_features=self.width)
        self.Q = nn.Linear(in_features=self.width, out_features=u_dim)

        # Fourier Layer 0
        self.spectral_conv0 = SpectralConv2d(
            u_dim=self.width,
            x_modes=self.x_modes, y_modes=self.y_modes,
        )
        self.W0 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 1
        self.spectral_conv1 = SpectralConv2d(
            u_dim=self.width,
            x_modes=self.x_modes, y_modes=self.y_modes,
        )
        self.W1 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 2
        self.spectral_conv2 = SpectralConv2d(
            u_dim=self.width,
            x_modes=self.x_modes, y_modes=self.y_modes,
        )
        self.W2 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 3
        self.spectral_conv3 = SpectralConv2d(
            u_dim=self.width,
            x_modes=self.x_modes, y_modes=self.y_modes,
        )
        self.W3 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input dim = [batch_size, u_dim, x_dim, y_dim)
        batch_size: int = input.shape[0]
        u_dim: int = input.shape[1]
        x_dim: int = input.shape[2]
        y_dim: int = input.shape[3]
        
        # Uplifting
        input: torch.Tensor = input.permute(0, 2, 3, 1)
        lifted_input: torch.Tensor = self.P(input)
        lifted_input: torch.Tensor = lifted_input.permute(0, 3, 1, 2)
        assert lifted_input.shape[1] == self.width
        
        # Block 0
        out1: torch.Tensor = self.spectral_conv0(lifted_input)
        out2: torch.Tensor = self.W0(lifted_input)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 1
        out1: torch.Tensor = self.spectral_conv1(out)
        out2: torch.Tensor = self.W1(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 2
        out1 = self.spectral_conv2(out)
        out2 = self.W2(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 3
        out1 = self.spectral_conv3(out)
        out2 = self.W3(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        out = out.permute(0, 2, 3, 1)
        projected_output: torch.Tensor = self.Q(out)
        projected_output: torch.Tensor = F.gelu(projected_output)
        projected_output: torch.Tensor = projected_output.permute(0, 3, 1, 2)
        return projected_output
    

class AdaptiveSpectralConv2d(nn.Module):

    def __init__(
        self, 
        u_dim: int,
        x_dim: int,
        y_dim: int,
        min_explanation: float
    ):
        super().__init__()

        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        All the process is conducted with adaptive x_modes and y_modes

        Parameters
        ---
        u_dim (int): number of physical fields
        """

        self.u_dim: int = u_dim
        self.x_dim: int = x_dim
        self.y_dim: int = y_dim
        self.out_y_dim: int = y_dim // 2 + 1

        assert 0 < min_explanation <= 1
        self.min_explanation: float = min_explanation

        self.scale: float = 1 / (u_dim * u_dim)
        self.weights_R = nn.Parameter(
            data=self.scale * torch.rand(u_dim, u_dim, self.x_dim, self.out_y_dim, dtype=torch.cfloat)
        )

    def R(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, self.weights_R)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4, 'Expected input of shape: (batch_size, self.u_dim, x_dim, y_dim)'
        assert input.shape[1] == self.u_dim, 'input.shape[1] must match self.u_dim'
        assert input.shape[2] == self.x_dim, 'input.shape[2] must match self.x_dim'
        assert input.shape[3] == self.y_dim, 'input.shape[3] must match self.y_dim'
        
        batch_size: int = input.shape[0]

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.rfft2(input)
        assert out_fft.shape == (batch_size, self.u_dim, self.x_dim, self.out_y_dim)

        # Linear transformation
        out_linear: torch.Tensor= self.R(input=out_fft)
        assert out_fft.shape == (batch_size, self.u_dim, self.x_dim, self.out_y_dim)

        # Compute modes
        with torch.no_grad():
            x_modes, y_modes = self.compute_modes(coeffs=self.weights_R)

        # Truncate spectrum:
        out_linear: torch.Tensor = out_linear[:, :, :x_modes, :y_modes]

        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(self.x_dim, self.y_dim))
        return out_ifft

    #subject to change
    def compute_modes(self, coeffs: torch.Tensor) -> Tuple[int, int]:
        strength_matrix: torch.Tensor = torch.norm(coeffs, p='fro', dim=(0, 1))
        assert strength_matrix.shape == (self.x_dim, self.out_y_dim)

        cumulative_strength_matrix: torch.Tensor = torch.cumsum(
            input=torch.cumsum(strength_matrix, dim=0), 
            dim=1,
        )
        explanation_ratio_matrix: torch.Tensor = (
            cumulative_strength_matrix / strength_matrix.sum()
        )

        for i in range(explanation_ratio_matrix.shape[0]):
            for j in range(explanation_ratio_matrix.shape[1]):
                if explanation_ratio_matrix[i, j] >= self.min_explanation:
                    return i, j

        raise RuntimeError(f'Cannot find mode in explanation ratio matrix {explanation_ratio_matrix}')


class AdaptiveFNO2d(nn.Module):

    def __init__(
       self, 
        u_dim: int,
        x_dim: int,
        y_dim: int,
        width: int, 
        min_explanation: float,
    ):
        super().__init__()

        self.u_dim: int = u_dim
        self.x_dim: int = x_dim
        self.y_dim: int = y_dim
        self.width: int = width
        self.min_explanation: float = min_explanation

        assert width > u_dim, '`width` should be greater than `u_dim` for the model to uplift the input dim'

        self.P = nn.Linear(in_features=u_dim, out_features=self.width)
        self.Q = nn.Linear(in_features=self.width, out_features=u_dim)

        # Fourier Layer 0
        self.spectral_conv0 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_dim=x_dim,
            y_dim=y_dim,
            min_explanation=min_explanation
        )
        self.W0 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 1
        self.spectral_conv1 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_dim=x_dim,
            y_dim=y_dim,
            min_explanation=min_explanation
        )
        self.W1 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 2
        self.spectral_conv2 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_dim=x_dim,
            y_dim=y_dim,
            min_explanation=min_explanation
        )
        self.W2 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )
        # Fourier Layer 3
        self.spectral_conv3 = AdaptiveSpectralConv2d(
            u_dim=width,
            x_dim=x_dim,
            y_dim=y_dim,
            min_explanation=min_explanation
        )
        self.W3 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, 
            kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input dim = [batch_size, u_dim, x_dim, y_dim)
        batch_size: int = input.shape[0]
        u_dim: int = input.shape[1]
        x_dim: int = input.shape[2]
        y_dim: int = input.shape[3]
        
        # Uplifting
        input: torch.Tensor = input.permute(0, 2, 3, 1)
        lifted_input: torch.Tensor = self.P(input)
        lifted_input: torch.Tensor = lifted_input.permute(0, 3, 1, 2)
        assert lifted_input.shape[1] == self.width
        
        # Block 0
        out1: torch.Tensor = self.spectral_conv0(lifted_input)
        out2: torch.Tensor = self.W0(lifted_input)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 1
        out1: torch.Tensor = self.spectral_conv1(out)
        out2: torch.Tensor = self.W1(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 2
        out1 = self.spectral_conv2(out)
        out2 = self.W2(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 3
        out1 = self.spectral_conv3(out)
        out2 = self.W3(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_dim, y_dim) ' 
            f'= {(batch_size, self.width, x_dim, y_dim)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        out = out.permute(0, 2, 3, 1)
        projected_output: torch.Tensor = self.Q(out)
        projected_output: torch.Tensor = F.gelu(projected_output)
        projected_output: torch.Tensor = projected_output.permute(0, 3, 1, 2)
        return projected_output


# under development
class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_blocks: int = 8, 
        sparsity_threshold: float = 0.01, 
        hard_thresholding_fraction: float = 1., 
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size_factor = hidden_size_factor

        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        self.w1 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size, self.block_size * hidden_size_factor)
        )
        self.b1 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size)
        )

    def forward(
        self, 
        input: torch.Tensor, 
    ):
        
        assert input.ndim == 4
        batch_size: int = input.shape[0]
        x_dim: int = input.shape[1]
        y_dim: int = input.shape[2]
        u_dim: int = input.shape[3]

        bias: torch.Tensor = x
        dtype = x.dtype

        out: torch.Tensor = torch.fft.rfft2(input, dim=(1, 2), norm="ortho")
        out: torch.Tensor = out.reshape(batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size)

        out1_real: torch.Tensor = torch.zeros(
            (batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor), 
            device=x.device,
        )
        out1_imag: torch.Tensor = torch.zeros(
            (batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor), 
            device=x.device,
        )
        out2_real: torch.Tensor = torch.zeros(out.shape, device=x.device)
        out2_imag: torch.Tensor = torch.zeros(out.shape, device=x.device)

        total_modes: int = (x_dim * y_dim) // 2 + 1
        kept_modes: int = int(total_modes * self.hard_thresholding_fraction)

        

        out1_real[:, :, :kept_modes, :, :] = F.relu(
            torch.einsum('Bxmbi,bio->Bxmbo', out[:, :, :kept_modes, :, :].real, self.w1[0])
            - torch.einsum('...bi,bio->...bo', out[:, :, :kept_modes, :, :].imag, self.w1[1])
            + self.b1[0]
        )

        out1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias



if __name__ == '__main__':
    # x = torch.rand(32, 2, 64, 64)
    # self = SpectralConv2d(u_dim=x.shape[1], x_modes=16, y_modes=16)
    # y = self(x)

    # self = FNO2d(u_dim=2, x_modes=5, y_modes=5, width=25)
    # y = self(x)

    # self = AdaptiveSpectralConv2d(u_dim=2, x_dim=64, y_dim=64, min_explanation=0.8)
    # y = self(x)

    x1 = torch.rand(32, 2, 64, 64)
    x2 = torch.rand(32, 2, 32, 32)

    # self = SpectralConv2d(u_dim=x1.shape[1], x_modes=16, y_modes=16)

    # y1 = self(x1)
    # y2 = self(x2)

    self = AdaptiveSpectralConv2d(u_dim=x1.shape[1], x_dim=64, y_dim=64, min_explanation=0.8)

    y1 = self(x1)
    y2 = self(x2)


