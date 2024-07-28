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
        assert input.ndim == 4, 'Expected input of shape: (batch_size, self.u_dim, x_res, y_res)'
        assert input.shape[1] == self.u_dim, 'input.shape[1] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_res: int = input.shape[2]
        y_res: int = input.shape[3]
        y_res: int = y_res // 2 + 1

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.rfft2(input)
        assert out_fft.shape == (batch_size, self.u_dim, x_res, y_res)

        # Linear transformation
        out_linear: torch.Tensor= self.R(
            input=out_fft[:, :, :self.x_modes, :self.y_modes],
        )
        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_res, y_res))
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
        # input dim = [batch_size, u_dim, x_res, y_res)
        batch_size: int = input.shape[0]
        u_dim: int = input.shape[1]
        x_res: int = input.shape[2]
        y_res: int = input.shape[3]
        
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
            f'(batch_size, self.width, x_res, y_res) ' 
            f'= {(batch_size, self.width, x_res, y_res)}'
        )
        out: torch.Tensor = out1 + out2
        out: torch.Tensor = F.gelu(out)

        # Block 1
        out1: torch.Tensor = self.spectral_conv1(out)
        out2: torch.Tensor = self.W1(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_res, y_res) ' 
            f'= {(batch_size, self.width, x_res, y_res)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 2
        out1 = self.spectral_conv2(out)
        out2 = self.W2(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_res, y_res) ' 
            f'= {(batch_size, self.width, x_res, y_res)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        # Block 3
        out1 = self.spectral_conv3(out)
        out2 = self.W3(out)
        assert out1.shape == out2.shape, (
            f'both out1 and out2 must have the same shape as '
            f'(batch_size, self.width, x_res, y_res) ' 
            f'= {(batch_size, self.width, x_res, y_res)}'
        )
        out = out1 + out2
        out = F.gelu(out)

        out = out.permute(0, 2, 3, 1)
        projected_output: torch.Tensor = self.Q(out)
        projected_output: torch.Tensor = F.gelu(projected_output)
        projected_output: torch.Tensor = projected_output.permute(0, 3, 1, 2)
        return projected_output
    
