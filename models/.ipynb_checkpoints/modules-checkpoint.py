from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm(nn.Module):

    def __init__(self, in_channels: int, affine: bool, track_running_stats: bool) -> None:
        super().__init__()
        self.normalizer = nn.InstanceNorm3d(
            num_features=in_channels, affine=affine, track_running_stats=track_running_stats,
        )
        self.in_channels: int = in_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        output: torch.Tensor = input.transpose(1, 2)
        output = self.normalizer(output).transpose(1, 2)
        assert input.shape == output.shape
        return output


# DONE
class PatchEmbedding(nn.Module):

    def __init__(
        self, 
        patch_size: Tuple[int, int], 
        n_patches: int,
        in_channels: int, 
        embedding_dim: int,
    ):
        """
        embedding_dim should be large enough compared to the patch size, 
        the effective embed dim is embedding_dim // prod(patch_size) 
        """
        super().__init__()
        self.patch_size: Tuple[int, int] = patch_size
        self.n_patches: int = n_patches
        self.in_channels: int = in_channels
        self.embedding_dim: int = embedding_dim
        
        self.projection = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.embedding_dim, 
            kernel_size=self.patch_size, stride=self.patch_size,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, in_timesteps, in_channels, x_resolution, y_resolution = input.shape
        assert in_channels == self.in_channels
        output: torch.Tensor = input.flatten(start_dim=0, end_dim=1)
        output = self.projection(output)
        output = output.reshape(batch_size, in_timesteps, self.embedding_dim, self.n_patches)
        return output.permute(0, 1, 3, 2)   # (batch_size, in_timesteps, n_patches, embedding_dim)


# DONE
class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim: int, n_patches: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_patches: int = n_patches
        self.weight = nn.Parameter(
            torch.randn(1, 1, self.n_patches, self.embedding_dim, dtype=torch.float)
        )

    def forward(self):
        return self.weight


# DONE
class DropPath(nn.Module):
    
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0 or not self.training:
            return input    # same as nn.Identity() in .eval() mode
        
        shape: Tuple[int, ...] = (input.shape[0],) + (1,) * (input.ndim - 1)
        random_tensor: torch.Tensor = (1 - self.p) + torch.rand(shape, dtype=input.dtype, device=input.device)
        mask: torch.Tensor = random_tensor.floor()
        output = input.div(1 - self.p) * mask
        return output


class FeatureMapping(nn.Module):
    """
    Implement Gaussian Fourier Feature Mapping
    https://arxiv.org/abs/2006.10739
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        assert n_channels % 2 == 0
        self.mapping: torch.Tensor = torch.randn(n_channels, n_channels // 2)
        self.scale = nn.Parameter(data=torch.tensor(1.))

    def forward(self, input: torch.Tensor):
        assert input.shape[-1] == self.n_channels
        output: torch.Tensor = torch.matmul(
            input=2 * torch.pi * input, 
            other=self.mapping.to(input.device) * self.scale
        )
        return torch.cat(tensors=[torch.sin(output), torch.cos(output)], dim=-1)


# DONE
class AFNOLayer(nn.Module):

    def __init__(
        self, 
        embedding_dim: int, 
        block_size: int,
        n_xpatches: int,
        n_ypatches: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.n_xpatches: int = n_xpatches
        self.n_ypatches: int = n_ypatches
        self.dropout_rate: float = dropout_rate
        self.n_blocks: int = self.embedding_dim // self.block_size

        self.scale: float = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.n_blocks, block_size, block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, self.n_blocks, block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.n_blocks, block_size, block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, self.n_blocks, block_size))

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
            nn.Dropout(p=dropout_rate),
        )
        self.droppath = DropPath(p=dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, n_timesteps, n_xpatches, n_ypatches, embedding_dim = input.shape
        assert embedding_dim == self.embedding_dim
        assert n_xpatches == self.n_xpatches
        assert n_ypatches == self.n_ypatches

        output: torch.Tensor = self.ln1(input)

        # Fourier transform (Token mixing)
        fourier_coeff: torch.Tensor = torch.fft.rfft2(input=output, dim=(2, 3), norm="ortho")
        # Linear transformation with shared weight (Channel mixing)
        x_modes = fourier_coeff.shape[2]
        y_modes = fourier_coeff.shape[3]
        assert (x_modes, y_modes) == (self.n_xpatches, self.n_ypatches // 2 + 1)

        fourier_coeff: torch.Tensor = fourier_coeff.reshape(
            batch_size, n_timesteps, x_modes, y_modes, self.n_blocks, self.block_size
        )

        output1_real = torch.zeros(fourier_coeff.shape, device=input.device)
        output1_imag = torch.zeros(fourier_coeff.shape, device=input.device)
        output2_real = torch.zeros(fourier_coeff.shape, device=input.device)
        output2_imag = torch.zeros(fourier_coeff.shape, device=input.device)

        ops: str = 'btxyni,nio->btxyno'
        output1_real = F.gelu(
            torch.einsum(ops, fourier_coeff.real, self.w1[0]) 
            - torch.einsum(ops, fourier_coeff.imag, self.w1[1]) 
            + self.b1[0]
        )
        output1_imag = F.gelu(
            torch.einsum(ops, fourier_coeff.imag, self.w1[0]) 
            + torch.einsum(ops, fourier_coeff.real, self.w1[1]) 
            + self.b1[1]
        )
        output2_real = (
            torch.einsum(ops, output1_real, self.w2[0]) 
            - torch.einsum(ops, output1_imag, self.w2[1]) 
            + self.b2[0]
        )
        output2_imag = (
            torch.einsum(ops, output1_imag, self.w2[0]) 
            + torch.einsum(ops, output1_real, self.w2[1]) 
            + self.b2[1]
        )
        output: torch.Tensor = torch.stack([output2_real, output2_imag], dim=-1)
        output = F.softshrink(output, lambd=0.01)
        output = torch.view_as_complex(output)
        output = output.reshape(batch_size, n_timesteps, x_modes, y_modes, embedding_dim)

        # Inverse Fourier transform (Token demixing)
        output: torch.Tensor = torch.fft.irfft2(
            input=output, 
            s=(self.n_xpatches, self.n_ypatches),
            dim=(2, 3), 
            norm="ortho",
        )
        assert output.shape == (
            batch_size, n_timesteps, self.n_xpatches, self.n_ypatches, embedding_dim
        )
        # Skip connection
        output = output + input
        residual = output
        # MLP
        output = self.ln2(output)
        output = self.mlp(output)
        assert output.shape == input.shape
        # Skip connection + Drop path
        output = self.droppath(output) + residual
        return output   # output.shape == input.shape


class Filter(nn.Module):

    def __init__(self, n_channels: int, n_hiddens: int, depth: int) -> None:
        super().__init__()
        self.n_channels: int = n_channels
        self.n_hiddens: int = n_hiddens
        self.depth: int = depth

        self.conv1 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_hiddens, 
            kernel_size=3, padding=1, stride=1,
        )
        self.blocks = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Conv2d(in_channels=n_hiddens, out_channels=n_hiddens, kernel_size=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=n_hiddens, out_channels=n_hiddens, kernel_size=1, stride=1),
                )
                for _ in range(self.depth)
            ]
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_hiddens, out_channels=n_channels, 
            kernel_size=3, padding=1, stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, n_timesteps, n_channels, x_resolution, y_resolution = input.shape
        output: torch.Tensor = input.flatten(start_dim=0, end_dim=1)
        # First conv block
        output = self.conv1(output)
        # Residual connection blocks
        for block in self.blocks:
            output = block(output) + output
        
        # Last conv block
        output = self.conv2(output)
        assert output.shape == (
            batch_size * n_timesteps, n_channels, x_resolution, y_resolution
        )
        return output.reshape(batch_size, n_timesteps, n_channels, x_resolution, y_resolution)
        

# DONE
class LinearDecoder(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        out_timesteps: int,
        n_xpatches: int,
        n_ypatches: int,
        patch_size: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.out_timesteps: int = out_timesteps
        self.n_xpatches: int = n_xpatches
        self.n_ypatches: int = n_ypatches
        self.patch_size: Tuple[int, int] = patch_size
        self.x_resolution: int = patch_size[0] * n_xpatches
        self.y_resolution: int = patch_size[1] * n_ypatches

        self.temporal_decoder = nn.Sequential(
            nn.Linear(in_features=in_timesteps, out_features=in_timesteps * 4),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 4, out_features=in_timesteps * 4),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 4, out_features=out_timesteps),
        )
        self.spatial_decoder = nn.Linear(
            in_features=in_channels,
            out_features=patch_size[0] * patch_size[1] * out_channels,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, in_timesteps = input.shape[:2]
        assert (batch_size, in_timesteps, self.n_xpatches, self.n_ypatches, self.in_channels) == input.shape
        # Temporal decoding
        output: torch.Tensor = self.temporal_decoder(input.permute(0, 2, 3, 4, 1))
        output = output.permute(0, 4, 1, 2, 3) # batch_size, out_timesteps, self.n_xpatches, self.n_ypatches, self.in_channels
        # Spatial decoding
        output = self.spatial_decoder(output)
        assert output.shape == (
            batch_size, self.out_timesteps, self.n_xpatches, self.n_ypatches, self.patch_size[0] * self.patch_size[1] * self.out_channels
        )
        # Reshape
        output = output.reshape(
            batch_size, self.out_timesteps, self.n_xpatches, self.n_ypatches, self.patch_size[0], self.patch_size[1], self.out_channels
        )
        output = output.permute(0, 1, 6, 2, 4, 3, 5)
        output = output.flatten(start_dim=3, end_dim=4)
        output = output.flatten(start_dim=4, end_dim=5)
        assert output.shape == (
            batch_size, self.out_timesteps, self.out_channels, self.x_resolution, self.y_resolution
        )
        return output


class GlobalAttention(nn.Module):

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int,
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_heads: int = n_heads
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            bias=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
            ),
        )
        self.query_ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.key_ln = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(
        self, 
        global_context: torch.Tensor, 
        local_context: torch.Tensor,
    ) -> torch.Tensor:
        assert global_context.ndim == local_context.ndim == 5
        assert global_context.shape[:2] == local_context.shape[:2]
        batch_size, in_timesteps = local_context.shape[:2]
        assert global_context.shape[-1] == local_context.shape[-1] == self.embedding_dim
        # NOTE: global_context and local_context may have diferent number of patches

        global_context_reshaped: torch.Tensor = self._transform_shape(input=global_context)
        local_context_reshaped: torch.Tensor = self._transform_shape(input=local_context)
        # Cross attention
        output: torch.Tensor = self.cross_attention(
            query=self.query_ln(local_context_reshaped), 
            key=self.key_ln(global_context_reshaped),
            value=global_context_reshaped,
            attn_mask=None,
            need_weights=False, # to save significant memory for large sequence length
        )[0]
        output = local_context_reshaped + output
        output = self.feedforward(output) + output
  
        output = self._untransform_shape(
            input=output, in_timesteps=in_timesteps, 
            n_xpatches=local_context.shape[2], n_ypatches=local_context.shape[3], 
        )
        assert output.shape == local_context.shape
        return output

    @staticmethod
    def _transform_shape(input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, in_timesteps, n_xpatches, n_ypatches, embedding_dim = input.shape
        output: torch.Tensor = input.flatten(start_dim=1, end_dim=3) # attend to different space in different time
        return  output.permute(1, 0, 2) # n_timesteps * n_xpatches * n_ypatches, batch_size, embedding_dim
    
    @staticmethod
    def _untransform_shape(input: torch.Tensor, in_timesteps: int, n_xpatches: int, n_ypatches: int) -> torch.Tensor:
        assert input.ndim == 3
        sequence_length, batch_size, embedding_dim = input.shape
        output: torch.Tensor = input.reshape(in_timesteps, n_xpatches, n_ypatches, batch_size, embedding_dim)
        return output.permute(3, 0, 1, 2, 4)   # batch_size, n_timesteps, n_xpatches, n_ypatches, embedding_dim



if __name__ == '__main__':

    device = torch.device('cuda')
    self = LinearDecoder(
        in_channels=128, out_channels=2, 
        in_timesteps=12, out_timesteps=12,
        n_xpatches=32, n_ypatches=64, patch_size=(16, 16)
    ).to(device)
    x = torch.rand(32, 12, 128, 32, 64).to(device)
    y = self(x)
    print(y.shape)


