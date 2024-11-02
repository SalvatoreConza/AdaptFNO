from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import SpectralConv3d


class FNO3D(nn.Module):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_tmodes:int, n_hmodes: int, n_wmodes: int,
        n_layers: int,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.embedding_dim: int = embedding_dim
        self.n_tmodes: int = n_tmodes
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.n_layers: int = n_layers
        assert self.embedding_dim > self.in_channels

        self.norm = nn.BatchNorm3d(num_features=in_channels)
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
        )
        self.spectral_convs = nn.ModuleList(
            modules=[
                SpectralConv3d(embedding_dim=embedding_dim, n_tmodes=n_tmodes, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_layers)
            ]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.decoder_weights = nn.Parameter(
            data=0.01 * torch.randn(in_timesteps, out_timesteps, embedding_dim, out_channels, device='cuda'),
            requires_grad=True,
        )

    def forward(self, input: torch.Tensor):
        batch_size: int = input.shape[0]
        H, W = input.shape[-2:]
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, H, W)
        output: torch.Tensor = self.norm(input.transpose(1, 2))   # (batch_size, self.in_channels, self.in_timesteps, H, W)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape ==  (batch_size, self.in_timesteps, H, W, self.in_channels)
        output = self.embedding_layer(output)
        assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)

        # Fourier Layers
        assert self.n_hmodes <= H // 2 + 1 and self.n_wmodes <= W // 2 + 1, 'choose smaller n_hmodes/n_wmodes for downsampled input size'
        for i in range(self.n_layers):
            out1: torch.Tensor = self.spectral_convs[i](output)
            assert out1.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            out2: torch.Tensor = self.Ws[i](output)
            assert out2.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            output = out1 + out2
            output = F.gelu(output)

        # Decoder
        assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
        output = torch.einsum('nshwe,steo->nthwo', output, self.decoder_weights)
        output = output.permute(0, 1, 4, 2, 3)
        assert output.shape == (batch_size, self.out_timesteps, self.out_channels, H, W)
        return output
    
