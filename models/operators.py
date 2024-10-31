from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from models.modules import GlobalAttention, SpectralConv3d


class GlobalOperator(nn.Module):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_tmodes: int, n_hmodes: int, n_wmodes: int,
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
        self.Rs = nn.ModuleList(
            modules=[
                nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.Qv = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 2, out_features=out_channels),
        )
        self.Qt = nn.Sequential(
            nn.Linear(in_features=in_timesteps, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=out_timesteps),
        )
        self.global_attention = None
        self.positional_encoding = None

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size: int = input.shape[0]
        H, W = input.shape[-2:]
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, H, W)
        output: torch.Tensor = self.norm(input.transpose(1, 2))   # (batch_size, self.in_channels, self.in_timesteps, H, W)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape ==  (batch_size, self.in_timesteps, H, W, self.in_channels)

        # easy to go overflow operation
        with autocast(device_type='cuda', enabled=False):
            output = self.embedding_layer(output.float())
            assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)

        # Fourier Layers
        out_contexts: List[torch.Tensor] = []
        for i in range(self.n_layers):
            with autocast(device_type='cuda', enabled=False):
                out1: torch.Tensor = self.spectral_convs[i](output.float())
                assert out1.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)

            out2: torch.Tensor = self.Ws[i](output)
            assert out2.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            output = out1 + out2
            if i < self.n_layers - 1:   # not the last layer
                output = F.gelu(output)

            assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            out_contexts.append(output)

        # Decoder
        assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
        output = self.Qv(output)
        assert output.shape == (batch_size, self.in_timesteps, H, W, self.out_channels)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape == (batch_size, H, W, self.out_channels, self.in_timesteps)
        output = self.Qt(output).permute(0, 4, 3, 1, 2)
        assert output.shape == (batch_size, self.out_timesteps, self.out_channels, H, W)
        return output, *out_contexts


class LocalOperator(nn.Module):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        local_embedding_dim: int, global_embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_tmodes:int, n_hmodes: int, n_wmodes: int,
        n_layers: int,
        n_attention_heads: int,
        global_patch_size: Tuple[int, int],
        train_local_resolution: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.local_embedding_dim: int = local_embedding_dim
        self.global_embedding_dim: int = global_embedding_dim
        self.n_tmodes: int = n_tmodes
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.n_layers: int = n_layers
        assert self.local_embedding_dim > self.in_channels

        self.norm = nn.BatchNorm3d(num_features=in_channels)
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=local_embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim),
        )
        self.spectral_convs = nn.ModuleList(
            modules=[
                SpectralConv3d(embedding_dim=local_embedding_dim, n_tmodes=n_tmodes, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_layers)
            ]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.Rs = nn.ModuleList(
            modules=[
                nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.Qv = nn.Sequential(
            nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 2, out_features=local_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 2, out_features=local_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 2, out_features=out_channels),
        )
        self.Qt = nn.Sequential(
            nn.Linear(in_features=in_timesteps, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=in_timesteps * 2),
            nn.ReLU(),
            nn.Linear(in_features=in_timesteps * 2, out_features=out_timesteps),
        )
        self.n_attention_heads: int = n_attention_heads
        self.global_patch_size: Tuple[int, int] = global_patch_size
        self.train_local_resolution: Tuple[int, int] = train_local_resolution
        self.train_local_H, self.train_local_W = train_local_resolution
        self.global_attention = GlobalAttention(
            global_embedding_dim=global_embedding_dim, local_embedding_dim=local_embedding_dim, 
            n_heads=self.n_attention_heads, global_patch_size=global_patch_size,
        )

    def forward(self, input: torch.Tensor, global_contexts: List[torch.Tensor], out_resolution: Tuple[int, int] | None = None):
        batch_size: int = input.shape[0]
        in_H, in_W = input.shape[-2:]
        assert (in_H, in_W) == self.train_local_resolution
        out_H, out_W = in_H, in_W   # defaul output's resolution
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, in_H, in_W)
        output: torch.Tensor = self.norm(input.transpose(1, 2))   # (batch_size, self.in_channels, self.in_timesteps, H, W)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape ==  (batch_size, self.in_timesteps, in_H, in_W, self.in_channels)
        # validate global_contexts
        assert self.n_layers <= len(global_contexts), 'LocalOperator.n_layers must not exceed GlobalOperator.n_layers'
        # Only select the last `n_layers` global contexts
        global_contexts = global_contexts[-self.n_layers:]

        # easy to go overflow operation
        with autocast(device_type='cuda', enabled=False):
            output = self.embedding_layer(output.float())
            assert output.shape == (batch_size, self.in_timesteps, in_H, in_W, self.local_embedding_dim)
        
        # during inferencing of local operator
        if (not self.training) and (out_resolution is not None):
            out_H, out_W = out_resolution   # override
            output = self._upsampling(output, out_resolution=out_resolution)
            assert output.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)

        assert self.n_hmodes <= out_H // 2 + 1 and self.n_wmodes <= out_W // 2 + 1, 'choose smaller n_hmodes/n_wmodes for downsampled input size'

        # Fourier Layers
        for i in range(self.n_layers):
            with autocast(device_type='cuda', enabled=False):
                out1: torch.Tensor = self.spectral_convs[i](output.float())
                assert out1.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)

            out2: torch.Tensor = self.Ws[i](output)
            assert out2.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)
            output = out1 + out2
            if i < self.n_layers - 1:   # not the last layer
                output = F.gelu(output)

            # Condition on input context (shared cross attention)
            in_context: torch.Tensor = global_contexts[i]
            output = self.global_attention(global_context=in_context, local_context=output)

        # Decoder
        assert output.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)
        output = self.Qv(output)
        assert output.shape == (batch_size, self.in_timesteps, out_H, out_W, self.out_channels)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape == (batch_size, out_H, out_W, self.out_channels, self.in_timesteps)
        output = self.Qt(output).permute(0, 4, 3, 1, 2)
        assert output.shape == (batch_size, self.out_timesteps, self.out_channels, out_H, out_W)
        return output

    def _upsampling(self, input: torch.Tensor, out_resolution: Tuple[int, int]) -> torch.Tensor:
        assert input.ndim == 5  # (batch_size, in_timesteps, H, W, embedding_dim)
        batch_size, in_timesteps, H, W, embedding_dim = input.shape
        out_H, out_W = out_resolution
        output: torch.Tensor = input.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        output = F.interpolate(output, size=(out_resolution), mode='bicubic')
        assert output.shape == (batch_size * in_timesteps, embedding_dim, out_H, out_W)
        output = output.permute(0, 2, 3, 1).reshape(batch_size, in_timesteps, out_H, out_W, embedding_dim)
        return output