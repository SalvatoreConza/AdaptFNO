from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import GlobalAttention, SpectralConv2d


class GlobalOperator(nn.Module):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_hmodes: int, n_wmodes: int,
        n_layers: int,
        spatial_resolution: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.embedding_dim: int = embedding_dim
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.n_layers: int = n_layers
        self.spatial_resolution: Tuple[int, int] = spatial_resolution
        assert self.embedding_dim > self.in_channels

        self.norm = nn.BatchNorm3d(num_features=in_channels)
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
        )
        self.spectral_convs = nn.ModuleList(
            modules=[
                SpectralConv2d(embedding_dim=embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_layers)
            ]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.H, self.W = spatial_resolution

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size: int = input.shape[0]
        H, W = input.shape[-2:]
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, H, W)
        output: torch.Tensor = input.transpose(1, 2)    # (batch_size, self.in_channels, self.in_timesteps, H, W)
        output = self.norm(output)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape ==  (batch_size, self.in_timesteps, H, W, self.in_channels)

        output = self.embedding_layer(output)
        assert output.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
        # Fourier Layers
        out_contexts: List[torch.Tensor] = []
        for i in range(self.n_layers):
            out1: torch.Tensor = self.spectral_convs[i](output)
            assert out1.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            out2: torch.Tensor = self.Ws[i](output)
            assert out2.shape == (batch_size, self.in_timesteps, H, W, self.embedding_dim)
            output = out1 + out2
            output = self.mlps[i](output) + output
            out_contexts.append(output)

        return out_contexts


class LocalOperator(nn.Module):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        local_embedding_dim: int, global_embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_hmodes: int, n_wmodes: int,
        n_layers: int,
        n_attention_heads: int,
        global_patch_size: Tuple[int, int],
        spatial_resolution: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.local_embedding_dim: int = local_embedding_dim
        self.global_embedding_dim: int = global_embedding_dim
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.n_layers: int = n_layers
        self.spatial_resolution: Tuple[ int, int] = spatial_resolution
        self.local_H, self.local_W = spatial_resolution
        assert self.local_embedding_dim > self.in_channels

        self.norm = nn.BatchNorm3d(num_features=in_channels)
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=local_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim),
        )
        self.spectral_convs = nn.ModuleList(
            modules=[
                SpectralConv2d(embedding_dim=local_embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_layers)
            ]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim)
                for _ in range(n_layers)
            ]
        )
        self.n_attention_heads: int = n_attention_heads
        self.global_patch_size: Tuple[int, int] = global_patch_size
        self.global_attentions = nn.ModuleList(
            GlobalAttention(
                global_embedding_dim=global_embedding_dim, local_embedding_dim=local_embedding_dim, 
                n_heads=self.n_attention_heads, global_patch_size=global_patch_size,
            )
            for _ in range(n_layers)
        )
        self.mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(in_features=local_embedding_dim, out_features=local_embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim * 4),
                    nn.ReLU(),
                    nn.Linear(in_features=local_embedding_dim * 4, out_features=local_embedding_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder_weight0 = nn.Parameter(
            data=0.01 * torch.randn(in_timesteps, out_timesteps * 4, local_embedding_dim, out_channels * 4, device='cuda'),
            requires_grad=True,
        )
        self.decoder_weight1 = nn.Parameter(
            data=0.01 * torch.randn(out_timesteps * 4, out_timesteps * 4, out_channels * 4, out_channels * 4, device='cuda'),
            requires_grad=True,
        )
        self.decoder_weight2 = nn.Parameter(
            data=0.01 * torch.randn(out_timesteps * 4, out_timesteps * 4, out_channels * 4, out_channels * 4, device='cuda'),
            requires_grad=True,
        )
        self.decoder_weight3 = nn.Parameter(
            data=0.01 * torch.randn(out_timesteps * 4, out_timesteps, out_channels * 4, out_channels, device='cuda'),
            requires_grad=True,
        )

    def forward(self, input: torch.Tensor, global_contexts: List[torch.Tensor], out_resolution: Tuple[int, int] | None = None):
        batch_size: int = input.shape[0]
        in_H, in_W = input.shape[-2:]
        assert (in_H, in_W) == self.spatial_resolution
        out_H, out_W = in_H, in_W   # defaul output's resolution
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, in_H, in_W)
        output: torch.Tensor = input.transpose(1, 2)    # (batch_size, self.in_channels, self.in_timesteps, H, W)
        output = self.norm(output)
        output = output.permute(0, 2, 3, 4, 1)
        assert output.shape == (batch_size, self.in_timesteps, in_H, in_W, self.in_channels)
        # validate global_contexts
        assert self.n_layers <= len(global_contexts), 'LocalOperator.n_layers must not exceed GlobalOperator.n_layers'
        # Only select the last `n_layers` global contexts
        global_contexts = global_contexts[-self.n_layers:]
        output = self.embedding_layer(output)
        assert output.shape == (batch_size, self.in_timesteps, in_H, in_W, self.local_embedding_dim)

        # during inferencing of local operator
        if (not self.training) and (out_resolution is not None):
            out_H, out_W = out_resolution   # override
            output = self._upsampling(output, out_resolution=out_resolution)
            assert output.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)

        # Fourier Layers
        assert self.n_hmodes <= out_H // 2 + 1 and self.n_wmodes <= out_W // 2 + 1, 'choose smaller n_hmodes/n_wmodes for downsampled input size'
        for i in range(self.n_layers):
            # Condition on input context (shared cross attention)
            output = self.global_attentions[i](global_context=global_contexts[i], local_context=output) + output
            # Standard FNO transformations 
            out1: torch.Tensor = self.spectral_convs[i](output)
            assert out1.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)
            out2: torch.Tensor = self.Ws[i](output)
            assert out2.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)
            output = out1 + out2
            output = self.mlps[i](output) + output

        # Decoder
        assert output.shape == (batch_size, self.in_timesteps, out_H, out_W, self.local_embedding_dim)
        output = torch.einsum('nshwe,steo->nthwo', output, self.decoder_weight0)
        output = torch.relu(output)
        output = torch.einsum('nshwe,steo->nthwo', output, self.decoder_weight1)
        output = torch.relu(output)
        output = torch.einsum('nshwe,steo->nthwo', output, self.decoder_weight2)
        output = torch.relu(output)
        output = torch.einsum('nshwe,steo->nthwo', output, self.decoder_weight3)
        output = output.permute(0, 1, 4, 2, 3)
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