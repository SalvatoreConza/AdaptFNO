from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import InstanceNorm, GlobalAttention, SpectralConv3d, LinearDecoder


class _BaseOperator(nn.Module):

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

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim * 8, out_features=embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim * 8, out_features=embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim * 8, out_features=embedding_dim),
        )  
        self.H, self.W = spatial_resolution
        self.layer_norm = nn.LayerNorm(normalized_shape=(in_timesteps, self.H, self.W, embedding_dim))
        self.spectral_convs = nn.ModuleList(
            modules=[
                SpectralConv3d(embedding_dim=embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_timesteps * embedding_dim, out_features=in_timesteps * embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=in_timesteps * embedding_dim * 8, out_features=out_timesteps * embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=out_timesteps * embedding_dim * 8, out_features=out_timesteps * embedding_dim * 8),
            nn.GELU(),
            nn.Linear(in_features=out_timesteps * embedding_dim * 8, out_features=out_timesteps * out_channels),
        )
        self.global_attention = None

    def _forward(
        self, 
        input: torch.Tensor, 
        in_contexts: List[torch.Tensor] | None = None,
    ):
        batch_size: int = input.shape[0]
        assert input.shape == (batch_size, self.in_timesteps, self.in_channels, self.H, self.W)
        output: torch.Tensor = input.permute(0, 1, 3, 4, 2)  # (batch_size, self.in_timesteps, self.H, self.W, self.in_channels)
        # validate in_contexts
        if in_contexts is not None:
            assert self.n_layers <= len(in_contexts), 'LocalOperator.n_layers must not exceed GlobalOperator.n_layers'
            assert self.embedding_dim == in_contexts[0].shape[-1], 'LocalOperator.embedding_dim must be equal to GlobalOperator.embedding_dim'
            # Only select the last `n_layers` global contexts
            in_contexts = in_contexts[-self.n_layers:]

        output: torch.Tensor = self.embedding_layer(output)
        output = self.layer_norm(output)
        assert output.shape == (batch_size, self.in_timesteps, self.H, self.W, self.embedding_dim)
        # Fourier Layers
        out_contexts: List[torch.Tensor] = []
        for i in range(self.n_layers):
            spectral_conv: SpectralConv3d = self.spectral_convs[i]
            output = spectral_conv(output)
            assert output.shape == (batch_size, self.in_timesteps, self.H, self.W, self.embedding_dim)
            out_contexts.append(output)

            if in_contexts is not None:
                assert self.global_attention is not None, "`self.global_attention` must be defined in subclass"
                # Condition on input context (shared cross attention)
                output = self.global_attention(global_context=in_contexts[i], local_context=output)

        # Decoder
        assert output.shape == (batch_size, self.in_timesteps, self.H, self.W, self.embedding_dim)
        output = output.permute(0, 2, 3, 1, 4).flatten(start_dim=-2, end_dim=-1)
        output: torch.Tensor = self.decoder(output)
        assert output.shape == (batch_size, self.H, self.W, self.out_timesteps * self.out_channels)
        output.reshape(batch_size, self.H, self.W, self.out_timesteps, self.out_channels).permute(0, 3, 4, 1, 2)
        assert output.shape == input.shape == (batch_size, self.out_timesteps, self.out_channels, self.H, self.W)
        return output, *out_contexts


class GlobalOperator(_BaseOperator):

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._forward(input, in_contexts=None)


class LocalOperator(_BaseOperator):

    def __init__(
        self, 
        in_channels: int, out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_hmodes: int, n_wmodes: int,
        n_layers: int,
        spatial_resolution: Tuple[int, int],
        n_attention_heads: int,
    ):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            embedding_dim=embedding_dim, 
            in_timesteps=in_timesteps, out_timesteps=out_timesteps,
            n_hmodes=n_hmodes, n_wmodes=n_wmodes,
            n_layers=n_layers,
            spatial_resolution=spatial_resolution,
            n_attention_heads=n_attention_heads,
        )
        self.n_attention_heads: int = n_attention_heads
        self.global_attention = GlobalAttention(embedding_dim=self.embedding_dim, n_heads=self.n_attention_heads)

    def forward(self, input: torch.Tensor, global_contexts: List[torch.Tensor]) -> torch.Tensor:
        return self._forward(input, in_contexts=global_contexts)[0]


