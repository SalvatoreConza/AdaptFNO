from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    InstanceNorm, PatchEmbedding, PositionalEmbedding, 
    GlobalAttention, AFNOLayer, LinearDecoder
)

class _BaseOperator(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_layers: int, block_size: int,
        spatial_resolution: Tuple[int, int],
        patch_size: Tuple[int, int],
        dropout_rate: float,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.embedding_dim: int = embedding_dim
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.n_layers: int = n_layers
        self.block_size: int = block_size
        self.spatial_resolution: Tuple[int, int] = spatial_resolution
        self.patch_size: Tuple[int, int] = patch_size
        self.dropout_rate: float = dropout_rate

        assert self.embedding_dim > self.in_channels, 'embedding_dim must be large enough'
        assert self.embedding_dim % self.block_size == 0 and self.embedding_dim >= self.block_size, 'embedding_dim must be divisible by block_size'
        self.n_blocks: int = self.embedding_dim // self.block_size

        assert self.spatial_resolution[0] % self.patch_size[0] == 0, f'spatial_resolution {self.spatial_resolution} must be divisible by patch_size {self.patch_size}'
        assert self.spatial_resolution[1] % self.patch_size[1] == 0, f'spatial_resolution {self.spatial_resolution} must be divisible by patch_size {self.patch_size}'
        self.n_xpatches: int = self.spatial_resolution[0] // self.patch_size[0]
        self.n_ypatches: int = self.spatial_resolution[1] // self.patch_size[1]
        self.n_patches: int = self.n_xpatches * self.n_ypatches
        
        self.instance_norm = InstanceNorm(in_channels=in_channels, affine=False, track_running_stats=False)
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size, n_patches=self.n_patches,
            in_channels=self.in_channels, embedding_dim=self.embedding_dim,
        )
        self.positional_embedding = PositionalEmbedding(
            embedding_dim=self.embedding_dim, n_patches=self.n_patches
        )
        self.afno_layers = nn.ModuleList(
            modules=[
                AFNOLayer(
                    embedding_dim=embedding_dim, block_size=block_size, 
                    n_xpatches=self.n_xpatches, n_ypatches=self.n_ypatches, 
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear_decoder = LinearDecoder(
            in_channels=self.embedding_dim, out_channels=self.out_channels,
            in_timesteps=self.in_timesteps, out_timesteps=self.out_timesteps,
            n_xpatches=self.n_xpatches, n_ypatches=self.n_ypatches,
            patch_size=self.patch_size,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.global_attention = None

    def _get_embedding(self, input: torch.Tensor) -> torch.Tensor:
        # input dim = [batch_size, in_timesteps, in_channels, x_res, y_res)
        batch_size: int = input.shape[0]
        assert self.in_timesteps == input.shape[1]
        assert self.in_channels == input.shape[2]
        assert self.spatial_resolution == (input.shape[3], input.shape[4])

        # Patch + Positional Embedding
        patch_embedding: torch.Tensor = self.patch_embedding(input)
        positional_embedding: torch.Tensor = self.positional_embedding()
        embedding = self.dropout(patch_embedding + positional_embedding)  # broadcasted
        assert embedding.shape == (batch_size, self.in_timesteps, self.n_patches, self.embedding_dim)
        return embedding

    def _forward(
        self, 
        input: torch.Tensor, 
        in_contexts: List[torch.Tensor] | None = None,
    ):
        batch_size: int = input.shape[0]
        if in_contexts is not None:
            assert self.n_layers <= len(in_contexts), 'LocalOperator.n_layers must not exceed GlobalOperator.n_layers'
            assert self.embedding_dim == in_contexts[0].shape[-1], (
                'LocalOperator.embedding_dim must be equal to GlobalOperator.embedding_dim'
            )
            in_contexts = in_contexts[-self.n_layers:]  # Only select the last `n_layers` global contexts

        output: torch.Tensor = self.instance_norm(input)
        embedding: torch.Tensor = self._get_embedding(output)
        assert embedding.shape == (batch_size, self.in_timesteps, self.n_patches, self.embedding_dim)
        # Fourier Layers
        out_contexts: List[torch.Tensor] = []
        output: torch.Tensor = embedding.reshape(
            batch_size, self.in_timesteps, self.n_xpatches, self.n_ypatches, self.embedding_dim
        )
        for i in range(self.n_layers):
            afno_layer: AFNOLayer = self.afno_layers[i]
            output = afno_layer(output)
            # Append out_contexts
            out_contexts.append(output)

            if in_contexts is not None:
                assert self.global_attention is not None, "`self.global_attention` must be defined in subclass"
                # Condition on input context (shared cross attention)
                output = self.global_attention(global_context=in_contexts[i], local_context=output)

        # Linear decoder
        assert output.shape == (
            batch_size, self.in_timesteps, self.n_xpatches, self.n_ypatches, self.embedding_dim
        )
        output: torch.Tensor = self.linear_decoder(output)
        assert output.shape == (
            batch_size, self.out_timesteps, self.out_channels, self.spatial_resolution[0], self.spatial_resolution[1]
        )
        return output, *out_contexts


class GlobalOperator(_BaseOperator):

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._forward(input, in_contexts=None)


class LocalOperator(_BaseOperator):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        embedding_dim: int,
        in_timesteps: int, out_timesteps: int,
        n_layers: int, block_size: int,
        spatial_resolution: Tuple[int, int],
        patch_size: Tuple[int, int],
        dropout_rate: float,
        n_attention_heads: int,
    ):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            embedding_dim=embedding_dim, 
            in_timesteps=in_timesteps, out_timesteps=out_timesteps,
            n_layers=n_layers, block_size=block_size, 
            spatial_resolution=spatial_resolution, patch_size=patch_size, 
            dropout_rate=dropout_rate
        )
        self.n_attention_heads: int = n_attention_heads
        self.global_attention = GlobalAttention(embedding_dim=self.embedding_dim, n_heads=self.n_attention_heads)

    def forward(self, input: torch.Tensor, global_contexts: List[torch.Tensor]) -> torch.Tensor:
        return self._forward(input, in_contexts=global_contexts)[0]



if __name__ == '__main__':

    device = torch.device('cuda')
    global_input: torch.Tensor = torch.rand((4, 12, 2, 256, 256)).to(device)
    local_input: torch.Tensor = torch.rand((4, 12, 2, 128, 128)).to(device)

    global_operator = GlobalOperator(
        in_channels=2, out_channels=2,
        embedding_dim=512,
        in_timesteps=12, out_timesteps=12,
        n_layers=4,
        block_size=16,
        spatial_resolution=(256, 256),
        patch_size=(4, 4),
    ).to(device)

    global_output, *global_contexts = global_operator(input=global_input)
    print(f'global_output: {global_output.shape}')
    for global_context in global_contexts:
        print(f'global_context: {global_context.shape}')

    local_operator = LocalOperator(
        in_channels=2, 
        embedding_dim=global_operator.embedding_dim, 
        in_timesteps=global_operator.in_timesteps, 
        out_timesteps=global_operator.out_timesteps,
        n_layers=global_operator.n_layers,
        block_size=16,
        spatial_resolution=(128, 128),
        patch_size=(4, 4),
        n_attention_heads=8,
    ).to(device)
    
    local_output: torch.Tensor = local_operator(
        input=local_input, global_contexts=global_contexts,
    )[0]
    print(f'local_output: {local_output.shape}')




