from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):

    def __init__(self, embedding_dim: int, n_tmodes: int, n_hmodes: int, n_wmodes: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_tmodes: int = n_tmodes
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.scale: float = 0.02
        self.weights_real = nn.Parameter(
            self.scale * torch.randn((2, n_tmodes, n_hmodes, n_wmodes, embedding_dim, embedding_dim), dtype=torch.float)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn((2, n_tmodes, n_hmodes, n_wmodes, embedding_dim, embedding_dim), dtype=torch.float)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        N, T, H, W, E = input.shape
        assert E == self.embedding_dim
        # FFT
        fourier_coeff: torch.Tensor = torch.fft.rfftn(input, dim=(1, 2, 3), norm="ortho")
        output_real = torch.zeros((N, T, H, W, E), device='cuda')
        output_imag = torch.zeros((N, T, H, W, E), device='cuda')

        pos_freq_slice: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(None, self.n_tmodes), slice(None, self.n_hmodes), slice(None, self.n_wmodes), slice(None), 
        )   # [:, :self.n_tmodes, :self.n_hmodes, :self.n_wmodes, :] 
        neg_freq_slice: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(-self.n_tmodes, None), slice(-self.n_hmodes, None), slice(None, self.n_wmodes), slice(None), 
        )   # [:, -self.t_hmodes:, -self.n_hmodes:, :self.n_wmodes, :]
        output_real[pos_freq_slice], output_imag[pos_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[pos_freq_slice],
            input_imag=fourier_coeff.imag[pos_freq_slice],
            weights_real=self.weights_real[0],
            weights_imag=self.weights_imag[0],
        )
        output_real[neg_freq_slice], output_imag[neg_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[neg_freq_slice],
            input_imag=fourier_coeff.imag[neg_freq_slice],
            weights_real=self.weights_real[1],
            weights_imag=self.weights_imag[1],
        )
        # IFFT
        output: torch.Tensor = torch.complex(output_real, output_imag)
        output = torch.fft.irfftn(output, s=(T, H, W), dim=(1, 2, 3), norm="ortho")
        assert output.shape == input.shape == (N, T, H, W, E)
        return output

    @staticmethod
    def complex_mul(
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops: str = 'nthwi,thwio->nthwo'
        real_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_real) - torch.einsum(ops, input_imag, weights_imag)
        )
        imag_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_imag) + torch.einsum(ops, input_imag, weights_real)
        )
        return real_part, imag_part


class GlobalAttention(nn.Module):

    def __init__(self, 
        global_embedding_dim: int, 
        local_embedding_dim: int, 
        n_heads: int, global_patch_size: Tuple[int, int]
    ):
        super().__init__()
        self.global_embedding_dim: int = global_embedding_dim
        self.local_embedding_dim: int = local_embedding_dim
        self.n_heads: int = n_heads
        self.global_patch_size: Tuple[int, int] = global_patch_size
        self.hpatch_size, self.wpatch_size = global_patch_size
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_features=self.hpatch_size * self.wpatch_size * global_embedding_dim, out_features=global_embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=global_embedding_dim, out_features=global_embedding_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.local_embedding_dim, 
            # kdim=self.hpatch_size * self.wpatch_size * self.global_embedding_dim, 
            # vdim=self.hpatch_size * self.wpatch_size * self.global_embedding_dim,
            kdim=self.global_embedding_dim, 
            vdim=self.global_embedding_dim,
            num_heads=n_heads, batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=self.local_embedding_dim, out_features=self.local_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.local_embedding_dim * 2, out_features=self.local_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.local_embedding_dim * 2, out_features=self.local_embedding_dim),
        )

    def forward(self, global_context: torch.Tensor, local_context: torch.Tensor) -> torch.Tensor:
        assert global_context.ndim == local_context.ndim == 5
        assert global_context.shape[:2] == local_context.shape[:2]
        batch_size, in_timesteps = local_context.shape[:2]
        assert global_context.shape[-1] == self.global_embedding_dim
        assert local_context.shape[-1] == self.local_embedding_dim
        # NOTE: global_context and local_context may have diferent spatial resolution
        local_H, local_W = local_context.shape[2:4]
        global_H, global_W = global_context.shape[2:4]
        assert global_H % self.hpatch_size == 0
        assert global_W % self.wpatch_size == 0

        h_patches: int = global_H // self.hpatch_size
        w_patches: int = global_W // self.wpatch_size
        patched_global_context: torch.Tensor = global_context.reshape(
            batch_size, in_timesteps, h_patches, self.hpatch_size, w_patches, self.wpatch_size, self.global_embedding_dim
        )
        patched_global_context = patched_global_context.transpose(3, 4)
        patched_global_context = patched_global_context.flatten(start_dim=4, end_dim=6)
        patched_global_context = patched_global_context.flatten(start_dim=1, end_dim=3)
        assert patched_global_context.shape == (
            batch_size, in_timesteps * h_patches * w_patches, self.hpatch_size * self.wpatch_size * self.global_embedding_dim
        )
        patched_global_context = self.feature_mlp(patched_global_context)
        local_context_reshaped: torch.Tensor = local_context.flatten(start_dim=1, end_dim=3) 
        assert local_context_reshaped.shape == (batch_size, in_timesteps * local_H * local_W, self.local_embedding_dim)

        # Cross attention
        output: torch.Tensor = self.cross_attention(
            query=local_context_reshaped, 
            key=patched_global_context,
            value=patched_global_context,
            attn_mask=None,
            need_weights=False, # to save significant memory for large sequence length
        )[0]
        assert output.shape == local_context_reshaped.shape

        # Add residual connection and feedforward
        output = self.feedforward(local_context_reshaped + output) + output
        output = output.reshape(batch_size, in_timesteps, local_H, local_W, self.local_embedding_dim)
        assert output.shape == local_context.shape
        return output


