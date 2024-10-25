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

        padded_T: int = self.next_power_of_2(T)
        padded_H: int = self.next_power_of_2(H)
        padded_W: int = self.next_power_of_2(W)
        if (padded_T, padded_H, padded_W) != (T, H, W):
            padded_input: torch.Tensor = F.pad(
                input=input,
                pad=(0, 0, 0, padded_W - W, 0, padded_H - H, 0, padded_T - T),
                mode='constant', value=0
            )
        else:
            padded_input: torch.Tensor = input
        # FFT
        fourier_coeff: torch.Tensor = torch.fft.rfftn(padded_input, dim=(1, 2, 3), norm="ortho")
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

    @staticmethod
    def next_power_of_2(x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()


class GlobalAttention(nn.Module):

    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_heads: int = n_heads
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 8),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 8, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
        )
        self.query_ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.key_ln = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, global_context: torch.Tensor, local_context: torch.Tensor) -> torch.Tensor:
        assert global_context.ndim == local_context.ndim == 5
        assert global_context.shape[:2] == local_context.shape[:2]
        batch_size, in_timesteps = local_context.shape[:2]
        assert global_context.shape[-1] == local_context.shape[-1] == self.embedding_dim
        # NOTE: global_context and local_context may have diferent spatial resolution
        n_local_hmodes, n_local_wmodes = local_context.shape[2:4]
        n_global_hmodes, n_global_wmodes = global_context.shape[2:4]

        global_context_reshaped: torch.Tensor = global_context.flatten(start_dim=1, end_dim=3)
        local_context_reshaped: torch.Tensor = local_context.flatten(start_dim=1, end_dim=3)
        # Cross attention
        output: torch.Tensor = self.cross_attention(
            query=self.query_ln(local_context_reshaped), 
            key=self.key_ln(global_context_reshaped),
            value=global_context_reshaped,
            attn_mask=None,
            need_weights=False, # to save significant memory for large sequence length
        )[0]
        assert output.shape == (batch_size, in_timesteps * n_local_hmodes * n_local_wmodes, self.embedding_dim)
        output = local_context_reshaped + output
        output = self.feedforward(output) + output
        output = output.reshape(batch_size, in_timesteps, n_local_hmodes, n_local_wmodes, self.embedding_dim)
        assert output.shape == local_context.shape
        return output


