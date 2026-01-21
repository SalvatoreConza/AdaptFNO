import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Import the classes you provided
from models.operators import GlobalOperator, LocalOperator

class AdaptFNOInpainting(nn.Module):
    def __init__(
        self,
        in_channels: int,       # e.g., 2 (1 Wind + 1 Mask)
        out_channels: int,      # e.g., 1 (Reconstructed Wind)
        img_size: Tuple[int, int], # High-res size (e.g., 128, 128)
        patch_size: Tuple[int, int],
        embedding_dim: int = 256,
        global_downsample_factor: int = 2, # How much smaller is the global view?
        n_layers: int = 4,
        block_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.global_factor = global_downsample_factor
        
        # Calculate Global Resolution (Low Res)
        self.global_size = (
            img_size[0] // global_downsample_factor, 
            img_size[1] // global_downsample_factor
        )

        # --- 1. Global Operator (Coarse Context) ---
        self.global_op = GlobalOperator(
            in_channels=in_channels,
            out_channels=out_channels, # Intermediate coarse output
            embedding_dim=embedding_dim,
            in_timesteps=1, out_timesteps=1, # Inpainting usually T=1
            n_layers=n_layers,
            block_size=block_size,
            spatial_resolution=self.global_size,
            patch_size=patch_size, # Assuming patch size stays same, or scale it too
            dropout_rate=dropout,
        )

        # --- 2. Local Operator (Fine Detail) ---
        self.local_op = LocalOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            in_timesteps=1, out_timesteps=1,
            n_layers=n_layers,
            block_size=block_size,
            spatial_resolution=img_size, # High Res
            patch_size=patch_size,
            dropout_rate=dropout,
            n_attention_heads=4, # For the cross-attention
        )

    def forward(self, x_high_res: torch.Tensor):
        """
        Args:
            x_high_res: (B, T, C, H, W) - The sparse input + mask
        """
        # 1. Downsample input for Global Operator
        # We merge B and T for 2D interpolation, then reshape back
        B, T, C, H, W = x_high_res.shape
        x_flat = x_high_res.view(B * T, C, H, W)
        
        x_low_res = F.interpolate(
            x_flat, 
            size=self.global_size, 
            mode='bilinear', 
            align_corners=False
        )
        x_low_res = x_low_res.view(B, T, C, self.global_size[0], self.global_size[1])

        # 2. Run Global Operator
        # We ignore global_pred for the final output, but use it for loss if desired
        global_pred, *global_contexts = self.global_op(x_low_res)

        # 3. Run Local Operator with Cross-Attention
        # The LocalOperator provided in your code already handles the context injection
        local_pred = self.local_op(x_high_res, global_contexts)

        return local_pred