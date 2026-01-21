import torch
import torch.nn as nn

class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, predicted, target, mask=None):
        """
        predicted: (B, 1, 1, H, W)
        target:    (B, 1, 1, H, W)
        mask:      (B, 1, 1, H, W) - Optional, if you want to weight valid pixels more
        """
        # 1. Pixel-wise reconstruction loss
        loss_mse = self.mse(predicted, target)
        loss_l1 = self.l1(predicted, target)
        
        # 2. Gradient Loss (Prevents blurry blobs)
        pred_grad_x = torch.abs(predicted[..., :-1] - predicted[..., 1:])
        target_grad_x = torch.abs(target[..., :-1] - target[..., 1:])
        loss_grad_x = self.l1(pred_grad_x, target_grad_x)
        
        pred_grad_y = torch.abs(predicted[..., :-1, :] - predicted[..., 1:, :])
        target_grad_y = torch.abs(target[..., :-1, :] - target[..., 1:, :])
        loss_grad_y = self.l1(pred_grad_y, target_grad_y)

        # 3. Consistency Loss (Optional):
        # Ensure the model didn't "change" the pixels we actually observed!
        if mask is not None:
             # Calculate loss only on the valid pixels
             valid_loss = self.l1(predicted * mask, target * mask)
             return loss_mse + 0.5 * loss_l1 + 0.1 * (loss_grad_x + loss_grad_y) + 2.0 * valid_loss

        return loss_mse + 0.5 * loss_l1 + 0.1 * (loss_grad_x + loss_grad_y)