import os
from typing import List, Tuple, Optional, Callable

import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_groundtruths_2d(
    groundtruths: torch.Tensor,
    reduction: Callable[[torch.Tensor], torch.Tensor] | None = None,
    resolution: Tuple[int, int] | None = None,
) -> None:
    
    assert groundtruths.ndim == 4   # (timesteps, u_dim, x_resolution, y_resolution)

    if reduction is not None:
        groundtruths: torch.Tensor = reduction(groundtruths)

    assert groundtruths.shape[1] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got groundtruths.shape[1]={groundtruths.shape[1]}'
    )
    # Prepare output directory and move tensor to CPU
    destination_directory: str = './plots/groundtruth'
    os.makedirs(destination_directory, exist_ok=True)
    groundtruths: torch.Tensor = groundtruths.to(device=torch.device('cpu'))

    # Resize:
    if resolution is not None:
        groundtruths: torch.Tensor = F.interpolate(input=groundtruths, size=resolution, mode='bicubic')

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruths.shape[2]
    y_res: int = groundtruths.shape[3]
    aspect_ratio: float = x_res / y_res

    # Set plot configuration
    cmap: str = 'jet'

    for idx in range(groundtruths.shape[0]):
        field: torch.Tensor = groundtruths[idx]
        figwidth: float = 10.
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * aspect_ratio))
        ax.imshow(
            field.squeeze(dim=0).rot90(k=2).flip(dims=(1,)),
            origin="lower",
            vmin=0, vmax=field.max().item(),
            cmap=cmap,
        )
        ax.set_title(f'$groundtruth$', fontsize=15)
        
        # fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=0.05)
        fig.tight_layout()
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)    



def plot_predictions_2d(
    groundtruths: torch.Tensor,
    predictions: torch.Tensor,
    timestamps: List[str],
    metrics_notes: List[str],
    reduction: Callable[[torch.Tensor], torch.Tensor] | None = None,
    resolution: Tuple[int, int] | None = None,
) -> None:

    assert groundtruths.shape == predictions.shape
    assert groundtruths.ndim == 4   # (timesteps, u_dim, x_resolution, y_resolution)
    
    if reduction is not None:
        groundtruths: torch.Tensor = reduction(groundtruths)
        predictions: torch.Tensor = reduction(predictions)

    assert groundtruths.shape[1] == predictions.shape[1] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got predictions.shape[1]={predictions.shape[1]} and '
        f'got groundtruths.shape[1]={groundtruths.shape[1]}'
    )
    assert len(metrics_notes) == groundtruths.shape[0]

    # Prepare output directory and move tensor to CPU
    destination_directory: str = './plots/prediction'
    os.makedirs(destination_directory, exist_ok=True)
    groundtruths = groundtruths.to(device=torch.device('cpu'))
    predictions = predictions.to(device=torch.device('cpu'))

    # Resize:
    if resolution is not None:
        groundtruths: torch.Tensor = F.interpolate(input=groundtruths, size=resolution, mode='bicubic')
        predictions: torch.Tensor = F.interpolate(input=predictions, size=resolution, mode='bicubic')

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruths.shape[2]
    y_res: int = groundtruths.shape[3]
    aspect_ratio: float = x_res / y_res

    # Set plot configuration
    cmap: str = 'jet'
    vmin: float = 0.
    vmax: float = groundtruths.max().item()

    for t in range(predictions.shape[0]):
        gt_field: torch.Tensor = groundtruths[t]
        pred_field: torch.Tensor = predictions[t]
        figwidth: float = 5.
        fig, axs = plt.subplots(2, 1, figsize=(figwidth, 2 * figwidth * aspect_ratio))
        axs[0].imshow(
            gt_field.squeeze(dim=0).rot90(k=2).flip(dims=(1,)),
            origin="lower",
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        axs[0].set_title(f'groundtruth - {timestamps[t]}', fontsize=12)
        axs[1].imshow(
            pred_field.squeeze(dim=0).rot90(k=2).flip(dims=(1,)),
            origin="lower",
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        axs[1].set_title(f'prediction - {metrics_notes[t]}', fontsize=12)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.95, hspace=0.15)
        
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)



    

