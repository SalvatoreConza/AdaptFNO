import os
from typing import List, Tuple, Optional, Callable

import datetime as dt
import matplotlib.pyplot as plt
import torch


def plot_2d(
    batch_field: torch.Tensor,
    reduction: Callable[[torch.Tensor], torch.Tensor] = None,
) -> None:
    
    assert batch_field.ndim == 5   # (batch_size, t_dim, u_dim, x_resolution, y_resolution)
    assert batch_field.shape[1] == 1, 'Expect `t_dim` to be 1'

    if reduction is not None:
        batch_field: torch.Tensor = reduction(batch_field)

    assert batch_field.shape[2] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got batch_field.shape[2]={batch_field.shape[2]} and '
    )
    # Prepare output directory and move tensor to CPU
    destination_directory: str = './plots'
    os.makedirs(destination_directory, exist_ok=True)
    batch_field: torch.Tensor = batch_field.to(device=torch.device('cpu'))

    # Ensure that the plot respect the tensor's shape
    x_res: int = batch_field.shape[3]
    y_res: int = batch_field.shape[4]
    aspect_ratio: float = x_res / y_res

    # Set plot configuration
    cmap: str = 'gist_earth'

    for idx in range(batch_field.shape[0]):
        field: torch.Tensor = batch_field[idx]
        figwidth: float = 10.
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * aspect_ratio))
        ax.imshow(
            field.squeeze(dim=(0, 1)).rot90(k=2).flip(dims=(1,)),
            origin="lower",
            vmin=field.min().item(), vmax=field.max().item(),
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
    notes: Optional[List[str]] = None,
    reduction: Callable[[torch.Tensor], torch.Tensor] = None,
) -> None:

    assert groundtruths.shape == predictions.shape
    assert groundtruths.ndim == 5   # (batch_size, t_dim, u_dim, x_resolution, y_resolution)
    assert groundtruths.shape[1] == 1, 'Expect `t_dim` to be 1'
    
    if reduction is not None:
        groundtruths: torch.Tensor = reduction(groundtruths)
        predictions: torch.Tensor = reduction(predictions)

    assert groundtruths.shape[2] == predictions.shape[2] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got predictions.shape[2]={predictions.shape[2]} and '
        f'got groundtruths.shape[2]={groundtruths.shape[2]}'
    )
    assert notes is None or len(notes) == groundtruths.shape[0]

    # Prepare output directory and move tensor to CPU
    destination_directory: str = './results'
    os.makedirs(destination_directory, exist_ok=True)
    groundtruths = groundtruths.to(device=torch.device('cpu'))
    predictions = predictions.to(device=torch.device('cpu'))

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruths.shape[3]
    y_res: int = groundtruths.shape[4]
    aspect_ratio: float = x_res / y_res

    # Set plot configuration
    cmap: str = 'gist_earth'
    vmin = min(groundtruths.min().item(), predictions.min().item())
    vmax = max(groundtruths.max().item(), predictions.max().item())

    for idx in range(predictions.shape[0]):
        gt_field: torch.Tensor = groundtruths[idx]
        pred_field: torch.Tensor = predictions[idx]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(
            gt_field.squeeze(dim=(0, 1)).rot90(k=2).flip(dims=(1,)),
            aspect=aspect_ratio, origin="lower",
            extent=[-1., 1., -1., 1.],
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        axs[0].set_title(f'$groundtruth$', fontsize=20)
        axs[1].imshow(
            pred_field.squeeze(dim=(0, 1)).rot90(k=2).flip(dims=(1,)),
            aspect=aspect_ratio, origin="lower",
            extent=[-1., 1., -1., 1.],
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        if notes is None:
            axs[1].set_title(f'$prediction$', fontsize=20)
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=0.05)
        else:
            axs[1].set_title(f'$prediction$\n${notes[idx]}$', fontsize=20)
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.85, wspace=0.05)
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)


# TEST
if __name__ == '__main__':
    
    from functools import partial
    from torch.utils.data import DataLoader
    from era5.wind.datasets import Wind2dERA5
    from common.functional import compute_velocity_field
    
    data = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        latitude=(90, -90),
        longitude=(0, 360),
        fromdate='20240630',
        todate='20240630',
        bundle_size=1,
        window_size=1,
        resolution=(720, 1440),
        to_float16=False,
    )
    loader = DataLoader(data, batch_size=1000)
    input, output = next(iter(loader))
    print(input.shape)
    print(output.shape)

    plot_2d(batch_field=output, reduction=partial(compute_velocity_field, dim=2))


    

