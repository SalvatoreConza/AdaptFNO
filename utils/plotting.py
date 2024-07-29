import os
from typing import List, Tuple, Optional, Callable

import datetime as dt
import matplotlib.pyplot as plt
import torch


def plot_2d(
    *states: Tuple[torch.Tensor],
    timesteps: List[int],
    dim_names: List[str],
    filename: str,
):

    for state in states:
        assert state.ndim == 3
        assert len(timesteps) == len(states)
        state.to(device=torch.device('cpu'))

    u_dim, x_dim, y_dim = state.shape
    assert len(dim_names) == u_dim

    fig, axs = plt.subplots(len(timesteps), u_dim, figsize=(5 * u_dim, 5 * len(timesteps)))
    for t_idx, t in enumerate(timesteps):
        for dim, dim_name in enumerate(dim_names):
            axs[t_idx, dim].imshow(
                states[t_idx][dim],
                aspect="auto",
                origin="lower",
                extent=[-1., 1., -1., 1.],
            )
            axs[t_idx, dim].set_xticks([])
            axs[t_idx, dim].set_yticks([])
            axs[t_idx, dim].set_title(f"${dim_name}(t={t})$", fontsize=40)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.01, top=0.99, wspace=0.2, hspace=0.25)
    plt.savefig(filename)


def plot_predictions_2d(
    groundtruths: torch.Tensor,
    predictions: torch.Tensor,
    notes: Optional[List[str]] = None,
    reduction: Callable[[torch.Tensor], torch.Tensor] = None,
) -> None:

    assert groundtruths.shape == predictions.shape
    assert groundtruths.ndim == 5   # (batch_size, t_dim, u_dim, x_resolution, y_resolution)
    assert groundtruths.shape[1] == 1, 'Expect `t_dim` to be 1'

    groundtruths: torch.Tensor = reduction(groundtruths)
    predictions: torch.Tensor = reduction(predictions)

    assert groundtruths.shape[2] == predictions.shape[2] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got predictions.shape[2]={predictions.shape[2]} and '
        f'got groundtruths.shape[2]={groundtruths.shape[2]}'
    )
    assert notes is None or len(notes) == groundtruths.shape[0]

    os.makedirs(f"{os.getenv('PYTHONPATH')}/results", exist_ok=True)

    groundtruths = groundtruths.to(device=torch.device('cpu'))
    predictions = predictions.to(device=torch.device('cpu'))

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruths.shape[3]
    y_res: int = groundtruths.shape[4]
    aspect_ratio: float = y_res / x_res

    # Set plot configuration
    cmap: str = 'plasma'
    vmin = min(groundtruths.min().item(), predictions.min().item())
    vmax = max(groundtruths.max().item(), predictions.max().item())

    for idx in range(predictions.shape[0]):
        gt_field: torch.Tensor = groundtruths[idx]
        pred_field: torch.Tensor = predictions[idx]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(
            gt_field.squeeze(dim=(0, 1)),
            aspect=aspect_ratio, origin="lower",
            extent=[-1., 1., -1., 1.],
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(f'$groundtruth$', fontsize=20)
        axs[1].imshow(
            pred_field.squeeze(dim=(0, 1)),
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
            f"{os.getenv('PYTHONPATH')}/results/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)