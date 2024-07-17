import os
import pathlib
import time
from typing import Optional, Dict, TextIO, Any, List, Tuple
from collections import defaultdict
import datetime as dt

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Accumulator:
    """
    A utility class for accumulating values for multiple metrics.
    """

    def __init__(self) -> None:
        self.__records: defaultdict[str, float] = defaultdict(float)

    def add(self, **kwargs: Any) -> None:
        """
        Add values to the accumulator.

        Parameters:
            - **kwargs: named metric and the value is the amount to add.
        """
        metric: str
        value: float
        for metric, value in kwargs.items():
            # Each keyword argument represents a metric name and its value to be added
            self.__records[metric] += value
    
    def reset(self) -> None:
        """
        Reset the accumulator by clearing all recorded metrics.
        """
        self.__records.clear()

    def __getitem__(self, key: str) -> float:
        """
        Retrieve a record by key.

        Parameters:
            - key (str): The record key name.

        Returns:
            - float: The record value.
        """
        return self.__records[key]


class EarlyStopping:
    """
    A simple early stopping utility to terminate training when a monitored metric stops improving.

    Attributes:
        - patience (int): The number of epochs with no improvement after which training will be stopped.
        - tolerance (float): The minimum change in the monitored metric to qualify as an improvement,
        - considering the direction of the metric being monitored.
        - bestscore (float): The best score seen so far.
    """
    
    def __init__(self, patience: int, tolerance: float = 0.) -> None:
        """
        Initializes the EarlyStopping instance.
        
        Parameters:
            - patience (int): Number of epochs with no improvement after which training will be stopped.
            - tolerance (float): The minimum change in the monitored metric to qualify as an improvement. 
            Defaults to 0.
        """
        self.patience: int = patience
        self.tolerance: float = tolerance
        self.bestscore: float = float('inf')
        self.__counter: int = 0

    def __call__(self, value: float) -> None:
        """
        Update the state of the early stopping mechanism based on the new metric value.

        Parameters:
            - value (float): The latest value of the monitored metric.
        """
        # Improvement or within tolerance, reset counter
        if value <= self.bestscore + self.tolerance:
            self.bestscore: float = value
            self.__counter: int = 0

        # No improvement, increment counter
        else:
            self.__counter += 1

    def __bool__(self) -> bool:
        """
        Determine if the training process should be stopped early.

        Returns:
            - bool: True if training should be stopped (patience exceeded), otherwise False.
        """
        return self.__counter >= self.patience


class Timer:

    """
    A class used to time the duration of epochs and batches.
    """
    def __init__(self) -> None:
        """
        Initialize the Timer.
        """
        self.__epoch_starts: Dict[int, float] = dict()
        self.__epoch_ends: Dict[int, float] = dict()
        self.__batch_starts: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.__batch_ends: Dict[int, Dict[int, float]] = defaultdict(dict)

    def start_epoch(self, epoch: int) -> None:
        """
        Start timing an epoch.

        Parameters:
            epoch (int): The epoch number.
        """
        self.__epoch_starts[epoch] = time.time()

    def end_epoch(self, epoch: int) -> None:
        """
        End timing an epoch.

        Parameters:
            - epoch (int): The epoch number.
        """
        self.__epoch_ends[epoch] = time.time()

    def start_batch(self, epoch: int, batch: Optional[int] = None) -> None:
        """
        Start timing a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int, optional): The batch number. If not provided, the next batch number is used.
        """
        if batch is None:
            if self.__batch_starts[epoch]:
                batch: int = max(self.__batch_starts[epoch].keys()) + 1
            else:
                batch: int = 1
        self.__batch_starts[epoch][batch] = time.time()
    
    def end_batch(self, epoch: int, batch: Optional[int] = None) -> None:
        """
        End timing a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int, optional): The batch number. If not provided, the last started batch number is used.
        """
        if batch is None:
            if self.__batch_starts[epoch]:
                batch: int = max(self.__batch_starts[epoch].keys())
            else:
                raise RuntimeError(f"no batch has started")
        self.__batch_ends[epoch][batch] = time.time()
    
    def time_epoch(self, epoch: int) -> float:
        """
        Get the duration of an epoch.

        Parameters:
            - epoch (int): The epoch number.

        Returns:
            - float: The duration of the epoch in seconds.
        """
        result: float = self.__epoch_ends[epoch] - self.__epoch_starts[epoch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"epoch {epoch} ends before starts")
    
    def time_batch(self, epoch: int, batch: int) -> float:
        """
        Get the duration of a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int): The batch number.

        Returns:
            - float: The duration of the batch in seconds.
        """
        result: float = self.__batch_ends[epoch][batch] - self.__batch_starts[epoch][batch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"batch {batch} in epoch {epoch} ends before starts")
        

class Logger:

    """
    A class used to log the training process.

    This class provides methods to log messages to a file and the console. 
    """
    def __init__(
        self, 
        logfile: str = f"{os.environ['PYTHONPATH']}/.log/{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
    ) -> None:
    
        """
        Initialize the logger.

        Parameters:
            - logfile (str, optional): The path to the logfile. 
            Defaults to a file in the .log directory with the current timestamp.
        """
        self.logfile: pathlib.Path = pathlib.Path(logfile)
        os.makedirs(name=self.logfile.parent, exist_ok=True)
        self._file: TextIO = open(self.logfile, mode='w')

    def log(
        self, 
        epoch: int, 
        n_epochs: int, 
        batch: Optional[int] = None, 
        n_batches: Optional[int] = None, 
        took: Optional[float] = None, 
        **kwargs: Any,
    ) -> None:
        """
        Log a message to console and a log file

        Parameters:
            - epoch (int): The current epoch.
            - n_epochs (int): The total number of epochs.
            - batch (int, optional): The current batch. Defaults to None.
            - n_batches (int, optional): The total number of batches. Defaults to None.
            - took (float, optional): The time it took to process the batch or epoch. Defaults to None.
            - **kwargs: Additional metrics to log.
        """
        suffix: str = ', '.join([f'{metric}: {value:.3e}' for metric, value in kwargs.items()])
        prefix: str = f'Epoch {epoch}/{n_epochs} | '
        if batch is not None:
            prefix += f'Batch {batch}/{n_batches} | '
        if took is not None:
            prefix += f'Took {took:.2f}s | '
        logstring: str = prefix + suffix
        print(logstring)
        self._file.write(logstring + '\n')

    def __del__(self) -> None:
        """
        Close the logfile at garbage collected.
        """
        self._file.close()


class CheckPointSaver:
    """
    A class used to save PyTorch model checkpoints.

    Attributes:
        - dirpath (pathlib.Path): The directory where the checkpoints are saved.
    """

    def __init__(self, dirpath: str) -> None:
        """
        Initialize the CheckPointSaver.

        Parameters:
            - dirpath (os.PathLike): The directory where the checkpoints are saved.
        """
        self.dirpath: pathlib.Path = pathlib.Path(dirpath)
        os.makedirs(name=self.dirpath, exist_ok=True)

    def save(self, model: nn.Module, filename: str) -> None:
        """
        Save checkpoint to a .pt file.

        Parameters:
            - model (nn.Module): The PyTorch model to save.
            - filename (str): the checkpoint file name
        """
        torch.save(obj=model, f=os.path.join(self.dirpath, filename))


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

    u_dim: int = state.shape[0]
    x_dim: int = state.shape[1]
    y_dim: int = state.shape[2]

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
) -> None:
    
    assert groundtruths.shape == predictions.shape
    assert groundtruths.ndim == 4
    assert groundtruths.shape[1] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got predictions.shape[1]={predictions.shape[1]}'
    )
    assert notes is None or len(notes) == groundtruths.shape[0]

    os.makedirs(f"{os.getenv('PYTHONPATH')}/results", exist_ok=True)

    groundtruths = groundtruths.to(device=torch.device('cpu'))
    predictions = predictions.to(device=torch.device('cpu'))

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruths.shape[2]
    y_res: int = groundtruths.shape[3]
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
            gt_field.squeeze(dim=0), 
            aspect=aspect_ratio, origin="lower", 
            extent=[-1., 1., -1., 1.], 
            vmin=vmin, vmax=vmax,
            cmap=cmap, 
        )
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(f'$groundtruth$', fontsize=20)
        axs[1].imshow(
            pred_field.squeeze(dim=0), 
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


if __name__ == '__main__':
    # from datasets import OneShotDiffReact2d
    # dataset = OneShotDiffReact2d(
    #     dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5'
    # )

    # timesteps = [[0, 5], [10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]

    # tensors = []
    # for start_step, end_step in timesteps:
    #     dataset = OneShotDiffReact2d(
    #         dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
    #         input_step=start_step,
    #         target_step=end_step,
    #     )
    #     index = 0
    #     input, target = dataset[index]
    #     tensors.extend([input, target])

    # plot_2d(
    #     *tensors,
    #     timesteps=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     dim_names=['u_1', 'u_2'],
    #     filename='sample.png'
    # )


    # plot_2d(
    #     input=input, 
    #     target=target, 
    #     dim_names=['u_1', 'u_2'],
    #     timesteps=[5, 10],
    #     filename='test.png',
    # )

    # TESTING `plot_predictions_2d`:
    from datasets import OneShotDiffReact2d
    from torch.utils.data import Subset, DataLoader
    from processes import loss_function

    dataset = OneShotDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
    )
    subset = Subset(dataset, indices=[990, 995])
    dataloader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

    batch_groundtruths: List[torch.Tensor] = []
    batch_predictions: List[torch.Tensor] = []

    metrics = Accumulator()

    with torch.no_grad():
        for batch, (batch_inputs, gt_targets) in enumerate(dataloader, start=1):
            print(batch_inputs.shape)
            pred_targets: torch.Tensor = batch_inputs
            batch_groundtruths.append(gt_targets)
            batch_predictions.append(pred_targets)
            loss: torch.Tensor = loss_function(predictions=pred_targets, groundtruth=gt_targets)
            metrics.add(val_mse=loss.item(), val_rmse=loss.item() ** 0.5)

    groundtruths = torch.cat(batch_groundtruths, dim=0)
    groundtruths = (groundtruths ** 2).sum(dim=1, keepdim=True) ** 0.5

    predictions = torch.cat(batch_predictions, dim=0)
    predictions = (predictions ** 2).sum(dim=1, keepdim=True) ** 0.5
    
    plot_predictions_2d(
        groundtruths=groundtruths, 
        predictions=predictions, 
        notes=['MSE: 0.3535, RMSE: 0.3245', 'MSE: 0.3535, RMSE: 0.3245'],
    )
    # plot_predictions_2d(groundtruths=groundtruths, predictions=predictions)







