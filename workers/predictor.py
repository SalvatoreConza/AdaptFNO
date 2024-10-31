from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import partial
import datetime as dt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.amp import autocast, GradScaler

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.functional import compute_velocity_field
from common.plotting import plot_predictions_2d

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import ERA5_6Hour_Prediction



class GlobalOperatorPredictor:

    def __init__(self, global_operator: GlobalOperator):
        self.global_operator: GlobalOperator = global_operator.cuda().eval()
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')

    def predict(self, dataset: ERA5_6Hour_Prediction, plot_resolution: Tuple[int, int] | None) -> None:
        # Batch size should be 1 since len(dataset) == 1
        dataloader = DataLoader(dataset, batch_size=1)
        # Extract the single sample from the prediction dataset:
        global_input: torch.Tensor; global_groundtruth: torch.Tensor
        global_input, global_groundtruth = next(iter(dataloader))

        # Keep track of groundtruths and predictions
        timestamps: List[str] = []
        metric_notes: List[str] = []

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16):
                # Make one-step prediction
                global_prediction: torch.Tensor = self.global_operator(input=global_input)[0]
                assert global_prediction.shape == global_groundtruth.shape
                # Compute prediction timestamps
                prediction_timestamps: List[dt.datetime] = dataset.compute_out_timestamps()
                assert len(prediction_timestamps) == global_prediction.shape[1]
                # Compute metrics separately for each timestep
                for idx, prediction_timestamp in enumerate(tqdm(prediction_timestamps, desc=f"On date {dataset.ondate.strftime('%Y-%m-%d')}: ")):
                    global_prediction_t: torch.Tensor = global_prediction[:, idx, :, :, :]
                    global_groundtruth_t: torch.Tensor = global_groundtruth[:, idx, :, :, :]
                    total_mse: float = self.loss_function(input=global_prediction_t, target=global_groundtruth_t).item()
                    mean_mse: float = total_mse / global_prediction_t.numel()
                    mean_rmse: float = mean_mse ** 0.5

                    timestamps.append(f'{prediction_timestamp.strftime("%Y-%m-%d %H:00")}')
                    metric_notes.append(f'MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}')

        # Plot the prediction
        plot_predictions_2d(
            groundtruth=global_groundtruth.squeeze(dim=0), 
            prediction=global_prediction.squeeze(dim=0), 
            timestamps=timestamps,
            metrics_notes=metric_notes, 
            reduction=partial(compute_velocity_field, dim=1),
            resolution=plot_resolution,
        )


class LocalOperatorPredictor:

    def __init__(self, global_operator: GlobalOperator, local_operator: LocalOperator):
        self.global_operator: GlobalOperator = global_operator.cuda().eval()
        self.local_operator: LocalOperator = local_operator.cuda().eval()
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')
        self.train_H, self.train_W = self.local_operator.train_local_resolution

    def predict(self, dataset: ERA5_6Hour_Prediction, plot_resolution: Tuple[int, int] | None) -> None:
        # Batch size should be 1 since len(dataset) == 1
        dataloader = DataLoader(dataset, batch_size=1)
        # Extract the single sample from the prediction dataset:
        global_input: torch.Tensor; local_input: torch.Tensor; local_groundtruth: torch.Tensor
        global_input, _, local_input, local_groundtruth = next(iter(dataloader))
        local_input = self._downsample(local_input, size=(self.train_H, self.train_W))
        assert local_input.shape[-2:] == (self.train_H, self.train_W)

        # Keep track of groundtruths and predictions
        timestamps: List[str] = []
        metric_notes: List[str] = []
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16):
                # Make one-step prediction
                global_contexts: Tuple[torch.Tensor, ...]
                _, *global_contexts = self.global_operator(input=global_input)
                # print(local_groundtruth.shape)
                local_prediction: torch.Tensor = self.local_operator(
                    input=local_input, global_contexts=list(global_contexts), out_resolution=local_groundtruth.shape[-2:],
                )
                assert local_prediction.shape == local_groundtruth.shape
                # Compute prediction timestamps
                prediction_timestamps: List[dt.datetime] = dataset.compute_out_timestamps()
                assert len(prediction_timestamps) == local_prediction.shape[1]
                # Compute metrics separately for each timestep
                for idx, prediction_timestamp in enumerate(tqdm(prediction_timestamps, desc=f"On date {dataset.ondate.strftime('%Y-%m-%d')}: ")):
                    local_prediction_t: torch.Tensor = local_prediction[:, idx, :, :, :]
                    local_groundtruth_t: torch.Tensor = local_groundtruth[:, idx, :, :, :]
                    total_mse: float = self.loss_function(input=local_prediction_t, target=local_groundtruth_t).item()
                    mean_mse: float = total_mse / local_prediction_t.numel()
                    mean_rmse: float = mean_mse ** 0.5

                    timestamps.append(f'{prediction_timestamp.strftime("%Y-%m-%d %H:00")}')
                    metric_notes.append(f'MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}')

        # Plot the prediction
        plot_predictions_2d(
            groundtruth=local_groundtruth.squeeze(dim=0), 
            prediction=local_prediction.squeeze(dim=0), 
            timestamps=timestamps,
            metrics_notes=metric_notes, 
            reduction=partial(compute_velocity_field, dim=1),
            resolution=plot_resolution,
        )

    def _downsample(self, input: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        assert input.ndim == 5
        output: torch.Tensor = F.interpolate(input.flatten(start_dim=0, end_dim=1), size=size, mode='bicubic')
        output = output.reshape(input.shape[0], input.shape[1], input.shape[2], size[0], size[1])
        return output

