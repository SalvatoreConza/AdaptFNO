from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import partial
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import RegularizedPowerError
from common.functional import compute_velocity_field
from common.plotting import plot_predictions_2d

from models.operators import GlobalOperator, LocalOperator
from era5.datasets import Wind2dERA5



class GlobalOperatorPredictor:

    def __init__(
        self, 
        global_operator: GlobalOperator,
        device: torch.device,
    ):
        self.device: torch.device = device
        self.global_operator: GlobalOperator = global_operator.to(device=self.device)
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum').to(device=self.device)

    def predict(self, dataset: Wind2dERA5, plot_resolution: Tuple[int, int] | None) -> None:
        # Set the operator to evaluation mode
        self.global_operator.eval()
        # Batch size must be 1, must not shuffle
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # Use only the first input from the test dataset:
        global_input: torch.Tensor = next(iter(dataloader))[0].to(device=self.device)
        # Compute how many bundle from last step to keep 
        n_retained_bundles: int = dataset.bundle_size - 1   # since each step predicts 1 bundle   

        # Keep track of groundtruths and predictions
        global_groundtruths: List[torch.Tensor] = []
        global_predictions: List[torch.Tensor] = []
        timestamps: List[str] = []
        metric_notes: List[str] = []

        with torch.no_grad():
            # Make multi-step prediction
            for bundle_id, (_, global_groundtruth) in enumerate(dataloader):
                # Make one-step prediction
                global_prediction: torch.Tensor = self.global_operator(input=global_input)[0]
                global_groundtruth: torch.Tensor = global_groundtruth.to(device=self.device)
                assert global_prediction.shape == global_groundtruth.shape
                global_predictions.append(global_prediction)
                global_groundtruths.append(global_groundtruth)
                # Prepare input for next step
                if n_retained_bundles == 0:
                    global_input = global_prediction
                else: 
                    global_input = torch.cat(
                        tensors=[
                            global_input[:, -n_retained_bundles * dataset.bundle_size:, :, :, :],
                            global_prediction,
                        ],
                        dim=1
                    )
                # Compute prediction timestamps
                prediction_timestamps: List[dt.datetime] = dataset.compute_timestamp(bundle_idx=bundle_id)[1]
                # Compute metrics
                total_mse: float = self.loss_function(input=global_prediction, target=global_groundtruth).item()
                mean_mse: float = total_mse / global_prediction.numel()
                mean_rmse: float = mean_mse ** 0.5
                timestamps.extend([f'{t.strftime("%Y-%m-%d %H:00")}' for t in prediction_timestamps])
                metric_notes.extend([f'MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}'] * dataset.bundle_size)

        global_predictions: torch.Tensor = torch.cat(tensors=global_predictions, dim=1).squeeze(dim=0)
        global_groundtruths: torch.Tensor = torch.cat(tensors=global_groundtruths, dim=1).squeeze(dim=0)
        assert global_predictions.shape[0] == global_groundtruths.shape[0] == len(metric_notes)

        # Plot the prediction
        plot_predictions_2d(
            groundtruths=global_groundtruths, 
            predictions=global_predictions, 
            timestamps=timestamps,
            metrics_notes=metric_notes, 
            reduction=partial(compute_velocity_field, dim=1),
            resolution=plot_resolution,
        )


class LocalOperatorPredictor:

    def __init__(
        self, 
        global_operator: GlobalOperator,
        local_operator: LocalOperator,
        device: torch.device,
    ):
        self.device: torch.device = device
        self.global_operator: GlobalOperator = global_operator.to(device=self.device)
        self.local_operator: LocalOperator = local_operator.to(device=self.device)
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')

    def predict(self, dataset: Wind2dERA5, plot_resolution: Tuple[int, int] | None) -> None:
        # Set the operator to evaluation mode
        self.global_operator.eval()
        self.local_operator.eval()
        # Batch size must be 1, must not shuffle
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # Use only the first input from the test dataset:
        first_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = next(iter(dataloader))
        global_input: torch.Tensor = first_input[0].to(device=self.device)
        local_input: torch.Tensor = first_input[2].to(device=self.device)
        # Compute how many bundle from last step to keep 
        n_retained_bundles: int = dataset.bundle_size - 1   # since each step predicts 1 bundle

        local_groundtruths: List[torch.Tensor] = []
        local_predictions: List[torch.Tensor] = []
        timestamps: List[str] = []
        metric_notes: List[str] = []

        with torch.no_grad():
            # Make multi-step prediction
            for bundle_id, (_, _, _, local_groundtruth) in enumerate(dataloader):
                # Make one-step prediction
                global_contexts: Tuple[torch.Tensor, ...]
                global_prediction, *global_contexts = self.global_operator(input=global_input)
                local_prediction: torch.Tensor = self.local_operator(
                    input=local_input, global_contexts=list(global_contexts)
                )
                local_groundtruth: torch.Tensor = local_groundtruth.to(device=self.device)
                assert local_prediction.shape == local_groundtruth.shape
                local_groundtruths.append(local_groundtruth)
                local_predictions.append(local_prediction)
                # Prepare input for next step
                if n_retained_bundles == 0:
                    global_input = global_prediction
                    local_input = local_prediction
                else: 
                    global_input = torch.cat(
                        tensors=[
                            global_input[:, -n_retained_bundles * dataset.bundle_size:, :, :, :],
                            global_prediction,
                        ],
                        dim=1
                    )
                    local_input = torch.cat(
                        tensors=[
                            local_input[:, -n_retained_bundles * dataset.bundle_size:, :, :, :],
                            local_prediction,
                        ],
                        dim=1
                    )
                # Compute prediction timestamps
                prediction_timestamps: List[dt.datetime] = dataset.compute_timestamp(bundle_idx=bundle_id)[1]
                # Compute metrics
                total_mse: float = self.loss_function(input=local_prediction, target=local_groundtruth).item()
                mean_mse: float = total_mse / local_prediction.numel()
                mean_rmse: float = mean_mse ** 0.5
                timestamps.extend([f'{t.strftime("%Y-%m-%d %H:00")}' for t in prediction_timestamps])
                metric_notes.extend([f'MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}'] * dataset.bundle_size)

        local_predictions: torch.Tensor = torch.cat(tensors=local_predictions, dim=1).squeeze(dim=0)
        local_groundtruths: torch.Tensor = torch.cat(tensors=local_groundtruths, dim=1).squeeze(dim=0)
        assert local_predictions.shape[0] == local_groundtruths.shape[0] == len(metric_notes)

        # Plot the prediction
        plot_predictions_2d(
            groundtruths=local_groundtruths, 
            predictions=local_predictions, 
            timestamps=timestamps,
            metrics_notes=metric_notes, 
            reduction=partial(compute_velocity_field, dim=1),
            resolution=plot_resolution,
        )


