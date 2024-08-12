from typing import List, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.plotting import plot_predictions_2d
from common.functional import compute_velocity_field
from common.losses import RegularizedPowerError

from models.windnet import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5



class GlobalOperatorTrainer:

    def __init__(
        self, 
        model: GlobalOperator,
        optimizer: Optimizer,
        spectral_regularization_coef: float,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        self.model: nn.Module = model.to(device=device)
        self.optimizer: Optimizer = optimizer
        self.spectral_regularization_coef: float = spectral_regularization_coef
        self.noise_level: float = noise_level
        self.train_dataset: Wind2dERA5 = train_dataset
        self.val_dataset: Wind2dERA5 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.device: torch.device = device

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = RegularizedPowerError(lambda_=spectral_regularization_coef, power=2)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(
            model=self.model,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.model.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (batch_input, batch_groundtruth) in enumerate(self.train_dataloader, start=1):
                timer.start_batch(epoch, batch)
                assert batch_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_input.shape
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_input: torch.Tensor = (
                    batch_input + torch.randn(size=batch_input.shape, device=self.device) * batch_input.std() * self.noise_level
                )
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                self.optimizer.zero_grad()
                batch_prediction: torch.Tensor
                _, batch_prediction = self.model(input=batch_input)
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.model.spectral_convolutions
                    ],
                    prediction=batch_prediction, 
                    groundtruth=batch_groundtruth,
                )
                loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(
                    total_scaled_error=mean_scaled_error.item() * batch_size,
                    total_power_error=mean_power_error.item() * batch_size, 
                    total_weight_magnitude=mean_weight_magnitude.item() * batch_size,
                    total_loss=loss.item() * batch_size,
                    n_samples=batch_size,
                )
                timer.end_batch(epoch=epoch)
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_scaled_error=train_metrics['total_scaled_error'] / train_metrics['n_samples'], 
                    train_power_error=train_metrics['total_power_error'] / train_metrics['n_samples'], 
                    train_weight_magnitude=train_metrics['total_weight_magnitude'] / train_metrics['n_samples'], 
                    train_loss=train_metrics['total_loss'] / train_metrics['n_samples'], 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.model.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'epoch{epoch}.pt',
                )
            
            # Reset metric records for next epoch
            train_metrics.reset()
            
            # Evaluate
            val_scaled_error, val_power_error, val_weight_magnitude, val_loss = self.evaluate()
            timer.end_epoch(epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_scaled_error=val_scaled_error,
                val_power_error=val_power_error, 
                val_weight_magnitude=val_weight_magnitude,
                val_loss=val_loss,
            )
            print('=' * 20)

            early_stopping(value=val_power_error)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(
                model_states=self.model.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.model.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_input, batch_groundtruth in self.val_dataloader:
                assert batch_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_input.shape
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor
                _, batch_prediction = self.model(input=batch_input)
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.model.spectral_convolutions
                    ],
                    prediction=batch_prediction, 
                    groundtruth=batch_groundtruth,
                )
                # Accumulate the val_metrics
                val_metrics.add(
                    total_scaled_error=mean_scaled_error.item() * batch_size,
                    total_power_error=mean_power_error.item() * batch_size, 
                    total_weight_magnitude=mean_weight_magnitude.item() * batch_size,
                    total_loss=loss.item() * batch_size,
                    n_samples=batch_size,
                )

        # Compute the aggregate metrics
        val_scaled_error: float = val_metrics['total_scaled_error'] / val_metrics['n_samples']
        val_power_error: float = val_metrics['total_power_error'] / val_metrics['n_samples']
        val_weight_magnitude: float = val_metrics['total_weight_magnitude'] / val_metrics['n_samples']
        val_loss: float = val_metrics['total_loss'] / val_metrics['n_samples']
        return val_scaled_error, val_power_error, val_weight_magnitude, val_loss



