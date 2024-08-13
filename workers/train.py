from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import RegularizedPowerError

from models.operators import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5


class _BaseOperatorTrainer(ABC):

    def __init__(
        self, 
        optimizer: Optimizer,
        spectral_regularization_coef: float,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        self.optimizer: Optimizer = optimizer
        self.spectral_regularization_coef: float = spectral_regularization_coef
        self.noise_level: float = noise_level
        self.train_dataset: Wind2dERA5 = train_dataset
        self.val_dataset: Wind2dERA5 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.device: torch.device = device

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
        )
        self.loss_function: nn.Module = RegularizedPowerError(
            lambda_=spectral_regularization_coef, 
            power=2,
        )
    
    @abstractmethod
    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        pass


class GlobalOperatorTrainer(_BaseOperatorTrainer):

    def __init__(
        self, 
        global_operator: GlobalOperator,
        optimizer: Optimizer,
        spectral_regularization_coef: float,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        super().__init__(
            optimizer=optimizer, 
            spectral_regularization_coef=spectral_regularization_coef, 
            noise_level=noise_level, 
            train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            device=device,
        )
        self.global_operator: GlobalOperator = global_operator.to(device)

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
            model=self.global_operator,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.global_operator.train()
        
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
                _, batch_prediction = self.global_operator(input=batch_input)
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.global_operator.spectral_convolutions
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
                    model_states=self.global_operator.state_dict(), 
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
                model_states=self.global_operator.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> Tuple[float, float, float, float]:
        val_metrics = Accumulator()
        self.global_operator.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_input, batch_groundtruth in self.val_dataloader:
                assert batch_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_input.shape
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor
                _, batch_prediction = self.global_operator(input=batch_input)
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.global_operator.spectral_convolutions
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



class LocalOperatorTrainer(_BaseOperatorTrainer):

    def __init__(
        self, 
        local_operator: LocalOperator,
        global_operator: GlobalOperator,
        optimizer: Optimizer,
        spectral_regularization_coef: float,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        super().__init__(
            optimizer=optimizer, 
            spectral_regularization_coef=spectral_regularization_coef, 
            noise_level=noise_level, 
            train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            device=device,
        )
        self.local_operator: LocalOperator = local_operator.to(device)
        self.global_operator: GlobalOperator = global_operator.to(device)

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
            model=self.local_operator,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.local_operator.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (
                batch_global_input, _, 
                batch_local_input, batch_local_groundtruth
            ) in enumerate(self.train_dataloader, start=1):
                
                timer.start_batch(epoch, batch)
                assert batch_local_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_local_input.shape
                batch_local_input: torch.Tensor = batch_local_input.to(device=self.device)
                batch_local_input: torch.Tensor = (
                    batch_local_input 
                    + torch.randn(size=batch_local_input.shape, device=self.device) * batch_local_input.std() * self.noise_level
                )
                batch_local_groundtruth: torch.Tensor = batch_local_groundtruth.to(device=self.device)
                self.optimizer.zero_grad()

                batch_global_context: torch.Tensor
                batch_global_context, _ = self.global_operator(input=batch_global_input)
                batch_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_context=batch_global_context,
                )
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.local_operator.spectral_convolutions
                    ],
                    prediction=batch_prediction, 
                    groundtruth=batch_local_groundtruth,
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
                    model_states=self.local_operator.state_dict(), 
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
                model_states=self.local_operator.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> Tuple[float, float, float, float]:
        val_metrics = Accumulator()
        self.local_operator.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch, (
                batch_global_input, _, 
                batch_local_input, batch_local_groundtruth
            ) in enumerate(self.train_dataloader, start=1):
                assert batch_local_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_local_input.shape
                batch_local_input: torch.Tensor = batch_local_input.to(device=self.device)
                batch_local_groundtruth: torch.Tensor = batch_local_groundtruth.to(device=self.device)

                batch_global_context: torch.Tensor
                batch_global_context, _ = self.global_operator(input=batch_global_input)
                batch_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_context=batch_global_context,
                )
                mean_scaled_error, mean_power_error, mean_weight_magnitude, loss = self.loss_function(
                    spectral_weights=[
                        layer.Ws.weights for layer in self.local_operator.spectral_convolutions
                    ],
                    prediction=batch_prediction, 
                    groundtruth=batch_local_groundtruth,
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



