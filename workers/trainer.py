import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from common.losses import TemporalMSE
from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from models.operators import GlobalOperator, LocalOperator
from models.benchmarks import FNO3D
from era5.datasets import ERA5_6Hour


class MainTrainer:

    def __init__(
        self, 
        local_operator: LocalOperator,
        global_operator: GlobalOperator,
        train_dataset: ERA5_6Hour,
        val_dataset: ERA5_6Hour,
        train_batch_size: int,
        val_batch_size: int,
        learning_rate: float,
    ):
        self.train_dataset: ERA5_6Hour = train_dataset
        self.val_dataset: ERA5_6Hour = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.learning_rate: float = learning_rate

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = TemporalMSE(n_timesteps=train_dataset.out_timesteps, reduction='sum')

        if torch.cuda.device_count() > 1:
            self.global_operator: GlobalOperator = nn.DataParallel(global_operator).cuda()
            self.local_operator: GlobalOperator = nn.DataParallel(local_operator).cuda()
        elif torch.cuda.device_count() == 1:
            self.global_operator: GlobalOperator = global_operator.cuda()
            self.local_operator: LocalOperator = local_operator.cuda()
        else:
            self.global_operator: GlobalOperator = global_operator
            self.local_operator: LocalOperator = local_operator

        self.train_H, self.train_W = self.local_operator.spatial_resolution
        self.learning_rate: float = learning_rate
        self.optimizer = Adam(params=chain(global_operator.parameters(), local_operator.parameters()), lr=learning_rate)

    def train(self, n_epochs: int, patience: int, tolerance: float, checkpoint_path: Optional[str] = None, save_frequency: int = 5) -> None:
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        global_saver = CheckpointSaver(model=self.global_operator, dirpath=os.path.join(checkpoint_path, 'global'))
        local_saver = CheckpointSaver(model=self.local_operator, dirpath=os.path.join(checkpoint_path, 'local'))
        self.global_operator.train(); self.local_operator.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            for batch, (batch_global_input, batch_local_input, batch_local_groundtruth) in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}/{n_epochs}: '), start=1):
                timer.start_batch(epoch, batch)
                assert batch_local_input.ndim == 5
                batch_size, in_timesteps, in_channels = batch_local_input.shape[:3]
                batch_size, out_timesteps, out_channels = batch_local_groundtruth.shape[:3]

                batch_local_input = self._downsample(input=batch_local_input, size=(self.train_H, self.train_W))
                assert batch_local_input.shape == (batch_size, in_timesteps, in_channels, self.train_H, self.train_W)
                batch_local_groundtruth = self._downsample(input=batch_local_groundtruth, size=(self.train_H, self.train_W))
                assert batch_local_groundtruth.shape == (batch_size, out_timesteps, out_channels, self.train_H, self.train_W)

                self.optimizer.zero_grad()
                batch_global_contexts: Tuple[torch.Tensor, ...] = self.global_operator(input=batch_global_input)
                batch_local_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_contexts=list(batch_global_contexts),
                )
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(input=batch_local_prediction, target=batch_local_groundtruth)
                mean_mse_loss: torch.Tensor = total_mse_loss / batch_local_prediction.numel()
                # Backpropagation
                mean_mse_loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())
                timer.end_batch(epoch=epoch)
                # Log
                mean_train_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_rmse=mean_train_mse ** 0.5, train_mse=mean_train_mse, 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                global_saver.save(model_states=self.global_operator.state_dict(), filename=f'epoch{epoch}.pt')
                local_saver.save(model_states=self.local_operator.state_dict(), filename=f'epoch{epoch}.pt')
            
            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            del batch_global_input, batch_local_input, batch_local_groundtruth
            val_rmse, val_mse = self.evaluate()
            timer.end_epoch(epoch)
            # Log
            logger.log(epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), val_rmse=val_rmse, val_mse=val_mse)
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            global_saver.save(model_states=self.global_operator.state_dict(), filename=f'epoch{epoch}.pt')
            local_saver.save(model_states=self.local_operator.state_dict(), filename=f'epoch{epoch}.pt')
            

    def evaluate(self) -> Tuple[float, float]:
        val_metrics = Accumulator()
        self.global_operator.eval(); self.local_operator.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_global_input, batch_local_input, batch_local_groundtruth in self.val_dataloader:
                assert batch_local_input.ndim == 5
                batch_size, in_timesteps, in_channels, raw_H, raw_W = batch_local_input.shape
                batch_size, out_timesteps, out_channels, raw_H, raw_W = batch_local_groundtruth.shape

                batch_local_input = self._downsample(input=batch_local_input, size=(self.train_H, self.train_W))
                assert batch_local_input.shape == (batch_size, in_timesteps, in_channels, self.train_H, self.train_W)
                batch_local_groundtruth = self._downsample(input=batch_local_groundtruth, size=(self.train_H, self.train_W))
                assert batch_local_groundtruth.shape == (batch_size, out_timesteps, out_channels, self.train_H, self.train_W)
                
                # Forward propagation
                batch_global_contexts: Tuple[torch.Tensor, ...] = self.global_operator(input=batch_global_input)
                batch_local_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_contexts=list(batch_global_contexts),
                )
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(
                    input=batch_local_prediction, target=batch_local_groundtruth,
                )
                # Accumulate the val_metrics
                val_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())

        # Compute the aggregate metrics
        val_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        val_rmse: float = val_mse ** 0.5
        return val_rmse, val_mse

    def _downsample(self, input: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        assert input.ndim == 5
        output: torch.Tensor = F.interpolate(input.flatten(start_dim=0, end_dim=1), size=size, mode='bicubic')
        output = output.reshape(input.shape[0], input.shape[1], input.shape[2], size[0], size[1])
        return output


class BenchmarkTrainer:

    def __init__(
        self, 
        net: FNO3D,
        train_dataset: ERA5_6Hour,
        val_dataset: ERA5_6Hour,
        train_batch_size: int,
        val_batch_size: int,
        learning_rate: float,
    ):
        self.train_dataset: ERA5_6Hour = train_dataset
        self.val_dataset: ERA5_6Hour = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.learning_rate: float = learning_rate

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = TemporalMSE(n_timesteps=train_dataset.out_timesteps, reduction='sum')

        if torch.cuda.device_count() > 1:
            self.net: FNO3D = nn.DataParallel(net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net: FNO3D = net.cuda()
        else:
            self.net: FNO3D = net

        self.learning_rate: float = learning_rate
        self.optimizer = Adam(params=net.parameters(), lr=learning_rate)

    def train(self, n_epochs: int, patience: int, tolerance: float, checkpoint_path: Optional[str] = None, save_frequency: int = 5) -> None:
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(model=self.net, dirpath=os.path.join(checkpoint_path, 'benchmarks'))
        self.net.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            for batch, (_, batch_local_input, batch_local_groundtruth) in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}/{n_epochs}: '), start=1):
                timer.start_batch(epoch, batch)
                assert batch_local_input.ndim == 5
                self.optimizer.zero_grad()
                batch_local_prediction: torch.Tensor = self.net(input=batch_local_input)
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(input=batch_local_prediction, target=batch_local_groundtruth)
                mean_mse_loss: torch.Tensor = total_mse_loss / batch_local_prediction.numel()
                # Backpropagation
                mean_mse_loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())
                timer.end_batch(epoch=epoch)
                # Log
                mean_train_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_rmse=mean_train_mse ** 0.5, train_mse=mean_train_mse, 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(model_states=self.net.state_dict(), filename=f'fno3d_epoch{epoch}.pt')
            
            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            del batch_global_input, batch_local_input, batch_local_groundtruth
            val_rmse, val_mse = self.evaluate()
            timer.end_epoch(epoch)
            # Log
            logger.log(epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), val_rmse=val_rmse, val_mse=val_mse)
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(model_states=self.net.state_dict(), filename=f'fno3d_epoch{epoch}.pt')

    def evaluate(self) -> Tuple[float, float]:
        val_metrics = Accumulator()
        self.net.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_global_input, batch_local_input, batch_local_groundtruth in self.val_dataloader:
                assert batch_local_input.ndim == 5
                # Forward propagation
                batch_local_prediction: torch.Tensor = self.net(input=batch_local_input)
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(
                    input=batch_local_prediction, target=batch_local_groundtruth,
                )
                # Accumulate the val_metrics
                val_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())

        # Compute the aggregate metrics
        val_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        val_rmse: float = val_mse ** 0.5
        return val_rmse, val_mse
