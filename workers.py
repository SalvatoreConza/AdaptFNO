from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from utils.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from utils.plotting import plot_predictions_2d

class Trainer:

    def __init__(
        self, 
        model: nn.Module,
        optimizer: Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.model: nn.Module = model.to(device=device)
        self.optimizer: Optimizer = optimizer
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.device: torch.device = device

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = nn.MSELoss(reduction='mean')

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
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                self.optimizer.zero_grad()
                batch_prediction: torch.Tensor = self.model(input=batch_input)
                mse_loss = self.loss_function(input=batch_prediction, target=batch_groundtruth)
                mse_loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(
                    total_mse=mse_loss.item() * batch_prediction.shape[0], 
                    n_samples=batch_prediction.shape[0],
                )
                timer.end_batch(epoch=epoch)
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_mse=train_metrics['total_mse'] / train_metrics['n_samples'], 
                    train_rmse=(train_metrics['total_mse'] / train_metrics['n_samples']) ** 0.5, 
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
            val_mse, val_rmse = self.evaluate()
            timer.end_epoch(epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_mse=val_mse, val_rmse=val_rmse,
            )
            print('=' * 20)

            early_stopping(val_rmse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(self.model, filename=f'epoch{epoch}.pt')

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.model.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch, (batch_input, batch_groundtruth) in enumerate(self.val_dataloader, start=1):
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor = self.model(input=batch_input)
                mse_loss = self.loss_function(input=batch_prediction, target=batch_groundtruth)
                # Accumulate the val_metrics
                val_metrics.add(
                    total_mse=mse_loss.item() * batch_prediction.shape[0],
                    n_samples=batch_prediction.shape[0],
                )

        # Compute the aggregate metrics
        val_mse: float = val_metrics['total_mse'] / val_metrics['n_samples']
        val_rmse: float = val_mse ** 0.5
        return val_mse, val_rmse


class Predictor:

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model: nn.Module = model.to(device=device)
        self.device: torch.device = device
        self.loss_function: nn.Module = nn.MSELoss(reduction='mean')

    def predict(self, dataset: Dataset) -> None:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # sample-level method, not batch-level

        batch_groundtruths: List[torch.Tensor] = []
        batch_predictions: List[torch.Tensor] = []
        batch_metrics: List[str] = []

        with torch.no_grad():
            # Loop through each batch
            for batch_input, batch_groundtruth in dataloader:
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor = self.model(input=batch_input)
                
                mse_loss: torch.Tensor = self.loss_function(input=batch_prediction, target=batch_groundtruth)
                mse: float = mse_loss.item()
                rmse: float = mse ** 0.5

                batch_groundtruths.append(batch_groundtruth)
                batch_predictions.append(batch_prediction)
                batch_metrics.append(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}')

            predictions = torch.cat(tensors=batch_predictions, dim=0).to(device=self.device)
            groundtruths = torch.cat(tensors=batch_groundtruths, dim=0).to(device=self.device)
            assert predictions.shape == groundtruths.shape
            # Plot the prediction
            plot_predictions_2d(groundtruths=groundtruths, predictions=predictions)

