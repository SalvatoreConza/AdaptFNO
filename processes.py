from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from utils import Accumulator, EarlyStopping, Timer, Logger, CheckPointSaver, plot_predictions_2d


def loss_function(predictions: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(input=predictions, target=groundtruth, reduction='mean')

def train(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: Optimizer,
    train_batch_size: int,
    val_batch_size: int,
    n_epochs: int,
    patience: int,
    tolerance: float,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    train_metrics = Accumulator()
    early_stopping = EarlyStopping(patience, tolerance)
    timer = Timer()
    logger = Logger()
    checkpoint_saver = CheckPointSaver(dirpath=checkpoint_path)
    model.train()

    # loop through each epoch
    for epoch in range(1, n_epochs + 1):
        timer.start_epoch(epoch)
        # Loop through each batch
        for batch, (batch_inputs, gt_targets) in enumerate(train_dataloader, start=1):
            timer.start_batch(epoch, batch)
            optimizer.zero_grad()
            pred_targets: torch.Tensor = model(batch_inputs)
            loss: torch.Tensor = loss_function(predictions=pred_targets, groundtruth=gt_targets)
            loss.backward()
            optimizer.step()

            # Accumulate the metrics
            train_metrics.add(mse=loss.item(), rmse=loss.item() ** 0.5)
            timer.end_batch(epoch=epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                batch=batch, n_batches=len(train_dataloader), 
                took=timer.time_batch(epoch, batch),
                train_mse=train_metrics['mse'] / batch, 
                train_rmse=train_metrics['rmse'] / batch,
            )

        # Save checkpoint
        if checkpoint_path:
            checkpoint_saver.save(model, filename=f'epoch{epoch}.pt')
        
        # Reset metric records for next epoch
        train_metrics.reset()
        
        # Evaluate
        val_mse, val_rmse = evaluate(model=model, dataloader=val_dataloader)
        timer.end_epoch(epoch)
        logger.log(
            epoch=epoch, n_epochs=n_epochs, 
            took=timer.time_epoch(epoch), 
            val_mse=val_mse, val_loss=val_rmse)
        print('='*20)

        early_stopping(val_rmse)
        if early_stopping:
            print('Early Stopped')
            break

    return model

    
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float, float]:
    metrics = Accumulator()
    model.eval()

    with torch.no_grad():
        # Loop through each batch
        for batch, (batch_inputs, gt_targets) in enumerate(dataloader, start=1):
            pred_targets: torch.Tensor = model(batch_inputs)
            loss: torch.Tensor = loss_function(predictions=pred_targets, groundtruth=gt_targets)
            # Accumulate the metrics
            metrics.add(val_mse=loss.item(), val_rmse=loss.item() ** 0.5)

    # Compute the aggregate metrics
    return metrics['val_mse'] / batch, metrics['val_rmse'] / batch
    

def predict(model: nn.Module, dataloader: DataLoader) -> None:
    assert dataloader.batch_size == 1, (
        f'This function is sample-level, not batch-level, '
        f'batch_size must set to 1, got {dataloader.batch_size}'
    )
    model.eval()
    batch_groundtruths: List[torch.Tensor] = []
    batch_predictions: List[torch.Tensor] = []
    metrics: List[str] = []
    with torch.no_grad():
        for batch, (batch_inputs, gt_targets) in enumerate(dataloader, start=1):
            pred_targets: torch.Tensor = model(batch_inputs)
            batch_groundtruths.append(gt_targets)
            batch_predictions.append(pred_targets)
            loss: torch.Tensor = loss_function(predictions=pred_targets, groundtruth=gt_targets)
            mse: float = loss.item()
            rmse: float = mse ** 0.5
            metrics.append(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}')

    # Express as a single physical field
    groundtruths: torch.Tensor = torch.cat(batch_groundtruths, dim=0)
    groundtruths: torch.Tensor = (groundtruths ** 2).sum(dim=1, keepdim=True) ** 0.5
    predictions: torch.Tensor = torch.cat(batch_predictions, dim=0)
    predictions: torch.Tensor = (predictions ** 2).sum(dim=1, keepdim=True) ** 0.5

    plot_predictions_2d(
        groundtruths=groundtruths.to(device=torch.device('cpu')), 
        predictions=predictions.to(device=torch.device('cpu')),
        notes=metrics,
    )







