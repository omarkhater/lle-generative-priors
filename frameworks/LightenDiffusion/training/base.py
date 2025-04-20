import torch
import copy
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Any
from torch.utils.data import DataLoader
import traceback
from abc import ABC, abstractmethod
import mlflow
import logging
from .tqdm_configuration import TqdmManager

class BaseTrainer(ABC):
    """
    Base trainer for Pytorch based neural networks.

    This class provides a standard training loop with optional validation,
    early stopping, and learning rate scheduling. It is designed to be
    inherited by specific model trainers that implement the training
    and validation logic.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        num_epochs (int): Maximum number of epochs to train.
        val_frequency (int): Frequency of validation (in epochs).
        patience (int): Number of epochs with no improvement after which training will be stopped.
        log_interval (int): Interval for logging training loss.

    Attributes:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        num_epochs (int): Maximum number of epochs to train.
        val_frequency (int): Frequency of validation (in epochs).
        patience (int): Number of epochs with no improvement after which training will be stopped.
        log_interval (int): Interval for logging training loss.
        best_loss (float): Best validation loss observed during training.
        best_epoch (int): Epoch at which the best validation loss was observed.
        best_state (Dict[str, Any]): State dictionary of the model at the best epoch.
        train_losses (List[float]): List of training losses for each epoch.
        val_losses (List[float]): List of validation losses for each epoch.

    Methods:

        train(): Main training loop.
        validate(): Validation loop.
        train_epoch(): Train for one epoch.
        validate_batch(): Validate a batch of data.
        calculate_loss(): Calculate loss for a batch of data.
        _gather_tensors(): Gather tensors to be used with calculate_loss.
        _check_early_stopping(): Check if early stopping criteria is met.
        after_validation(): Hook method called after validation is complete.
        cleanup_gpu(): Moves the model to CPU and clears GPU memory.
    
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        val_frequency: int = 5,
        patience: int = 5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.val_frequency = val_frequency
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self._validate_inputs(
            model, train_loader, val_loader, optimizer, device,
            scheduler, num_epochs, val_frequency, patience
        )

    def _validate_inputs(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        num_epochs: int,
        val_frequency: int,
        patience: int
    ):
        """
        Validates the inputs passed to the trainer.

        Raises:
            TypeError or ValueError: if any input is of the wrong type or invalid.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected 'model' to be an instance of nn.Module, got {type(model)}")
        if not isinstance(train_loader, DataLoader):
            raise TypeError(f"Expected 'train_loader' to be a DataLoader, got {type(train_loader)}")
        if not isinstance(val_loader, DataLoader):
            raise TypeError(f"Expected 'val_loader' to be a DataLoader, got {type(val_loader)}")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Expected 'optimizer' to be a torch.optim.Optimizer, got {type(optimizer)}")
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected 'device' to be a torch.device, got {type(device)}")
        if scheduler is not None:
            if not hasattr(scheduler, "step") or not callable(scheduler.step):
                raise TypeError(f"Expected 'scheduler' to have a callable 'step()' method, got {scheduler}")
            if not hasattr(scheduler, "state_dict") or not callable(scheduler.state_dict):
                raise TypeError(f"Expected 'scheduler' to have a callable 'state_dict()' method, got {scheduler}")
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer")
        if not isinstance(val_frequency, int) or val_frequency <= 0:
            raise ValueError("val_frequency must be a positive integer")
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        

    @abstractmethod
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        Must be implemented by the inherited class.
        """
        pass
    
    @abstractmethod
    def validate_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Validate a batch of data.
        Must be implemented by the inherited class.
        """
        pass

    @abstractmethod
    def calculate_loss(self, *args) -> float:
        """
        Calculate loss for a batch of data.
        Must be implemented by the inherited class.
        """
        pass

    @abstractmethod
    def _gather_tensors(self, *args) -> torch.Tensor:
        """
        Gather tensors to be used with calculate_loss.
        Must be implemented by the inherited class.
        """
        pass

    def train(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Main training loop with optional validation, early stopping, and LR scheduling.
        
        Returns:
            - model: The best trained model.
            - metrics: A dictionary containing training and validation losses.
        """
        try:
            prefix = "loss/"
            num_bad = 0
            epoch_bar = TqdmManager(
                total=self.num_epochs, 
                desc="Training Epochs", 
                leave=True,
                unit="epoch",
            )
            for epoch in range(self.num_epochs):
                loss_dict = self.train_epoch()

                train_loss = loss_dict.get("total_loss")
                if train_loss is None:
                    raise ValueError("Training loss not found in loss_dict. Does train_epoch return a key = total_loss?")
                
                self.train_losses.append(train_loss)
                epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")
                if mlflow.active_run():
                    mlflow.log_metric(f"{prefix}total", train_loss, step=epoch)
                    for key, value in loss_dict.items():
                        if key != "total_loss":
                            mlflow.log_metric(f"{prefix}{key}", value, step=epoch)
                if epoch > 0 and epoch % self.val_frequency == 0:
                    val_loss = self.validate(epoch+1)
                    self.val_losses.append(val_loss)
                    logging.info(f"Epoch {epoch+1}/{self.num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")                    
                    epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
                    stop, num_bad = self._check_early_stopping(epoch, val_loss, num_bad)
                    if stop:
                        break
                    if self.scheduler is not None:
                        logging.info(f"Step scheduler at epoch {epoch+1} with val_loss={val_loss:.4f}")
                        self.scheduler.step(val_loss)
                epoch_bar.update(1)
            
            epoch_bar.close()
            self.model.load_state_dict(self.best_state)
            metrics = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch + 1
            }
            return self.model, metrics
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()
            return None, {}
        finally:
            self.cleanup_gpu()
    
    def _check_early_stopping(self, epoch: int, val_loss: float, num_bad: int) -> Tuple[bool, int]:
        """
        Checks if the early stopping criteria is met and updates the best loss and model state.

        Args:
            epoch (int): The current epoch.
            val_loss (float): The current validation loss.
            num_bad (int): The current count of epochs without improvement.

        Returns:
            A tuple (stop, num_bad) where 'stop' is True if early stopping should be triggered,
            and num_bad is the updated count.
        """
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(self.model.state_dict())
            num_bad = 0
        else:
            num_bad += 1
            if num_bad >= self.patience:
                logging.info(f"Early stopping at epoch {epoch+1} after {num_bad} epochs without improvement.")
                return True, num_bad
        return False, num_bad

    def validate(self, current_epoch: int) -> float:
        """
        Validation loop that uses validate_batch to compute loss.
        Provides a standard validation workflow while allowing
        custom batch processing logic in child classes.

        Args:
            current_epoch (int): The current epoch number.
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_data = self.ensure_on_device(batch_data)
                loss = self.validate_batch(batch_data)
                running_loss += loss if isinstance(loss, float) else loss.item()
        
        self.after_validation(current_epoch)
        
        self.model.train()
        return running_loss / len(self.val_loader)

    def after_validation(self, current_epoch: int):
        """
        Hook method called after validation is complete.
        Child classes can override this to add visualizations
        or other post-validation processing.

        Args:
            current_epoch (int): The current epoch number.
        """
        pass

    def cleanup_gpu(self):
        """
        Moves the model to CPU and clears GPU memory.
        """
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_on_device(self, data):
        """
        Ensures the data is on the correct device.
        
        Args:
            data: The data to move to the device. Can be a tensor, list, tuple, or dict.
            
        Returns:
            The data on the correct device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.ensure_on_device(item) for item in data]
        elif isinstance(data, dict):
            return {k: self.ensure_on_device(v) for k, v in data.items()}
        else:
            return data
        
    def after_training(self):
        """
        Hook method called after training is complete.
        Child classes can override this to add visualizations
        or other post-training processing.
        """
        pass