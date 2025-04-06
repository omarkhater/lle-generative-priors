import torch
from tqdm import tqdm
import copy
from typing import Tuple, Dict, List, Optional

def train_model(
    model: torch.nn.Module,
    data_loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    criterion,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_frequency: int = 3,
    patience: int = 5
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a PyTorch model using the given data loaders, loss function, optimizer, and device.

    Args:
        model (torch.nn.Module):
            The model to train.
        data_loaders (Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]):
            A tuple containing the training and validation data loaders.
        criterion:
            The loss function to use. It should accept the model's outputs and target tensors.
            For example, if the model returns a tuple (estimated_reflectance, estimated_illumination)
            and the target is a raw image, you can use a wrapper (e.g., ctdn_loss_wrapper) that
            calls your CDTN loss function.
        optimizer (torch.optim.Optimizer):
            The optimizer to use.
        num_epochs (int):
            The number of epochs to train the model.
        batch_size (int):
            The batch size to use.
        device (torch.device):
            The device to use for training.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
            The learning rate scheduler to use. Default is None.
        val_frequency (int):
            Frequency (in epochs) to run validation. Default is 3.
        patience (int):
            Number of epochs with no improvement after which training will be stopped early.
            Default is 5.

    Returns:
        Tuple[torch.nn.Module, Dict[str, List[float]]]:
            A tuple containing:
                - The trained model.
                - A dictionary of losses with keys:
                    'train': List of average training losses per epoch.
                    'val': List of average validation losses per epoch.
                    'iteration': A dictionary mapping iteration counts to average loss.
                    'best_loss': The best validation loss achieved.
                    'best_epoch': The epoch at which the best validation loss was achieved.
    """
    train_loader, val_loader = data_loaders
    train_losses: List[float] = []
    val_losses: List[float] = []
    iteration_loss: Dict[int, float] = {}
    best_epoch: int = 0
    best_loss: float = float('inf')
    iteration_count: int = 0
    interval: int = max(5000 // batch_size, 1)
    best_model_state = copy.deepcopy(model.state_dict())
    num_bad_epochs: int = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        num_batches: int = len(train_loader)
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as train_bar:
            for _, (inputs, targets) in enumerate(train_bar):
                inputs = inputs.to(device)  # Expected shape: (B, 3, H, W)
                targets = targets.to(device)  # Expected shape: (B, 3, H, W)
                outputs = model(inputs)  # Expected to return a tuple, e.g., (estimated_reflectance, estimated_illumination)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                iteration_count += 1
                train_bar.set_postfix(loss=loss.item())
                if iteration_count % interval == 0:
                    avg_iter_loss = running_loss / iteration_count
                    iteration_loss[iteration_count] = avg_iter_loss

        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        if epoch % val_frequency == 0:
            avg_val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Epoch: {epoch}, avg_val_loss = {avg_val_loss:.8f}")
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_loss - 1e-3:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            
    model.load_state_dict(best_model_state)
    
    losses = {
        'train': train_losses,
        'val': val_losses,
        'iteration': iteration_loss,
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }

    print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss}")
    return model, losses

def validate_model(
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        device: torch.device
    ):
    """
    Validate pytorch model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to use for training.

    """
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader: 
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss

def validate_unsupervised(
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        device: torch.device
    ):
    """
    Validate unsupervised model on the validation set.

    Each validation sample is expected to contain two low images (shape: (B, 2, C, H, W)):
    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to use for training.
    """
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for low_imgs, _ in val_loader:
            low_imgs = low_imgs.to(device)
            input_imgs = low_imgs[:, 0, ...]   # Use first low image as input
            target_imgs = low_imgs[:, 1, ...]  # Use second low image as target
            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss

def get_optimizer(
        model: torch.nn.Module, 
        lr:float = 1e-3, 
        weight_decay:float =1e-5
    )-> torch.optim.Optimizer:
    """
    Get Adam optimizer for the model.

    Args:
        model (torch.nn.Module): The model for which to get the optimizer.
        lr (float), Optional (default=1e-3): The learning rate for the optimizer.
        weight_decay (float), Optional (default=1e-5): The weight decay for the optimizer.

    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(
        optimizer: torch.optim.Optimizer, 
        gamma:float=0.9
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:

    """
    Get learning rate scheduler for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to get the scheduler.
        gamma (float), Optional (default=0.9): The factor by which to reduce the learning rate.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        factor=gamma, 
        patience=5
    )

def train_unsupervised(
    model: torch.nn.Module,
    data_loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_frequency: int = 3,
    patience: int = 5
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a PyTorch model using paired low-light images in an unsupervised manner.
    
    Each training sample is expected to contain two low images (shape: (B, 2, C, H, W)):
      - The first image is used as input.
      - The second image is used as the reconstruction target.
    The provided label (used for supervised evaluation) is ignored during training.
    
    Args:
        model: The model to train.
        data_loaders: Tuple containing the training and validation DataLoaders.
        criterion: The loss function (e.g., L1, L2) comparing model output and target.
        optimizer: The optimizer.
        num_epochs: Number of epochs to train.
        batch_size: Batch size.
        device: The device to use.
        scheduler: Optional learning rate scheduler.
        val_frequency: Frequency (in epochs) for running validation.
        patience: Number of epochs with no improvement before early stopping.
        
    Returns:
        A tuple containing:
          - The best model (based on validation loss).
          - A dictionary with loss metrics (train, val, iteration, best_loss, best_epoch).
    """
    train_loader, val_loader = data_loaders
    train_losses: List[float] = []
    val_losses: List[float] = []
    iteration_loss: Dict[int, float] = {}
    best_epoch: int = 0
    best_loss: float = float('inf')
    iteration_count: int = 0
    interval: int = max(5000 // batch_size, 1)
    best_model_state = copy.deepcopy(model.state_dict())
    num_bad_epochs: int = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        num_batches: int = len(train_loader)
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as train_bar:
            for _, (low_imgs, _) in enumerate(train_bar):
                # low_imgs shape: (B, 2, C, H, W)
                low_imgs = low_imgs.to(device)
                input_imgs = low_imgs[:, 0, ...]   # First low image (B, C, H, W)
                target_imgs = low_imgs[:, 1, ...]  # Second low image (B, C, H, W)
                
                outputs = model(input_imgs)  # Model output expected shape: (B, C, H, W)
                loss = criterion(outputs, target_imgs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                iteration_count += 1
                train_bar.set_postfix(loss=loss.item())
                
                if iteration_count % interval == 0:
                    avg_iter_loss = running_loss / iteration_count
                    iteration_loss[iteration_count] = avg_iter_loss

        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        if epoch % val_frequency == 0:
            avg_val_loss = validate_unsupervised(model, val_loader, criterion, device)
            print(f"Epoch: {epoch}, avg_val_loss = {avg_val_loss:.8f}")
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_loss - 1e-3:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            
    model.load_state_dict(best_model_state)
    
    losses = {
        'train': train_losses,
        'val': val_losses,
        'iteration': iteration_loss,
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }
    
    print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss}")
    return model, losses