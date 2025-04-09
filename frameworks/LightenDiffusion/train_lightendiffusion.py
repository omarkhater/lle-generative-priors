import torch
from tqdm import tqdm
import copy
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Any
from torch.utils.data import DataLoader
from .losses import ctdn_loss_wrapper, stage2_loss_wrapper

def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    """
    Generic validation loop.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for low_imgs, high_imgs in val_loader:
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)
            # For Stage2 training, use two branches:
            # imgs_R = low_imgs[:, 0:1, ...] and imgs_L = low_imgs[:, 1:2, ...]
            out = model(low_imgs[:, 0:1, ...], low_imgs[:, 1:2, ...])
            stage2_out = out["stage2"]
            # Reconstruction: conditioning input - estimated noise.
            recon = stage2_out["x_condition"] - stage2_out["noise_est"]
            loss = criterion(recon, high_imgs)
            running_loss += loss.item()
    return running_loss / len(val_loader)


class BaseTrainer:
    """
    Base trainer that implements common training functionality.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        val_frequency: int = 5,
        patience: int = 5,
        log_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.val_frequency = val_frequency
        self.patience = patience
        self.log_interval = log_interval

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for i, (low_imgs, high_imgs) in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            low_imgs = low_imgs.to(self.device)
            high_imgs = high_imgs.to(self.device)
            # For Stage2 training the full pipeline expects two branches.
            out = self.model(low_imgs[:, 0:1, ...], low_imgs[:, 1:2, ...])
            stage2_out = out["stage2"]
            loss = self.criterion(stage2_out, high_imgs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                avg = running_loss / (i + 1)
                tqdm.write(f"  Batch {i+1}/{len(self.train_loader)}, loss={avg:.4f}")
        return running_loss / len(self.train_loader)

    def train(self) -> Tuple[nn.Module, Dict[str, Any]]:
        num_bad = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            if epoch % self.val_frequency == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
                if val_loss < self.best_loss - 1e-4:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    num_bad = 0
                else:
                    num_bad += 1
                    if num_bad >= self.patience:
                        print("Early stopping")
                        break
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
        self.model.load_state_dict(self.best_state)
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch + 1
        }
        return self.model, metrics
    
class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage1 (Encoder + Retinex Decomposition + Decoder).
    
    For Stage1 training, each training sample must contain paired low images with shape [B, 2, 3, H, W]:
      - low_imgs[:,0] is used as input.
      - low_imgs[:,1] is the target (raw/normal-light) image.
    Stage1 returns a list of dictionaries; we extract the first dictionary and use its (R, L)
    outputs with ctdn_loss_wrapper to compute the loss.
    """
    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for i, (low_imgs, _) in enumerate(tqdm(self.train_loader, desc="Training Stage1", leave=False)):
            low_imgs = low_imgs.to(self.device)
            outputs_list = self.model(low_imgs)  # Stage1 outputs list
            outputs = outputs_list[0]  # select the first processed image's outputs
            # Use the (R, L) tuple for loss computation using ctdn_loss_wrapper
            loss = ctdn_loss_wrapper((outputs["R"], outputs["L"]), low_imgs[:, 1, ...])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                avg = running_loss / (i + 1)
                tqdm.write(f"  Batch {i+1}/{len(self.train_loader)}, loss={avg:.4f}")
        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for low_imgs, _ in self.val_loader:
                low_imgs = low_imgs.to(self.device)
                outputs_list = self.model(low_imgs)
                outputs = outputs_list[0]
                loss = ctdn_loss_wrapper((outputs["R"], outputs["L"]), low_imgs[:, 1, ...])
                running_loss += loss.item()
        return running_loss / len(self.val_loader)

class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage2 (Diffusion model).
    
    Each training sample provides:
      - low_imgs: tensor of shape [B, m, 3, H, W] (with m ≥ 2), where low_imgs[:,0:1,...] is used for the R branch
        and low_imgs[:,1:2,...] is used for the L branch.
      - high_imgs: tensor of shape [B, 3, H, W] (target high-quality image).
    The full pipeline’s forward takes two branches and Stage2 returns a dictionary with keys:
         "noise_est" and "x_condition".
    The loss is computed using stage2_loss_wrapper.
    """
    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for i, (low_imgs, high_imgs) in enumerate(tqdm(self.train_loader, desc="Training Stage2", leave=False)):
            low_imgs = low_imgs.to(self.device)
            high_imgs = high_imgs.to(self.device)
            # Extract branches for diffusion: first image for R, second for L.
            low_imgs_R = low_imgs[:, 0:1, ...]
            low_imgs_L = low_imgs[:, 1:2, ...]
            outputs = self.model(low_imgs_R, low_imgs_L)
            stage2_out = outputs["stage2"]
            loss = stage2_loss_wrapper(stage2_out, high_imgs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                avg = running_loss / (i + 1)
                tqdm.write(f"  Batch {i+1}/{len(self.train_loader)}, loss={avg:.4f}")
        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for low_imgs, high_imgs in self.val_loader:
                low_imgs = low_imgs.to(self.device)
                high_imgs = high_imgs.to(self.device)
                low_imgs_R = low_imgs[:, 0:1, ...]
                low_imgs_L = low_imgs[:, 1:2, ...]
                outputs = self.model(low_imgs_R, low_imgs_L)
                stage2_out = outputs["stage2"]
                loss = stage2_loss_wrapper(stage2_out, high_imgs)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)