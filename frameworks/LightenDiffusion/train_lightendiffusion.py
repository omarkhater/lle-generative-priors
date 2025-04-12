import torch
from tqdm import tqdm
import copy
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Any
from torch.utils.data import DataLoader
from .losses import (
    ctdn_loss, 
    content_loss,
    stage2_loss_wrapper, 
    noise_loss, 
    self_constrained_consistency_loss
)
import traceback
import torch.nn.functional as F
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results_individual, visualize_stage1_results_avg

class BaseTrainer:
    """
    Base trainer that implements common training functionality.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: callable,
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
            out = self.model(low_imgs[:, 0:1, ...], low_imgs[:, 1:2, ...])
            stage2_out = out["stage2"]
            target_noise = stage2_out["x_condition"] - high_imgs
            diffusion_loss = noise_loss(stage2_out["noise_est"], target_noise)
            scc_loss = self_constrained_consistency_loss(stage2_out["f_low"], stage2_out["f_low_hat"])
            loss = stage2_loss_wrapper(diffusion_loss, scc_loss, self.criterion.lambda_scc)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                avg = running_loss / (i + 1)
                tqdm.write(f"  Batch {i+1}/{len(self.train_loader)}, loss={avg:.4f}")
        return running_loss / len(self.train_loader)

    def train(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Main training loop with optional validation, early stopping, and LR scheduling.
        Returns:
            (model, metrics) where metrics includes 'train_losses', 'val_losses', etc.
        """
        try:
            num_bad = 0
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                if epoch > 1 and epoch % self.val_frequency == 0:
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
        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()
            return None, {}
        finally:
            self.cleanup_gpu()
        
    
    def cleanup_gpu(self):
        """
        Moves the model to CPU and clears GPU memory.
        """
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage1 (Encoder + Retinex Decomposition + Decoder).
    
    For Stage1 training, each training sample must contain paired low images with shape [B, m, 3, H, W]:
    Stage1 returns a list of dictionaries; we extract the first dictionary and use its (R, L)
    outputs with ctdn_loss_wrapper to compute the loss.
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
        log_interval: int = 100,
        weight_cont: float = 0.1,
        weight_rec: float = 1.0,
        weight_ref: float = 0.1,
        weight_ill: float = 0.1,
        lambda_g: float = 0.2
    ):
        """
        Args:
            model (nn.Module): The Stage1 model (Encoder + Retinex + Decoder).
            train_loader (DataLoader): Training DataLoader yielding (low_imgs, _).
            val_loader (DataLoader): Validation DataLoader yielding (low_imgs, _).
            optimizer (torch.optim.Optimizer): Optimizer for Stage1 parameters.
            device (torch.device): 'cpu' or 'cuda' device.
            scheduler (Optional): Learning rate scheduler.
            num_epochs (int): Max number of epochs to train.
            val_frequency (int): Run validation every N epochs.
            patience (int): Early-stopping patience.
            log_interval (int): Print batch loss every N iterations.
            weight_rec (float): Weight for reconstruction term in ctdn_loss.
            weight_cont (float): Weight for content loss.
            weight_ref (float): Weight for reflectance-consistency term in ctdn_loss.
            weight_ill (float): Weight for illumination-smoothness term in ctdn_loss.
            lambda_g (float): Exponential weighting factor for gradient in ctdn_loss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.val_frequency = val_frequency
        self.patience = patience
        self.log_interval = log_interval

        # Stage1 loss hyperparams
        self.weight_rec = weight_rec
        self.weight_ref = weight_ref
        self.weight_ill = weight_ill
        self.weight_cont = weight_cont
        self.lambda_g = lambda_g

        # Tracking best model
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())

        # Logs
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_epoch(self) -> float:
        """
        Train for one epoch in unsupervised Stage1:
          1) Forward pass: model(low_imgs) => list of dictionaries [ {R, L}, {R, L}, ... ]
          2) Stack reflectances/illuminations => [B, m, C, H, W]
          3) ctdn_loss(...) => cross reconstruction + reflectance consistency + illumination smoothness
        """
        self.model.train()
        running_loss = 0.0
        
        for i, (low_imgs, _) in enumerate(tqdm(self.train_loader, desc="Training Stage1", leave=False)):
            low_imgs = low_imgs.to(self.device)  # shape [B, m, 3, H, W]
            
            # 1) Forward pass => list of length m
            outputs_list = self.model(low_imgs)
            # Each element is a dict: {"R": R_ij, "L": L_ij, ...}

            # 2) Gather R and L into a single tensor
            reflectances = []
            illuminations = []
            decoder_recons = []
            encoded_features = []
            for j in range(low_imgs.shape[1]):
                reflectances.append(outputs_list[j]["R"])  # shape [B, C, H/8, W/8]
                illuminations.append(outputs_list[j]["L"]) # shape [B, C, H/8, W/8]
                decoder_recons.append(outputs_list[j]["recon"]) # shape [B, 3, H, W]
                encoded_features.append(outputs_list[j]["f"]) # shape [B, C, H/8, W/8]

            reflectances = torch.stack(reflectances, dim=1)     # shape [B, m, C, H/8, W/8]
            illuminations = torch.stack(illuminations, dim=1)    # shape [B, m, C, H/8, W/8]
            reconstructions = torch.stack(decoder_recons, dim=1) # shape [B, m, 3, H, W]
            encoded_features = torch.stack(encoded_features, dim=1) # shape [B, m, C, H/8, W/8]
            
            # For debugging
            if i == 0:
                print(f"reflectances: {reflectances.shape}, illuminations: {illuminations.shape}, "
                      f"reconstructions: {reconstructions.shape}, encoded_features: {encoded_features.shape}")
            
            loss_ctdn = ctdn_loss(
                reflectances, 
                illuminations,
                encoded_features,  
                weight_rec=self.weight_rec,
                weight_ref=self.weight_ref,
                weight_ill=self.weight_ill,
                lambda_g=self.lambda_g
            )
            loss_con = content_loss(
                reconstructions,
                low_imgs
            )
            loss_total = loss_ctdn + self.weight_cont * loss_con
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            running_loss += loss_total.item()
            if (i + 1) % self.log_interval == 0:
                avg_loss = running_loss / (i + 1)
                avg_content_loss = loss_con.item()
                avg_ctdn_loss = loss_ctdn.item()
                tqdm.write(f"""
Batch {i+1}/{len(self.train_loader)}: content loss = {avg_content_loss:.4f}, ctdn loss: {avg_ctdn_loss} , Total loss={avg_loss:.4f}""")

        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Validation for Stage1: do the same procedure with no grad,
        compute the average ctdn_loss over the val_loader.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for low_imgs, _ in self.val_loader:
                low_imgs = low_imgs.to(self.device)

                outputs_list = self.model(low_imgs)
                reflectances = []
                illuminations = []
                encoded_features = []
                for j in range(low_imgs.shape[1]):
                    reflectances.append(outputs_list[j]["R"])
                    illuminations.append(outputs_list[j]["L"])
                    encoded_features.append(outputs_list[j]["f"])
                reflectances = torch.stack(reflectances, dim=1)
                illuminations = torch.stack(illuminations, dim=1)
                encoded_features = torch.stack(encoded_features, dim=1)

                loss_total = ctdn_loss(
                    reflectances, 
                    illuminations,
                    encoded_features,
                    weight_rec=self.weight_rec,
                    weight_ref=self.weight_ref,
                    weight_ill=self.weight_ill,
                    lambda_g=self.lambda_g
                )
                running_loss += loss_total.item()

        avg_loss = running_loss / len(self.val_loader)

        print("Visualizing Stage1 results using individual reconstructions...")
        visualize_stage1_results_individual(self.model, self.val_loader, num_samples=2)
        print("Visualizing Stage1 results using average reconstruction...")
        visualize_stage1_results_avg(self.model, self.val_loader, num_samples=2)
        return avg_loss

class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage2 (Diffusion model) that implements the loss as described in the paper.
    
    Expected inputs:
      - low_imgs: tensor of shape [B, m, 3, H, W] (with m ≥ 2), where:
           • low_imgs[:, 0:1, ...] is used for the reflectance branch (R);
           • low_imgs[:, 1:2, ...] is used for the illumination branch (L).
      - high_imgs: tensor of shape [B, 3, H, W] representing the target high-quality image.
      
    The pipeline's forward (i.e. LightenDiffusionPipeline) is expected to return a dictionary containing:
      - "stage2": a dict with keys:
             "noise_est": predicted noise from DiffusionUNet,
             "x_condition": computed as R * L.
      - "stage1_R": a list of dictionaries from Stage1; the first entry is used to extract:
             "f": the conditioning feature,
             "R": reflectance,
             "L": illumination.
             
    Losses are computed as:
      - Diffusion loss: L1 between predicted noise and (x_condition - high_imgs).
      - SCC loss: L1 between the restored feature (restored_image_features) and the reference feature, where:
             restored_image_features is obtained via reverse diffusion sampling (using sample_reverse)
             on the conditioning feature ("f"),
             reference_fea = low_R * (low_L)**gamma.
             
    Total loss is:
         L_total = L_diff + lambda_scc * L_scc.
    
    The trainer requires the following additional parameters:
      - lambda_scc: weight for the SCC loss.
      - betas: diffusion schedule (a tensor of shape [T]).
      - num_diffusion_timesteps: total number of diffusion timesteps (T).
      - num_sampling_timesteps: number of timesteps used for reverse diffusion sampling.
      - gamma: exponent applied to low_L to obtain the reference feature.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: callable,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 num_epochs: int = 100,
                 val_frequency: int = 5,
                 patience: int = 5,
                 log_interval: int = 100,
                 lambda_scc: float = 0.001,
                 betas: torch.Tensor = None,
                 num_diffusion_timesteps: int = 1000,
                 num_sampling_timesteps: int = 50,
                 gamma: float = 0.2):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device,
                         scheduler, num_epochs, val_frequency, patience, log_interval)
        self.lambda_scc = lambda_scc
        self.betas = betas
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_sampling_timesteps = num_sampling_timesteps
        self.gamma = gamma

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for i, (low_imgs, high_imgs) in enumerate(tqdm(self.train_loader, desc="Training Stage2", leave=False)):
            low_imgs = low_imgs.to(self.device)
            high_imgs = high_imgs.to(self.device)
            low_imgs_R = low_imgs[:, 0:1, ...]
            low_imgs_L = low_imgs[:, 1:2, ...]
            out = self.model(low_imgs_R, low_imgs_L)
            stage2_out = out["stage2"]
            image_size = high_imgs.shape[-2:]
            x0_resized = F.interpolate(
                stage2_out["x0"], 
                size= image_size, 
                mode="bilinear", 
                align_corners=False
            )
            target_noise = x0_resized - high_imgs
            noise_pred_resized = F.interpolate(
                stage2_out["noise_pred"], 
                size=image_size, 
                mode="bilinear", 
                align_corners=False
            )
            diffusion_loss = noise_loss(noise_pred_resized, target_noise)
            stage1_out = out["stage1_low"][0]
            low_condition = stage1_out["f"]
            restored_image_features = self.model.sample_reverse(low_condition,)

            low_R = stage1_out["R"]
            low_L = stage1_out["L"]
            reference_fea = low_R * torch.pow(low_L, self.gamma)
            scc_loss = self_constrained_consistency_loss(restored_image_features, reference_fea)
            total_loss = self.criterion(diffusion_loss, scc_loss, self.lambda_scc)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            if (i + 1) % self.log_interval == 0:
                avg_loss = running_loss / (i + 1)
                tqdm.write(f"  Batch {i+1}/{len(self.train_loader)}, loss = {avg_loss:.4f}")
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
                out = self.model(low_imgs_R, low_imgs_L)
                stage2_out = out["stage2"]
                image_size = high_imgs.shape[-2:]
                x0_resized = F.interpolate(
                    stage2_out["x0"], 
                    size= image_size, 
                    mode="bilinear", 
                    align_corners=False
                )
                target_noise = x0_resized - high_imgs
                noise_pred_resized = F.interpolate(
                    stage2_out["noise_pred"], 
                    size=image_size, 
                    mode="bilinear", 
                    align_corners=False
                )
                target_noise = x0_resized- high_imgs
                diffusion_loss = noise_loss(noise_pred_resized, target_noise)
                stage1_out = out["stage1_low"][0]
                low_condition = stage1_out["f"]
                restored_image_features = self.model.sample_reverse(low_condition)
                low_R = stage1_out["R"]
                low_L = stage1_out["L"]
                reference_fea = low_R * torch.pow(low_L, self.gamma)
                scc_loss = self_constrained_consistency_loss(restored_image_features, reference_fea)
                total_loss = self.criterion(diffusion_loss, scc_loss, self.lambda_scc)
                running_loss += total_loss.item()
        return running_loss / len(self.val_loader)

