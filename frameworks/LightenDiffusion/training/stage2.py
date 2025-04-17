from .base import BaseTrainer
import torch
from .tqdm_configuration import TqdmManager
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from .losses import noise_loss, self_constrained_consistency_loss
from frameworks.LightenDiffusion.visualization.visualize_stage2 import visualize_stage2_results
import logging
from evaluation.lighten_diffusion_stage2 import evaluate_stage2_metrics_avgfirst
from IPython.display import display
class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage2 (Diffusion model) that implements the loss as described in the paper.

    Expected inputs:
      - low_imgs: tensor of shape [B, m, 3, H, W] (with m ≥ 2). 
      - high_imgs: tensor of shape [B, 3, H, W] representing the target high-quality image.

    Note: There is no assumpution about pairing between low and high images.
      
    The pipeline's forward (i.e. LightenDiffusionPipeline) is expected to return a dictionary containing:
      - "stage2": a dict with keys:
             "noise": ground truth noise,
             "noise_pred": predicted noise from the DiffusionUNet.
      - "stage1_low": a dict with keys:
             "f": the conditioning feature,
             "R": reflectance,
             "L": illumination.
             
    Losses are computed as:
      - Diffusion loss: L1 between predicted noise and the ground truth noise.
      - SCC loss: L1 between the restored feature (obtained via reverse diffusion sampling on "f")
                 and the reference feature (computed as R * (L)**gamma).
                 
    Total loss is computed as:
         total_loss = diffusion_loss + lambda_scc * scc_loss.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 num_epochs: int = 100,
                 val_frequency: int = 5,
                 patience: int = 5,
                 lambda_scc: float = 0.001,
                 betas: torch.Tensor = None,
                 num_diffusion_timesteps: int = 1000,
                 num_sampling_timesteps: int = 50,
                 gamma: float = 0.2,
                 num_visualization: int = 1,
                 random_seed: int = 42,

                 ) -> None:
        super().__init__(model, train_loader, val_loader, optimizer, device,
                         scheduler, num_epochs, val_frequency, patience)
        self.lambda_scc = lambda_scc
        self.betas = betas
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_sampling_timesteps = num_sampling_timesteps
        self.gamma = gamma
        self.num_visualization = num_visualization
        self.random_seed = random_seed
        self._validate_dimensions(train_loader, "train_loader")
        self._validate_dimensions(val_loader, "val_loader")
        self._validate_model_format()

    def _validate_dimensions(self, loader: DataLoader, loader_name: str) -> None:
        """
        Validates that the data from the DataLoader has the expected dimensions.
        
        Expects:
            - low_imgs: a 5-D tensor with shape [B, m, 3, H, W].
            - high_imgs: a 4-D tensor with shape [B, 3, H, W].
        
        Args:
            loader (DataLoader): The DataLoader to validate.
            loader_name (str): Name of the DataLoader (used in error messages).
            
        Raises:
            TypeError or ValueError if the dimensions or types do not match expectations.
        """
        try:
            sample = next(iter(loader))
        except StopIteration:
            logging.warning(f"{loader_name} is empty; skipping dimension validation.")
            return

        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError(f"Expected {loader_name} to yield a tuple (low_imgs, high_imgs), but got {sample}")

        low_imgs, high_imgs = sample[:2]

        if not isinstance(low_imgs, torch.Tensor):
            raise TypeError(f"Expected low_imgs from {loader_name} to be a torch.Tensor, got {type(low_imgs)}")
        if len(low_imgs.shape) != 5:
            raise ValueError(f"Expected low_imgs from {loader_name} to be a 5-D tensor with shape [B, m, 3, H, W], but got shape {low_imgs.shape}")
        if low_imgs.shape[2] != 3:
            raise ValueError(f"Expected channel dimension (index 2) of low_imgs from {loader_name} to be 3, but got {low_imgs.shape[2]}")

        if not isinstance(high_imgs, torch.Tensor):
            raise TypeError(f"Expected high_imgs from {loader_name} to be a torch.Tensor, got {type(high_imgs)}")
        if len(high_imgs.shape) != 4:
            raise ValueError(f"Expected high_imgs from {loader_name} to be a 4-D tensor with shape [B, 3, H, W], but got shape {high_imgs.shape}")
        if high_imgs.shape[1] != 3:
            raise ValueError(f"Expected channel dimension (index 1) of high_imgs from {loader_name} to be 3, but got {high_imgs.shape[1]}")

        logging.info(f"{loader_name} dimension check passed: low_imgs shape {low_imgs.shape}, high_imgs shape {high_imgs.shape}")

    def _get_random_input(self) -> torch.Tensor:
        """
        Generates a random input tensor for the model.

        Returns:
            torch.Tensor: A random tensor of shape [1, m, 3, H, W].

        """
        dummy_low = torch.randn(1, 2, 3, 64, 64, device=self.device)  # Example: Batch=1, m=2 images
        dummy_high = torch.randn(1, 3, 64, 64, device=self.device)
        return (dummy_low, dummy_high)

    def _validate_model_format(self) -> None:
        """
        Validates that the Stage2 model's forward pass returns a dictionary with the expected format.
        
        Expected structure:
            - The output must be a dictionary containing keys "stage2" and "stage1_low".
            - The "stage2" dictionary must include:
                "noise": The ground truth noise tensor.
                "noise_pred": The predicted noise tensor.
            - The "stage1_low" dictionary must include:
                "f": The conditioning feature tensor.
                "R": The reflectance tensor.
                "L": The illumination tensor.
                
        Raises:
            ValueError: If the output does not conform to the expected format.
        """
        dummy_low, dummy_high = self._get_random_input()   
        outputs = self.model(dummy_low, dummy_high)
        if not isinstance(outputs, dict):
            raise ValueError("Stage2 model forward pass must return a dictionary.")
        required_keys = {"stage2", "stage1_low"}
        missing = required_keys - set(outputs.keys())
        if missing:
            raise ValueError(f"Stage2 forward pass output is missing keys: {missing}")
        
        stage2_dict = outputs["stage2"]
        required_stage2_keys = {"noise", "noise_pred"}
        missing_stage2 = required_stage2_keys - set(stage2_dict.keys())
        if missing_stage2:
            raise ValueError(f"Stage2 sub-dictionary is missing keys: {missing_stage2}")
        
        stage1_low_dict = outputs["stage1_low"]
        required_stage1_low_keys = {"f", "R", "L"}
        missing_stage1_low = required_stage1_low_keys - set(stage1_low_dict.keys())
        if missing_stage1_low:
            raise ValueError(f"Stage1_low sub-dictionary is missing keys: {missing_stage1_low}")

    def calculate_loss(
        self,
        restored_features,
        reference_feature: torch.Tensor,
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the total loss for Stage2 training.

        Args:
            restored_features (torch.Tensor): The restored features from the model.
            reference_feature (torch.Tensor): The reference feature (R * (L^gamma)).
            pred_noise (torch.Tensor): The predicted noise from the model.
            true_noise (torch.Tensor): The ground truth noise.

        Returns:
            tuple: (total_loss, diffusion_loss, scc_loss)
        """
        scc_loss = self_constrained_consistency_loss(
            restored_features, 
            reference_feature
        )
        diffusion_loss = noise_loss(pred_noise, true_noise) 
        total_loss =  diffusion_loss + self.lambda_scc * scc_loss
        return total_loss, diffusion_loss, scc_loss

    def train_epoch(self) -> float:
        """
        Train for one epoch over the training data.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.train()
        running_loss = 0.0
        running_scc_loss = 0.0
        running_diffusion_loss = 0.0
        batch_bar = TqdmManager(
            total=len(self.train_loader), 
            desc="Training Batches", 
            leave=True,
            unit = "batch",
        )
        for i, batch_data in enumerate(self.train_loader):
            batch_data = self.ensure_on_device(batch_data)
            low_imgs, high_imgs = batch_data
            loss_tesnors = self._gather_tensors(low_imgs, high_imgs)
            total_loss, diffusion_loss, scc_loss = self.calculate_loss(*loss_tesnors)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            running_loss += total_loss.item()
            running_scc_loss += scc_loss.item()
            running_diffusion_loss += diffusion_loss.item()
            avg_total_loss = running_loss / (i + 1)
            avg_scc_loss = running_scc_loss / (i + 1)
            avg_diffusion_loss = running_diffusion_loss / (i + 1)
            avg_weighted_scc_loss = self.lambda_scc * avg_scc_loss
            batch_bar.set_postfix(
                total_loss=f"{avg_total_loss:.4f}",
                diffusion_loss=f"{avg_diffusion_loss:.4f}",
                weighted_scc_loss=f"{avg_weighted_scc_loss:.4f}"
            )
            batch_bar.update(1)
        batch_bar.close()
        return {
            "total_loss": avg_total_loss,
            "diffusion_loss": avg_diffusion_loss,
            "scc_loss": avg_scc_loss
        }

    def validate_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Process and compute loss for a single validation batch.

        Args:
            batch (tuple): A tuple containing (low_imgs, high_imgs).

        Returns:
            float: The computed loss for the validation batch.
        """
        low_imgs, high_imgs = batch
        loss_tensors = self._gather_tensors(low_imgs, high_imgs)
        total_loss, _, _ = self.calculate_loss(*loss_tensors)
        return total_loss

    def after_validation(self):

        """
        Hook method to execute after the validation loop.
        This implementation visualizes Stage2 results.
        """
        
        val_metrics = evaluate_stage2_metrics_avgfirst(self.model, self.val_loader)
        display(val_metrics)
        print("Visualizing Stage2 results")
        visualize_stage2_results(
            self.model, 
            self.val_loader, 
            num_samples=self.num_visualization, 
            random_seed=self.random_seed
        )
    

    def _gather_tensors(
            self, 
            low_imgs: torch.Tensor, 
            high_imgs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Gather tensors from the model outputs for Stage2. To be used in calculate_loss.

        Args:

            low_imgs (torch.Tensor): Low-quality images of shape [B, m, 3, H, W].
            high_imgs (torch.Tensor): High-quality images of shape [B, 3, H, W].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - restored_features: Restored features from the model.
                - reference_feature: Reference feature (R * (L^gamma)).
                - pred_noise: Predicted noise from the model.
                - true_noise: Ground truth noise.

        """
        outputs = self.model(low_imgs, high_imgs)
        stage2_out = outputs["stage2"]
        true_noise = stage2_out["noise"]      
        pred_noise = stage2_out["noise_pred"]
        aggregated_low = outputs["stage1_low"]
        low_condition = aggregated_low["f"]
        restored_features = self.model.sample_reverse(low_condition)
        reference_feature = aggregated_low["R"] * torch.pow(aggregated_low["L"], self.gamma)

        return (
            restored_features, 
            reference_feature, 
            pred_noise, 
            true_noise
        )