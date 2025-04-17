from .base import BaseTrainer
import torch
import copy
import torch.nn as nn
from typing import List, Optional
from torch.utils.data import DataLoader
from .losses import ctdn_loss, content_loss
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
import logging
from .tqdm_configuration import TqdmManager
from IPython.display import display
import os

class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage1 (Encoder + Retinex Decomposition + Decoder).
    
    For Stage1 training, each training sample must contain paired low images with shape [B, m, 3, H, W].
    The model's forward method is expected to return a list of dictionaries, where each dictionary contains keys 
    such as "R", "L", "recon", and "f". The loss is computed by combining a ctdn_loss with a weighted content loss.
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
        weight_cont: float = 0.1,
        weight_rec: float = 1.0,
        weight_ref: float = 0.1,
        weight_ill: float = 0.1,
        lambda_g: float = 0.2,
        num_visualizations: int = 1,
        random_seed: int = 42,
        after_validate: bool = True,
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
            weight_rec (float): Weight for reconstruction term in ctdn_loss.
            weight_cont (float): Weight for content loss.
            weight_ref (float): Weight for reflectance-consistency term in ctdn_loss.
            weight_ill (float): Weight for illumination-smoothness term in ctdn_loss.
            lambda_g (float): Exponential weighting factor for gradient in ctdn_loss.
            num_visualizations (int): Number of samples to visualize during validation.
            random_seed (int): Seed for random selection of samples.
            after_validate (bool): Whether to run after-validation
        """
        super().__init__(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            device,
            scheduler, 
            num_epochs, 
            val_frequency, 
            patience, 
        )
        self.weight_rec = weight_rec
        self.weight_ref = weight_ref
        self.weight_ill = weight_ill
        self.weight_cont = weight_cont
        self.lambda_g = lambda_g
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.num_visualizations = num_visualizations
        self.random_seed = random_seed
        self.after_validate = after_validate
        self.all_val_metrics = []
        self._validate_dimensions(train_loader, "train_loader")
        self._validate_dimensions(val_loader, "val_loader")
        self._validate_model_format()


    def _validate_dimensions(self, loader: DataLoader, loader_name: str) -> None:
        """
        Validates that the data coming from the loader has the expected dimensions.
        Expected shape for low images: [B, m, 3, H, W].
        
        Args:
            loader (DataLoader): The data loader to test.
            loader_name (str): Name of the loader (for error messages).
            
        Raises:
            ValueError: If the data does not have the expected shape.
        """
        try:
            sample = next(iter(loader))
        except StopIteration:
            logging.warning(f"{loader_name} is empty; skipping dimension validation.")
            return

        low_imgs = sample[0]
        if not isinstance(low_imgs, torch.Tensor):
            raise TypeError(f"Expected low_imgs in {loader_name} to be a torch.Tensor, but got {type(low_imgs)}")
        if len(low_imgs.shape) != 5:
            raise ValueError(
                f"Expected low_imgs from {loader_name} to be a 5-D tensor with shape [B, m, 3, H, W], "
                f"but got shape {low_imgs.shape}"
            )
        if low_imgs.shape[2] != 3:
            raise ValueError(
                f"Expected the channel dimension (index 2) to be 3 in {loader_name}, but got {low_imgs.shape[2]}"
            )
        logging.info(f"{loader_name} dimension check passed: got low_imgs with shape {low_imgs.shape}")

    def calculate_loss(
            self, 
            low_imgs: torch.Tensor,
            reflectances: torch.Tensor,
            illuminations: torch.Tensor,
            reconstructions: torch.Tensor,
            encoded_features: torch.Tensor
        ) -> tuple:
        """
        Calculate the loss for Stage1:
        1) ctdn_loss: cross reconstruction + reflectance consistency + illumination smoothness
        2) content_loss: perceptual loss between reconstructions and low_imgs

        Args:
            low_imgs (torch.Tensor): Low-light images of shape [B, m, 3, H, W].
            reflectances (torch.Tensor): Reflectance outputs of shape [B, m, C, H/2^k, W/2^k].
            illuminations (torch.Tensor): Illumination outputs of shape [B, m, C, H/2^k, W/2^k].
            reconstructions (torch.Tensor): Decoder reconstructions of shape [B, m, 3, H/2^k, 2^k].
            encoded_features (torch.Tensor): Encoded features of shape [B, m, C, H/2^k, W/2^k].
        Returns:
            tuple: Total loss, ctdn_loss, and content_loss.
        """
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

        return loss_total, loss_ctdn, loss_con

    def train_epoch(self) -> dict:
        """
        Train for one epoch in unsupervised Stage1:
          1) Forward pass: model(low_imgs) => list of dictionaries [ {R, L}, {R, L}, ... ]
          2) Stack reflectances/illuminations => [B, m, C, H, W]
          3) ctdn_loss(...) => cross reconstruction + reflectance consistency + illumination smoothness

        Args:
            None
        Returns:
            dict: Dictionary containing average total loss, weighted content loss, and ctdn loss.
        """
        self.model.train()
        running_loss = 0.0
        running_ctdn_loss = 0.0
        running_con_loss = 0.0

        batch_bar = TqdmManager(
            total=len(self.train_loader), 
            desc="Training Batches", 
            leave=True, 
            unit = "batch",
        )
        for i, (low_imgs, _) in enumerate(self.train_loader):
            low_imgs = low_imgs.to(self.device)  
            outputs_list = self.model(low_imgs)
            loss_tesnors = self._gather_tensors(low_imgs, outputs_list)
            loss_total, loss_ctdn, loss_con = self.calculate_loss(*loss_tesnors)
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            running_loss += loss_total.item()
            running_ctdn_loss += loss_ctdn.item()
            running_con_loss += loss_con.item()

            avg_total_loss = running_loss / (i + 1)
            avg_ctdn_loss = running_ctdn_loss / (i + 1)
            avg_con_loss = running_con_loss / (i + 1)
            weighted_loss_content = self.weight_cont * avg_con_loss



            batch_bar.set_postfix(
                    total_loss=f"{avg_total_loss:.4f}",
                    weighted_content_loss=f"{weighted_loss_content:.4f}", 
                    ctdn_loss=f"{avg_ctdn_loss:.4f}"
                )  
            batch_bar.update(1)
    
        batch_bar.close()

        return {
            'total_loss': avg_total_loss,
            'weighted_content_loss': weighted_loss_content,
            'ctdn_loss': avg_ctdn_loss,
        }

    def validate_batch(self, batch: tuple) -> float:
        """
        Process a validation batch for Stage1.
        
        Args:
            batch (tuple): A batch from the validation loader containing 
                        (low_imgs, _) where _ is ignored.
        
        Returns:
            float: The computed loss for this batch.
        """
        low_imgs, _ = batch
        low_imgs = low_imgs.to(self.device)
        outputs_list = self.model(low_imgs)
        loss_tensors = self._gather_tensors(low_imgs, outputs_list)
        loss_total, _, _ = self.calculate_loss(*loss_tensors)
        return loss_total

    
    def after_validation(self, epoch: int):
        if self.after_validate:
            val_metrics = evaluate_stage1_metrics_individual(self.model, self.val_loader)
            self.all_val_metrics.append(val_metrics)
            # Do not display interactively.
            # Instead, generate validation plots and save them for artifact logging.
            if self.num_visualizations > 0:
                save_dir = f"outputs/visualizations/val_stage1/epoch_{epoch}"
                os.makedirs(save_dir, exist_ok=True)
                visualize_stage1_results(
                    self.model, 
                    self.val_loader, 
                    num_samples=self.num_visualizations,
                    seed=self.random_seed,
                    save_dir=save_dir,
                    show_plot=False
                )
        return
    
    def _gather_tensors(self, low_imgs, outputs_list):
        """
        Gather tensors from the model outputs for Stage1 training.
        Args:
            low_imgs (torch.Tensor): Low-light images of shape [B, m, 3, H, W].
            outputs_list (list): List of dictionaries containing model outputs.
        Returns:
            tuple: Stacked tensors for reflectances, illuminations, reconstructions, and encoded features.

        """
        reflectances = []
        illuminations = []
        decoder_recons = []
        encoded_features = []
        
        for j in range(low_imgs.shape[1]):
            reflectances.append(outputs_list[j]["R"])
            illuminations.append(outputs_list[j]["L"])
            decoder_recons.append(outputs_list[j]["recon"])
            encoded_features.append(outputs_list[j]["f"])
        
        reflectances = torch.stack(reflectances, dim=1)
        illuminations = torch.stack(illuminations, dim=1)
        reconstructions = torch.stack(decoder_recons, dim=1)
        encoded_features = torch.stack(encoded_features, dim=1)

        return (
            low_imgs,
            reflectances, 
            illuminations,
            reconstructions, 
            encoded_features
        )
    
    def _validate_model_format(self) -> None:
        """
        Validates that the Stage1 model's forward pass returns a list of dictionaries with the expected keys.
        
        Expected keys in each output dictionary:
            - "f": Compressed latent features from the encoder.
            - "R": Decomposed reflectance.
            - "L": Decomposed illumination.
            - "recon": Reconstructed image.
            - "features": Tuple of multi-scale skip features.
            
        Raises:
            ValueError: If the forward pass output is not a list or if any dictionary is missing the required keys.
        """
        dummy_input = self._get_random_input()
        outputs = self.model(dummy_input)
        if not isinstance(outputs, list):
            raise ValueError("Stage1 model forward pass must return a list of dictionaries.")
        expected_keys = {"f", "R", "L", "recon", "features"}
        for idx, output in enumerate(outputs):
            if not isinstance(output, dict):
                raise ValueError(f"Stage1 model output at index {idx} is not a dictionary.")
            missing = expected_keys - set(output.keys())
            if missing:
                raise ValueError(f"Stage1 output dictionary at index {idx} is missing keys: {missing}")
    
    def _get_random_input(self) -> torch.Tensor:
        """
        Generates a random input tensor for the model.

        Returns:
            torch.Tensor: A random tensor of shape [1, m, 3, H, W].
        """
        return torch.randn(1, 2, 3, 64, 64, device=self.device)