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
import os
from frameworks.LightenDiffusion.models.stage1 import Stage1


def with_curriculum(fn):
    """
        Decorator to apply curriculum learning before each training epoch.
    """
    def wrapper(self, epoch_number: int = None) -> dict:
        """
        Apply curriculum learning before each training epoch.
        Args:
            epoch_number (int): Current epoch number.
        Returns:
            dict: Dictionary containing average total loss, weighted content loss, and ctdn loss.
        """
        self._apply_curriculum(epoch_number)
        return fn(self, epoch_number)
    return wrapper

class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage1 (Encoder + Retinex Decomposition + Decoder).
    
    For Stage1 training, each training sample must contain paired low images with shape [B, m, 3, H, W].
    The model's forward method is expected to return a list of dictionaries, where each dictionary contains keys 
    such as "R", "L", "recon", and "f". The loss is computed by combining a ctdn_loss with a weighted content loss.
    """
    def __init__(
        self,
        model: Stage1,
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
        pretrain_content_ratio: float = .05, 
        pretrain_ctdn_ratio: float = .1,
        num_visualizations: int = 1,
        random_seed: int = 42,
        after_validate: bool = True,
        debug_gradients: bool = False,
        debug_gradients_every: int = 5,
        show_plot: bool = True,
        save_dir: str = None,

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
            pretrain_content_ratio: Fraction of total epochs to train with content-only.
            pretrain_ctdn_ratio: Fraction of total epochs to train with CTDN-only.
            num_visualizations (int): Number of samples to visualize during validation.
            random_seed (int): Seed for random selection of samples.
            after_validate (bool): Whether to run after-validation
            debug_gradients (bool): Whether to debug gradients.
            debug_gradients_every (int): Frequency of debugging gradients.
            show_plot (bool): Whether to show the plots when validating the model
            save_dir (str): Directory to save the model checkpoints. If None, no visuals are saved.

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
        self.debug_gradients = debug_gradients
        self.pretrain_content_ratio = pretrain_content_ratio
        self.pretrain_ctdn_ratio = pretrain_ctdn_ratio
        if self.debug_gradients:
            self.debug_gradients_every = debug_gradients_every
        else:
            self.debug_gradients_every = None
        self.show_plot = show_plot
        self.save_dir = save_dir

        self._final_weights = {
            'cont': weight_cont,
            'rec': weight_rec,
            'ref': weight_ref,
            'ill': weight_ill
        }

        logging.getLogger().setLevel(logging.DEBUG if self.debug_gradients else logging.INFO)
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


    @with_curriculum
    def train_epoch(self, epoch_number: int = None) -> dict:
        """
        Train for one epoch in unsupervised Stage1:
          1) Forward pass: model(low_imgs) => list of dictionaries [ {R, L}, {R, L}, ... ]
          2) Stack reflectances/illuminations => [B, m, C, H, W]
          3) ctdn_loss(...) => cross reconstruction + reflectance consistency + illumination smoothness

        Phased curriculum adjustments applied. 

        Args:
            None
        Returns:
            dict: Dictionary containing average total loss, weighted content loss, and ctdn loss.
        """
        self.model.train()
        running_loss = running_ctdn_loss = running_con_loss = 0.0

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

            if self.debug_gradients and self.debug_gradients_every is not None and i == 0:
                if epoch_number % self.debug_gradients_every == 0:
                    logging.debug(f"[🔍] Epoch {epoch_number}: Batch 0 - Debugging gradients")
                    self._debug_gradients() 
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

    def _apply_curriculum(self, epoch_number: int) -> None:
        """
        Adjust loss weights & parameter freezing based on epoch fraction. 
        
        Args:
            epoch_number (int): Current epoch number.

        """

        content_threshold = int(self.num_epochs * self.pretrain_content_ratio)
        ctdn_threshold = int(self.num_epochs * self.pretrain_ctdn_ratio)
        if epoch_number < content_threshold:
            logging.info(f"[📚] Curriculum: Training with content loss only until epoch {content_threshold}")
            self.weight_rec = self.weight_ref = self.weight_ill = 0.0
            for p in self.model.encoder.parameters(): p.requires_grad = True
            for p in self.model.decoder.parameters(): p.requires_grad = True
        
        elif epoch_number < ctdn_threshold:
            logging.info(f"[📚] Curriculum: Training with CTDN loss only until epoch {ctdn_threshold}")
   
            self.weight_rec = self._final_weights['rec']
            self.weight_ref = self._final_weights['ref']
            self.weight_ill = self._final_weights['ill']
            for p in self.model.encoder.parameters(): p.requires_grad = False
            for p in self.model.decoder.parameters(): p.requires_grad = False
        else:
            logging.info(f"[📚] Curriculum: Training with full loss after epoch {ctdn_threshold}")
            for p in self.model.parameters(): p.requires_grad = True
            self.weight_cont = self._final_weights['cont']
            self.weight_rec  = self._final_weights['rec']
            self.weight_ref  = self._final_weights['ref']
            self.weight_ill  = self._final_weights['ill']

    def _debug_gradients(
            self
        ) -> None:
        """
        Debug gradients for the model parameters.

        Args:
            batch_number (int): Current batch number.
            epoch_number (int): Current epoch number.
            log_freq (int): Frequency of logging gradients.
        """
        components = (
            ("encoder", self.model.encoder),
            ("decoder", self.model.decoder),
            ("decomposer", self.model.decomposer),
        )
            
        for part_name, module in components:
            for pname, p in module.named_parameters():
                if p.grad is not None:
                    logging.debug(f"[✅] {part_name} has grad ||.|| = {p.grad.norm()}")
                else:
                    logging.debug(f"[⚠️] {part_name} has no grad")


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
            if self.num_visualizations > 0:
                if self.save_dir:
                    save_dir = f"{self.save_dir}/epoch_{epoch}" 
                    os.makedirs(save_dir, exist_ok=True)
                else:
                    save_dir = None

                visualize_stage1_results(
                    self.model, 
                    self.val_loader, 
                    num_samples=self.num_visualizations,
                    seed=self.random_seed,
                    save_dir=save_dir,
                    show_plot=self.show_plot
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