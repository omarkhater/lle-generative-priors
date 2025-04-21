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
import mlflow
import numpy as np

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
        self._apply_curriculum(self.current_epoch)
        return fn(self)
    return wrapper

class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage1 (Encoder + Retinex Decomposition + Decoder).
    
    For Stage1 training, each training sample must contain paired low images with shape [B, m, 3, H, W].
    The model's forward method is expected to return a list of dictionaries, where each dictionary contains keys 
    such as "R", "L", "recon", and "f". The loss is computed by combining a ctdn_loss with a weighted content loss.
    
    Class Attributes:
        higher_is_better (set): Evaluation Metrics that are better when higher. To be used in mlflow logging.
        lower_is_better (set): Evaluation Metrics that are better when lower. To be used in mlflow logging.
    """
    higher_is_better = {"psnr", "ssim"}
    lower_is_better = {"tv_illumination", "pi", "niqe", "lpips"}
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
        weight_rec: float = 1.0,
        weight_ref: float = 0.1,
        weight_ill: float = 0.1,
        lambda_g: float = 0.2,
        pretrain_content_ratio: float = .05, 
        pretrain_ctdn_ratio: float = .1,
        num_visualizations: int = 1,
        random_seed: int = 42,
        show_plot: bool = True,
        save_dir: str = None,
        gradnorm_alpha: float = 1.5,

    ) -> None:
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
            weight_ref (float): Weight for reflectance-consistency term in ctdn_loss.
            weight_ill (float): Weight for illumination-smoothness term in ctdn_loss.
            lambda_g (float): Exponential weighting factor for gradient in ctdn_loss.
            num_visualizations (int): Number of samples to visualize during validation.
            random_seed (int): Seed for random selection of samples.
            show_plot (bool): Whether to show the plots when validating the model
            save_dir (str, optional): Base Local Directory to save training artifcats (visuals+models) if provided.
            gradnorm_alpha (float): Exponent for GradNorm balancing.
            pretrain_content_ratio (float): Ratio of epochs to pretrain content loss.
            pretrain_ctdn_ratio (float): Ratio of epochs to pretrain ctdn loss.
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
        self.lambda_g = lambda_g
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.num_visualizations = num_visualizations
        self.random_seed = random_seed
        self.all_val_metrics = []
        self.show_plot = show_plot
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.num_epochs = num_epochs
        self.current_epoch = 0

        self.log_w_con = nn.Parameter(torch.log(torch.tensor(2.0, device=self.device)))
        self.log_w_ctdn  = nn.Parameter(torch.zeros(1, device=self.device))
        self.optimizer.add_param_group(
            {
                "params": [self.log_w_con, self.log_w_ctdn],
                "lr": self.optimizer.param_groups[0]["lr"] * .5
            }
        )
        self.L0_con = None
        self.L0_ctdn = None
        self.gradnorm_alpha = gradnorm_alpha
        self.pretrain_content_ratio = pretrain_content_ratio
        self.pretrain_ctdn_ratio = pretrain_ctdn_ratio
        self._validate_dimensions(train_loader, "train_loader")
        self._validate_dimensions(val_loader, "val_loader")
        self._validate_model_format()
    
    def _freeze_module(self, module: nn.Module, freeze: bool) -> None:
        for p in module.parameters():
            p.requires_grad = not freeze
        
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
        Calculate the total loss for Stage1 training.


        Args:
            low_imgs (torch.Tensor): Low-light images of shape [B, m, 3, H, W].
            reflectances (torch.Tensor): Reflectance outputs of shape [B, m, C, H/2^k, W/2^k].
            illuminations (torch.Tensor): Illumination outputs of shape [B, m, C, H/2^k, W/2^k].
            reconstructions (torch.Tensor): Decoder reconstructions of shape [B, m, 3, H/2^k, 2^k].
            encoded_features (torch.Tensor): Encoded features of shape [B, m, C, H/2^k, W/2^k].
        Returns:
            tuple: 
            loss_total, raw_ctdn_loss, raw_content_loss
            (scaling is applied inside `loss_total` via learned GradNorm weights)
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
        w_ctdn = torch.exp(self.log_w_ctdn)
        w_con  = torch.exp(self.log_w_con)

        loss_total = w_ctdn * loss_ctdn + w_con * loss_con

        return loss_total, loss_ctdn, loss_con


    @with_curriculum
    def train_epoch(self) -> dict:
        """
        Train for one epoch with integrated GradNorm balancing.

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
        self._last_raw_con = self._last_raw_ctdn = None
        self.current_epoch += 1
        batch_bar = TqdmManager(
            total=len(self.train_loader), 
            desc="Training Batches", 
            leave=True, 
            unit = "batch",
        )
        for i, (low_imgs, _) in enumerate(self.train_loader):
            low_imgs = low_imgs.to(self.device)  
            outputs_list = self.model(low_imgs)
            loss_tensors = self._gather_tensors(low_imgs, outputs_list)
            loss_total, raw_loss_ctdn, raw_loss_con = self.calculate_loss(*loss_tensors)
            self._last_raw_con  = raw_loss_con.item()
            self._last_raw_ctdn = raw_loss_ctdn.item()
            self.optimizer.zero_grad()
            loss_total.backward(retain_graph=True)

            # kill any grad that just landed on the weights
            for p in (self.log_w_con, self.log_w_ctdn):
                if p.grad is not None:
                    p.grad = None

            gradnorm_loss = self._gradnorm_step(raw_loss_ctdn, raw_loss_con)
            
            gradnorm_loss.backward()

            if mlflow.active_run():
                mlflow.log_metric("grad_w_con",  self.log_w_con.grad.abs().mean().item(), step=self.current_epoch)
                mlflow.log_metric("grad_w_ctdn", self.log_w_ctdn.grad.abs().mean().item(), step=self.current_epoch)

            self._log_gradient_stats()
            self.optimizer.step()

            running_loss += loss_total.item()
            running_ctdn_loss += raw_loss_ctdn.item()
            running_con_loss += raw_loss_con.item()

            avg_total_loss = running_loss / (i + 1)
            avg_ctdn_loss = running_ctdn_loss / (i + 1)
            avg_con_loss = running_con_loss / (i + 1)

            scaled_con  = torch.exp(self.log_w_con).item()  * avg_con_loss
            scaled_ctdn = torch.exp(self.log_w_ctdn).item() * avg_ctdn_loss

            batch_bar.set_postfix(
                total_loss       = f"{avg_total_loss:.4f}",
                scaled_con_loss  = f"{scaled_con:.4f}",
                scaled_ctdn_loss = f"{scaled_ctdn:.4f}"
            )

            batch_bar.update(1)
    
        batch_bar.close()

        return {
            'total_loss': avg_total_loss,
            'scaled_con_loss': scaled_con,
            'scaled_ctdn_loss': scaled_ctdn,
        }

    def _apply_curriculum(self, epoch: int) -> None:
        """
        *Phase 1*  (0 → t1): warm‑up Encoder+Decoder with a **linear ramp**
        *Phase 2*  (t1 → t2): train CTDN only, again linearly ramping its loss weights
        *Phase 3*  ( ≥ t2):   joint fine‑tune; progressively un‑freeze encoder blocks
        """
        # --- phase boundaries ----------------------------------------------------
        T   = self.num_epochs
        t1  = int(self.pretrain_content_ratio * T)                
        t2  = int((self.pretrain_content_ratio + self.pretrain_ctdn_ratio) * T)
        logging.info(f"Training Encoder + Decoder only up to epoch {t1}")
        logging.info(f"Training CTDN only up to epoch {t2}")
        logging.info(f"Fine-tuning Encoder + Decoder + CTDN after epoch {t2} - {self.num_epochs}")

        if epoch < t1:

            self._freeze_module(self.model.encoder,   False)
            self._freeze_module(self.model.decoder,   False)
            self._freeze_module(self.model.decomposer, True)

        # ----------------------- PHASE 2 : CTDN only -----------------------------
        elif epoch < t2:

            self._freeze_module(self.model.encoder,   True)
            self._freeze_module(self.model.decoder,   True)
            self._freeze_module(self.model.decomposer,False)

        # ----------------------- PHASE 3 : Joint fine‑tune -----------------------
        else:

            # progressive un‑freezing of encoder blocks
            unfreeze_gap = 3                              # epochs between unfreezes
            blocks       = list(self.model.encoder.children())
            n_unfreeze   = min(len(blocks),
                            1 + (epoch - t2) // unfreeze_gap)
            for i, block in enumerate(blocks):
                self._freeze_module(block, freeze=(i >= n_unfreeze))
            self._freeze_module(self.model.decoder,   False)
            self._freeze_module(self.model.decomposer,False)


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

    
    def after_training(self) -> None:
        """
        Save the best model and log metrics to mlflow.

        """
        train_metrics = evaluate_stage1_metrics_individual(
            self.model, 
            self.train_loader
        )
        self._log_evaluation_metrics(train_metrics, prefix="train")

        local_path = f"{self.save_dir}/model/best_model.pth" if self.save_dir else None
        self.save_pytorch_model(self.best_state, local_path)
        if mlflow.active_run():
            mlflow.log_artifact(local_path )
            mlflow.log_metric("best_val_loss", self.best_loss)
            mlflow.log_metric("best_val_epoch", self.best_epoch)
            mlflow.pytorch.log_model(self.model, "model")
        return 
    
    def after_validation(self, epoch: int) -> None:
        """
        Save the best model and log metrics to mlflow.

        Args:
            epoch (int): Current epoch number.

        """
        val_metrics = evaluate_stage1_metrics_individual(self.model, self.val_loader)
        self.all_val_metrics.append(val_metrics)
        self._log_evaluation_metrics(val_metrics, prefix="val")
        epoch_visuals_save_dir = f"{self.save_dir}/epoch_{epoch}" if self.save_dir else None
        visualize_stage1_results(
            self.model, 
            self.val_loader, 
            num_samples=self.num_visualizations,
            seed=self.random_seed,
            save_dir=epoch_visuals_save_dir,
            show_plot=self.show_plot
        )
        return
    
    def _gather_tensors(self, low_imgs, outputs_list) -> tuple:
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
    
    def _log_gradient_stats(self) -> None:
        """
        Collect and log gradient statistics + current GradNorm weights.

        For each sub‑module (encoder, decomposer, decoder) we record:
        • mean |grad|   • min |grad|   • max |grad|

        We also record the *learned* task weights and their product with the
        current batch’s raw losses (if they were stashed on the trainer as
        `self._last_raw_con` / `self._last_raw_ctdn` in train_epoch).

        MLflow step index = self.current_epoch.
        """

        stats = {}
        # ── per‑module grad norms ────────────────────────────────────────────────
        for name, module in (
            ("encoder",    self.model.encoder),
            ("decomposer", self.model.decomposer),
            ("decoder",    self.model.decoder),
        ):
            norms = [p.grad.norm().item() for p in module.parameters()
                    if p.grad is not None]
            stats[f"grad/{name}/mean"] = float(np.mean(norms)) if norms else 0.0
            stats[f"grad/{name}/min"]  = float(np.min(norms)) if norms else 0.0
            stats[f"grad/{name}/max"]  = float(np.max(norms)) if norms else 0.0

        # ── GradNorm weights (always useful) ────────────────────────────────────
        w_con  = float(torch.exp(self.log_w_con ).item())
        w_ctdn = float(torch.exp(self.log_w_ctdn).item())
        stats.update({"w_con": w_con, "w_ctdn": w_ctdn})

        # ── Optionally log *scaled* losses if train_epoch cached them ───────────
        if hasattr(self, "_last_raw_con") and hasattr(self, "_last_raw_ctdn"):
            stats["scaled_con"]  = w_con  * float(self._last_raw_con)
            stats["scaled_ctdn"] = w_ctdn * float(self._last_raw_ctdn)

        # ── Push to MLflow or console ───────────────────────────────────────────
        if mlflow.active_run():
            mlflow.log_metrics(stats, step=self.current_epoch)
        else:
            for k, v in stats.items():
                logging.debug(f"{k}: {v:.4f}")

    
    @staticmethod
    def _log_evaluation_metrics(metrics: dict, prefix: str = "val") -> None:
        """
        Log evaluation metrics to mlflow.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics.
            prefix (str): Prefix for the metric names.
        """
        for name, val in metrics.items():
            if name in Stage1Trainer.higher_is_better:
                key = f"{prefix}/higher_is_better/{name}"
            elif name in Stage1Trainer.lower_is_better:
                key = f"{prefix}/lower_is_better/{name}"
            else:
                key = f"{prefix}/{name}"
            mlflow.log_metric(key, float(val))

    @staticmethod
    def save_pytorch_model(model: nn.Module, path: str) -> None:
        """
        Save the PyTorch model to a file.
        
        Args:
            model (nn.Module): The PyTorch model to save.
            path (str): Path to save the model.
        """
        import traceback
        try:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model, path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.info(f"Failed to save model to {path} with error\n")
            logging.info(f"{traceback.format_exc()}")

    def _gradnorm_step(self, raw_ctdn: torch.Tensor, raw_con: torch.Tensor):
        """
        One gradient‑balancing step (GradNorm) applied *after*
        back‑propagating loss_total and *before* optimiser.step().

        Args:
            raw_ctdn (torch.Tensor): Raw CTDN loss.
            raw_con (torch.Tensor): Raw content loss.
        
        Returns:
            None
        """
        # ── 0. bootstrap the reference losses ────────────────────────────────
        if self.L0_ctdn is None:          # happens only in the very first call
            self.L0_ctdn = raw_ctdn.detach().item()
            self.L0_con  = raw_con.detach().item()

        # ── 1. get the *current* task weights ────────────────────────────
        w_ctdn = torch.exp(self.log_w_ctdn)   # >0, trainable
        w_con  = torch.exp(self.log_w_con)    

        # ── 2. form *weighted* per‑task losses ───────────────────────────

        loss_ctdn_w = w_ctdn * raw_ctdn
        loss_con_w  = w_con  * raw_con

        # ── 3. pick the params to balance (only those still requires_grad) ─
        shared_params = [p for p in self.model.parameters() if p.requires_grad]

        # helper: compute mean L1 norm of task gradients (with graph)
        def _grad_norm(loss: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(
                loss, 
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
                grad_outputs=[torch.ones_like(loss)]
            )
            grads = [g for g in grads if g is not None]
            if not grads:
                return torch.tensor(0.0, device=self.device)
            norms = torch.stack([g.abs().mean() for g in grads])
            return norms.mean()

        # ── 4. actually compute the norms on the *weighted* losses ────────

        g_ctdn = _grad_norm(loss_ctdn_w)
        g_con = torch.clamp(_grad_norm(loss_con_w), min=1e-6)
        

        # ── 5. compute the GradNorm targets ─────────────
        alpha  = self.gradnorm_alpha
        r_ctdn = (raw_ctdn.detach() / self.L0_ctdn + 1e-8) ** alpha
        r_con  = (raw_con.detach()  / self.L0_con + 1e-8)  ** alpha
        C      = (g_ctdn + g_con).detach() / 2.0
        target_ctdn, target_con = C * r_ctdn, C * r_con

        # ── 6. form the squared‑error penalty ────────────────────────────
        loss_gradnorm = (g_ctdn - target_ctdn).pow(2) + (g_con - target_con).pow(2)

        if mlflow.active_run():
            mlflow.log_metric("gradnorm_loss", loss_gradnorm.item(), step=self.current_epoch)
            mlflow.log_metric("gradnorm/g_ctdn", g_ctdn.item(), step=self.current_epoch)
            mlflow.log_metric("gradnorm/g_con", g_con.item(), step=self.current_epoch)
            mlflow.log_metric("gradnorm/target_ctdn", target_ctdn.item(), step=self.current_epoch)
            mlflow.log_metric("gradnorm/target_con", target_con.item(), step=self.current_epoch)
        return loss_gradnorm