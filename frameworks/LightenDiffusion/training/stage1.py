from .base import BaseTrainer
import torch
import copy
import torch.nn as nn
from typing import List, Optional
from torch.utils.data import DataLoader
from .losses import (
    content_loss, 
    reconstruction_loss, 
    reflectance_consistency_loss, 
    illumination_smoothness_loss
)
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
import logging
from .tqdm_configuration import TqdmManager
import os
from frameworks.LightenDiffusion.models.stage1 import Stage1
import mlflow
import numpy as np
from typing import Dict, Tuple, Any

LOSS_REGISTRY = {
    "content": {
        "fn": content_loss,
        "inputs": {
            "reconstructions": "reconstructions",
            "input_images":    "low_imgs",
        },
    },
    "reconstruction": {
        "fn": reconstruction_loss,
        "inputs": {
            "reflectances":   "reflectances",
            "illuminations":  "illuminations",
            "features":       "encoded_features",
        },
    },
    "reflectance_consistency": {
        "fn": reflectance_consistency_loss,
        "inputs": {
            "reflectances": "reflectances",
        },
    },
    "illumination_smoothness": {
        "fn": illumination_smoothness_loss,
        "inputs": {
            "illuminations": "illuminations",
            "reflectances":  "reflectances",
        },
    },
}


LOGW_CLAMP_MIN, LOGW_CLAMP_MAX = -10.0, 10.0

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
        pretrain_content_ratio: float = .05, 
        pretrain_ctdn_ratio: float = .1,
        num_visualizations: int = 1,
        random_seed: int = 42,
        show_plot: bool = True,
        save_dir: str = None,
        gradnorm_alpha: float = 1.5,
        gradnorm_interval: int = 3,

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
            num_visualizations (int): Number of samples to visualize during validation.
            random_seed (int): Seed for random selection of samples.
            show_plot (bool): Whether to show the plots when validating the model
            save_dir (str, optional): Base Local Directory to save training artifcats (visuals+models) if provided.
            gradnorm_alpha (float): Exponent for GradNorm balancing.
            pretrain_content_ratio (float): Ratio of epochs to pretrain content loss.
            pretrain_ctdn_ratio (float): Ratio of epochs to pretrain ctdn loss.
            gradnorm_interval (int): Interval for computing GradNorm updates.
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
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in LOSS_REGISTRY
        })
        self.initial_raw_losses  = {name: None for name in LOSS_REGISTRY}
        self.gradnorm_alpha = gradnorm_alpha
        self.pretrain_content_ratio = pretrain_content_ratio
        self.pretrain_ctdn_ratio = pretrain_ctdn_ratio
        self.weight_optimizer = torch.optim.Adam(
            self.log_weights.values(),
            lr=self.optimizer.param_groups[0]["lr"] * .5,
            betas=(0.9, 0.999),
        )
        self.gradnorm_interval = gradnorm_interval
        self._batch_counter = 0
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
        builds kwargs from LOSS_REGISTRY[name]['inputs']
        Args:
            low_imgs (torch.Tensor): Low-light images of shape [B, m, 3, H, W].
            reflectances (torch.Tensor): Reflectance outputs of shape [B, m, C, H/2^k, W/2^k].
            illuminations (torch.Tensor): Illumination outputs of shape [B, m, C, H/2^k, W/2^k].
            reconstructions (torch.Tensor): Decoder reconstructions of shape [B, m, 3, H/2^k, 2^k].
            encoded_features (torch.Tensor): Encoded features of shape [B, m, C, H/2^k, W/2^k].
        Returns:
            tuple: 
                - loss_total (torch.Tensor): Total loss.
                - raw_losses (dict): Dictionary of raw losses.
                - weighted_losses (dict): Dictionary of weighted losses.
        """

        raw_losses = {}
        weighted_losses = {}
        inputs = {
            'low_imgs'      : low_imgs,
            'reflectances'  : reflectances,
            'illuminations' : illuminations,
            'reconstructions': reconstructions,
            'encoded_features': encoded_features,
        }
        for name, info in LOSS_REGISTRY.items():
            fn = info["fn"]
            mapping = info["inputs"]
            kwargs = {}
            for param, src in mapping.items():
                if src not in inputs:
                    raise KeyError(f"LOSS_REGISTRY[{name}]: expected input '{src}' but it's missing")
                kwargs[param] = inputs[src]
            raw_losses[name] = fn(**kwargs)

        for name, raw in raw_losses.items():
            w = torch.exp(self.log_weights[name])
            weighted_losses[name] = w * raw

        loss_total = sum(weighted_losses.values())
        return loss_total, raw_losses, weighted_losses


    @with_curriculum
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with integrated GradNorm balancing.

        Returns:
            A dict mapping metric names to values for this epoch.
        """
        self.model.train()
        self.current_epoch += 1
        self._batch_counter = 0

        batch_bar = self._make_tqdm()
        running_total, running_raw, running_weighted = self._init_running_stats()

        for i, batch in enumerate(self.train_loader):
            loss_total, raw_losses, weighted_losses = self._process_batch(batch)
            running_total, running_raw, running_weighted = self._accumulate_stats(
                running_total, running_raw, running_weighted,
                loss_total, raw_losses, weighted_losses
            )
            self._update_progress(batch_bar, running_total, running_weighted, i)

        batch_bar.close()
        return self._finalize_metrics(running_total, running_raw, running_weighted)
    
    def _make_tqdm(self) -> TqdmManager:
        """Create the batch progress bar."""
        return TqdmManager(
            total=len(self.train_loader),
            desc="Training Batches",
            leave=True,
            unit="batch",
        )
    
    def _init_running_stats(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Initialize accumulators.

        Returns:
            running_total: sum of total losses,
            running_raw: dict of sum of raw losses,
            running_weighted: dict of sum of weighted losses
        """
        running_total = 0.0
        running_raw = {name: 0.0 for name in LOSS_REGISTRY}
        running_weighted = {name: 0.0 for name in LOSS_REGISTRY}
        return running_total, running_raw, running_weighted
    
    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward, loss computation, GradNorm, backward, and optimizer step on one batch.

        Args:
            batch: (low_imgs, _) where low_imgs is a tensor.

        Returns:
            loss_total, raw_losses, weighted_losses
        """
        low_imgs, _ = batch
        low_imgs = low_imgs.to(self.device)
        self._batch_counter += 1
        outputs = self.model(low_imgs)
        loss_tensors = self._gather_tensors(low_imgs, outputs)
        loss_total, raw_losses, weighted_losses = self.calculate_loss(*loss_tensors)
        self._last_raw = {n: raw_losses[n].item() for n in raw_losses}
        self._last_weighted = {n: weighted_losses[n].item() for n in weighted_losses}
        self.optimizer.zero_grad()
        self.weight_optimizer.zero_grad()
        loss_total.backward(retain_graph=True)
        gradnorm_loss = self._maybe_gradnorm(raw_losses)
        gradnorm_loss.backward()
        self.optimizer.step()
        self.weight_optimizer.step()
        self._clamp_log_weights()
        self._log_gradient_stats()

        return loss_total, raw_losses, weighted_losses
    
    def _maybe_gradnorm(self, raw_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute GradNorm loss only every gradnorm_interval batches.

        Returns:
            The GradNorm loss tensor or zero.
        """
        if self._batch_counter % self.gradnorm_interval == 0:
            return self._gradnorm_step(raw_losses)
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _clamp_log_weights(self) -> None:
        """Clamp log_weights data for numerical stability."""
        for log_w in self.log_weights.values():
            log_w.data.clamp_(LOGW_CLAMP_MIN, LOGW_CLAMP_MAX)

    def _accumulate_stats(
        self,
        running_total: float,
        running_raw: Dict[str, float],
        running_weighted: Dict[str, float],
        loss_total: torch.Tensor,
        raw_losses: Dict[str, torch.Tensor],
        weighted_losses: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Update running sums with this batch’s losses.

        Returns:
            Updated running_total, running_raw, running_weighted
        """
        running_total += loss_total.item()
        for name in LOSS_REGISTRY:
            running_raw[name] += raw_losses[name].item()
            running_weighted[name] += weighted_losses[name].item()
        return running_total, running_raw, running_weighted

    def _update_progress(
        self,
        batch_bar: TqdmManager,
        running_total: float,
        running_weighted: Dict[str, float],
        batch_idx: int,
    ) -> None:
        """
        Update the progress bar with current averages.

        Args:
            batch_bar: the TqdmManager instance.
            running_total: sum of total losses so far.
            running_weighted: dict of weighted loss sums so far.
            batch_idx: index of the current batch (0-based).
        """
        avg_total = running_total / (batch_idx + 1)
        postfix = {"total": f"{avg_total:.4f}"}
        for name in LOSS_REGISTRY:
            avg_weighted = running_weighted[name] / (batch_idx + 1)
            postfix[f"weighted_{name}"] = f"{avg_weighted:.4f}"
        
        batch_bar.set_postfix(**postfix)
        batch_bar.update(1)

    def _finalize_metrics(
        self,
        running_total: float,
        running_raw: Dict[str, float],
        running_weighted: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute final epoch metrics.

        Returns:
            A dict with 'total_loss', 'raw_<name>' and 'weighted_<name>' entries.
        """
        N = len(self.train_loader)
        metrics: Dict[str, float] = {
            "total_loss": running_total / N
        }
        for name in LOSS_REGISTRY:
            metrics[f"raw_{name}"]      = running_raw[name]      / N
            metrics[f"weighted_{name}"] = running_weighted[name] / N
        return metrics

    def _apply_curriculum(self, epoch: int) -> None:
        """
        Soft curriculum via linear ramps on the log‐weights.

        We define two phase boundaries:
          t1 = pretrain_content_ratio  ⋅ num_epochs
          t2 = (pretrain_content_ratio + pretrain_ctdn_ratio) ⋅ num_epochs

        • Phase 1 (0 ≤ epoch < t1):
            content weight ramps from 0 → 1  
            CTDN sub‐loss weights stay at 0  
        • Phase 2 (t1 ≤ epoch < t2):
            content weight = 1  
            CTDN sub‐loss weights ramp from 0 → 1  
        • Phase 3 (epoch ≥ t2):
            all weights = 1  (and thereafter GradNorm is free to adapt)

        Args:
            epoch: zero‐based index of the current epoch.
        """
        T  = self.num_epochs
        t1 = int(self.pretrain_content_ratio * T)
        t2 = int((self.pretrain_content_ratio + self.pretrain_ctdn_ratio) * T)

        if t1 > 0:
            content_scale = min(1.0, epoch / t1)
        else:
            content_scale = 1.0

        if epoch < t1 or t2 <= t1:
            ctdn_scale = 0.0
        else:
            ctdn_scale = min(1.0, (epoch - t1) / (t2 - t1))

        eps = 1e-8
        self.log_weights['content'].data.fill_(
            np.log(content_scale + eps)
        )
        for name in ('reconstruction',
                     'reflectance_consistency',
                     'illumination_smoothness'):
            self.log_weights[name].data.fill_(
                np.log(ctdn_scale + eps)
            )

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
        Collect and log:
            • per‐module grad norms (encoder/decomposer/decoder)
            • all learned task weights and their grads
            • optionally the last raw and weighted losses for each task

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

        # learned task weights + their gradients
        for task_name, log_w in self.log_weights.items():
            w = float(torch.exp(log_w).item())
            stats[f"w/{task_name}"] = w

            if log_w.grad is not None:
                stats[f"grad_w/{task_name}"] = float(log_w.grad.abs().mean().item())
            else:
                stats[f"grad_w/{task_name}"] = 0.0

        # log the last raw & weighted losses if available
        if hasattr(self, "_last_raw") and hasattr(self, "_last_weighted"):
            for task_name, raw_val in self._last_raw.items():
                stats[f"raw/{task_name}"] = float(raw_val)
                stats[f"weighted/{task_name}"] = float(self._last_weighted[task_name])

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

    def _gradnorm_step(self, raw_losses: dict) -> torch.Tensor:
        """
        Compute the GradNorm loss for balancing the task weights.
        This function computes the GradNorm loss based on the gradients of the raw losses
        with respect to the model parameters. It uses the log weights to scale the raw losses
        and computes the GradNorm loss to balance the task weights.
        The GradNorm loss is computed as the sum of squared differences between the gradients
        and the target gradients, which are scaled by the raw losses and the initial L0 values.

        Args:
            raw_losses (dict): Dictionary of raw losses for each task.
                Each key is the name of the task and the value is the raw loss tensor.        
        Returns:
            torch.Tensor: The computed GradNorm loss.
            
        """
        for name, raw in raw_losses.items():
            if self.initial_raw_losses [name] is None:
                self.initial_raw_losses [name] = raw.detach().item()
        
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        grads = {}

        for name, raw in raw_losses.items():
            lw = torch.clamp(self.log_weights[name], LOGW_CLAMP_MIN , LOGW_CLAMP_MAX) # clamp before exp for numerical stability
            w = torch.exp(lw)
            g = torch.autograd.grad(
                w*raw,
                shared_params,
                retain_graph=True,
                create_graph=True,
                grad_outputs=[torch.ones_like(raw)],
                allow_unused=True,
            )
            norms = [t.abs().mean() for t in g if t is not None]
            grads[name] = torch.stack(norms).mean() if norms else torch.tensor(0.)
        
        alpha = self.gradnorm_alpha
        rates = {
            name: (raw_losses[name].detach() / self.initial_raw_losses [name] + 1e-8) ** alpha
            for name in raw_losses
        }

        C = sum(grads.values()).detach() / len(grads)
        targets = {name: C * rates[name] for name in grads}
        loss_gradnorm = sum((grads[name] - targets[name]) ** 2 for name in grads)

        if mlflow.active_run():
            mlflow.log_metric("gradnorm_loss", loss_gradnorm.item(), step=self.current_epoch)
            for name, raw in raw_losses.items():
                mlflow.log_metric(f"gradnorm/raw_{name}", raw.item(), step=self.current_epoch)
                mlflow.log_metric(f"gradnorm/target_{name}", targets[name].item(), step=self.current_epoch)
                mlflow.log_metric(f"gradnorm/g_{name}", grads[name].item(), step=self.current_epoch)
        return loss_gradnorm 
    