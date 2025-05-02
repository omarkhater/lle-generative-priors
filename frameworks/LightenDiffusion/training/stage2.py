from .base import BaseTrainer
import torch
from .tqdm_configuration import TqdmManager
from typing import Tuple, Optional, Dict, Any, List
from torch.utils.data import DataLoader
from .losses import noise_loss, self_constrained_consistency_loss
from frameworks.LightenDiffusion.visualization.visualize_stage2 import visualize_stage2_results
import logging, os
import mlflow
import numpy as np
from evaluation.lighten_diffusion_stage2 import evaluate_stage2_metrics_avgfirst
import torch.nn as nn
import functools

LOSS_REGISTRY = {
    "diffusion_loss": {
        "fn": noise_loss,
        "inputs": {
            "predicted_noise": "pred_noise", 
            "noise_target": "true_noise"
            },
        },
    "scc_loss": {
        "fn": self_constrained_consistency_loss,
        "inputs": {
            "f_low_hat": "restored_features",
            "f_low": "reference_feature"
        }
    }
}
LOGW_CLAMP_MIN, LOGW_CLAMP_MAX = -10.0, 10.0
EPS = 1e-8

def with_curriculum(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self._apply_curriculum(self.current_epoch)
        return fn(self, *args, **kwargs)
    return wrapper


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
    higher_is_better = {"psnr", "ssim"}
    lower_is_better = {"tv_illumination", "pi", "niqe", "lpips"}
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
                 after_validate: bool = True,
                 save_visualization_dir: Optional[str] = None,
                 gradnorm_alpha: float = 1.5,
                 gradnorm_interval: int = 3,
                 scc_patience: int = 5,
                 scc_ramp_length: int = 5,
                 curriculum_keys: List[str] = None,

                 ) -> None:
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
            curriculum_keys=curriculum_keys or ["diffusion_loss"]
            )
        
        # configurations
        self.lambda_scc = lambda_scc
        self.gamma = gamma
        self.num_visualization = num_visualization
        self.random_seed = random_seed
        self.save_visualization_dir = save_visualization_dir

        # gradnorm 
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in LOSS_REGISTRY
        })
        self.initial_raw_losses  = {name: None for name in LOSS_REGISTRY}
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_interval = gradnorm_interval
        self.weight_optimizer = torch.optim.Adam(
            self.log_weights.values(),
            lr=self.optimizer.param_groups[0]["lr"] * 0.5,
            betas=(0.9, 0.999)
        )

        self.betas = betas
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_sampling_timesteps = num_sampling_timesteps
        
        # curriculum state for SCC ramp
        self.scc_patience = scc_patience
        self.scc_ramp_length = scc_ramp_length
        self.ramp_started = False
        self.ramp_epoch = None

        self.after_validate = after_validate
        self.all_val_metrics = []
        self.best_diff_loss = float('inf')
        self.diff_plateau_cnt = 0

        self.current_epoch = 0
        self.best_loss = float('inf')
        
        self.initial_weighted_losses = {name: None for name in LOSS_REGISTRY}
        self.initial_total_loss = None
        self.curriculum_keys = curriculum_keys
        self._batch_counter = 0
        
        
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

    def _apply_curriculum(self, epoch: int) -> None:
        """
        Applies a curriculum learning strategy to the loss weights.
        The diffusion loss weight is ramped up from a small value to 1.0 over a specified number of epochs.
        The SCC loss weight is ramped down from 1.0 to a small value over the same number of epochs.
        Args:
            epoch (int): The current epoch number.
        
        Returns:
            None
        """
        self.log_weights['diffusion_loss'].data.fill_(0.0)

        if not self.ramp_started:
            fill = np.log(EPS)
        else:
            it = epoch - self.ramp_epoch
            frac = min(1.0, max(0.0, it / self.ramp_length))
            fill = np.log(frac + EPS)
        self.log_weights['scc_loss'].data.fill_(fill)


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
        
        weighted_losses = {}
        raw_losses = {}
        inputs = {
            "restored_features": restored_features,
            "reference_feature": reference_feature,
            "pred_noise": pred_noise,
            "true_noise": true_noise
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
        Train for one epoch over the training data.

        Returns:
            Dict[str, float]: A dict containing metrics for this epoch.
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

    def _init_running_stats(self) -> Dict[str, float]:
        """ Initialize running stats for the epoch.
        Args:
            None
        Returns:
            running_total (float): Running total loss.
            running_raw (dict): Running raw losses.
            running_weighted (dict): Running weighted losses.
        """
        running_total = 0.0
        running_raw = {name: 0.0 for name in LOSS_REGISTRY}
        running_weighted = {name: 0.0 for name in LOSS_REGISTRY}
        return running_total, running_raw, running_weighted

    def _process_batch(
            self, 
            batch: tuple
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single batch through forward and backward passes.

        Args:
            batch: Tuple of (low_imgs, high_imgs)

        Returns:
            Tuple of (total_loss, diffusion_loss, scc_loss)
        """
        low_imgs, high_imgs = self.ensure_on_device(batch)
        self._batch_counter += 1
        restored, reference, pred_noise, true_noise = self._gather_tensors(
            low_imgs, high_imgs)
        loss_total, raw_losses, weighted_losses = self.calculate_loss(
            restored, reference, pred_noise, true_noise)
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

    def _accumulate_stats(
        self,
        running_total: float,
        running_raw: Dict[str, float],
        running_weighted: Dict[str, float],
        loss_total: torch.Tensor,
        raw_losses: Dict[str, torch.Tensor],
        weighted_losses: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Accumulate stats for the current batch.
        
        Args:
            running_total (float): Running total loss.
            running_raw (dict): Running raw losses.
            running_weighted (dict): Running weighted losses.
            loss_total (torch.Tensor): Total loss for the current batch.
            raw_losses (dict): Raw losses for the current batch.
            weighted_losses (dict): Weighted losses for the current batch.
        Returns:
            running_total (float): Updated running total loss.
            running_raw (dict): Updated running raw losses.
            running_weighted (dict): Updated running weighted losses.
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
        batch_idx: int
    ) -> None:
        """
        Update the progress bar with current metrics.

        Args:
            batch_bar (TqdmManager): The progress bar instance.
            running_total (float): Running total loss.
            running_weighted (dict): Running weighted losses.
            batch_idx (int): Current batch index.
        
        Returns:
            None
        """
        avg_total = running_total / (batch_idx + 1)
        postfix = {"total": f"{avg_total:.4f}"}
        for name in LOSS_REGISTRY:
            avg_w = running_weighted[name] / (batch_idx + 1)
            postfix[f"weighted_{name}"] = f"{avg_w:.4f}"
        batch_bar.set_postfix(**postfix)
        batch_bar.update(1)

    def _finalize_metrics(
        self,
        running_total: float,
        running_raw: Dict[str, float],
        running_weighted: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Finalize metrics for the epoch.

        Args:
            running_total (float): Running total loss.
            running_raw (dict): Running raw losses.
            running_weighted (dict): Running weighted losses.
        Returns:
            metrics (dict): Finalized metrics for the epoch.
        """

        N = len(self.train_loader)
        metrics = {"total_loss": running_total / N}
        for name in LOSS_REGISTRY:
            metrics[f"raw_{name}"] = running_raw[name] / N
            metrics[f"weighted_{name}"] = running_weighted[name] / N
        return metrics

    def validate_batch(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> Dict[str, Any]:
        """
        Process and compute loss for a single validation batch.

        Args:
            batch (tuple): A tuple containing (low_imgs, high_imgs).

        Returns:
            Dict[str,Any]: Must include 'total_loss' and 'raw_losses' per BaseTrainer.
        """
        low_imgs, high_imgs = batch
        total_loss, diffusion_loss, scc_loss = self.calculate_loss(
            *self._gather_tensors(low_imgs, high_imgs)
        )
        raw_losses = {
            "diffusion_loss": diffusion_loss,
            "scc_loss": scc_loss
        }
        return {
            "total_loss": total_loss,
            "raw_losses": raw_losses
        }

    @with_curriculum
    def after_validation(self, current_epoch: int) -> None:
        """
        Hook method called after validation. 
        This method evaluates the model on the validation set and logs the metrics.

        Args:
            current_epoch (int): The current epoch number.
        Returns: 
            None
        """
        val_metrics = evaluate_stage2_metrics_avgfirst(self.model, self.val_loader)
        self.all_val_metrics.append(val_metrics)
        self._log_evaluation_metrics(val_metrics, prefix="val")
        avg_diff = self._avg_val_raws.get("diffusion_loss", float("nan"))
        if avg_diff < self.best_diff_loss:
            self.best_diff_loss = avg_diff
            self.diff_plateau_cnt = 0
        else:
            self.diff_plateau_cnt += 1
        if not self.ramp_started and self.diff_plateau_cnt >= self.scc_patience:
            self.ramp_started = True
            self.ramp_epoch = current_epoch

        if self.num_visualization and self.save_visualization_dir:
            save_dir = os.path.join(self.save_visualization_dir, f"epoch_{current_epoch}")
            os.makedirs(save_dir, exist_ok=True)
            visualize_stage2_results(
                self.model,
                self.val_loader,
                num_samples=self.num_visualization,
                random_seed=self.random_seed,
                save_dir=save_dir
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

    def after_training(self) -> None:
        """
        Hook method called after training is complete.
        Log best_val_loss / best_val_epoch and save model via MLflow.
        """
        train_metrics = evaluate_stage2_metrics_avgfirst(
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
    
    @staticmethod
    def _log_evaluation_metrics(metrics: dict, prefix: str = "val") -> None:
        """
        Log evaluation metrics to mlflow.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics.
            prefix (str): Prefix for the metric names.
        """
        for name, val in metrics.items():
            if name in Stage2Trainer.higher_is_better:
                key = f"{prefix}/higher_is_better/{name}"
            elif name in Stage2Trainer.lower_is_better:
                key = f"{prefix}/lower_is_better/{name}"
            else:
                key = f"{prefix}/{name}"
            mlflow.log_metric(key, float(val))

    def _log_gradient_stats(self) -> None:
        """
        Log gradient norms for each top-level submodule of the Stage2 model.
        """
        stats = {}
        for name, module in self.model.named_children():
            norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
            stats[f"grad/{name}/mean"] = float(np.mean(norms)) if norms else 0.0
            stats[f"grad/{name}/min"]  = float(np.min(norms)) if norms else 0.0
            stats[f"grad/{name}/max"]  = float(np.max(norms)) if norms else 0.0
        if mlflow.active_run():
            mlflow.log_metrics(stats)

    def _maybe_gradnorm(self, raw_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the gradient norm loss if the current batch is a multiple of gradnorm_interval.
        Args:
            raw_losses (dict): Dictionary of raw losses.
        Returns:
            torch.Tensor: Gradient norm loss.
        """
        if self._batch_counter % self.gradnorm_interval != 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return self._gradnorm_step(raw_losses)
    
    def _gradnorm_step(self, raw_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the gradient norm loss based on the raw losses.
        
        Args:
            raw_losses (dict): Dictionary of raw losses.
        
        Returns:
            torch.Tensor: Gradient norm loss.
        """
        for name, raw in raw_losses.items():
            if self.initial_raw_losses[name] is None:
                self.initial_raw_losses[name] = raw.detach().item()
        shared = [p for p in self.model.parameters() if p.requires_grad]
        grads = {}
        for name, raw in raw_losses.items():
            lw = torch.clamp(self.log_weights[name], LOGW_CLAMP_MIN, LOGW_CLAMP_MAX)
            g = torch.autograd.grad(
                torch.exp(lw) * raw,
                shared,
                retain_graph=True,
                create_graph=True,
                grad_outputs=[torch.ones_like(raw)]
            )
            norms = [t.abs().mean() for t in g if t is not None]
            grads[name] = torch.stack(norms).mean() if norms else torch.tensor(0.)
        rates = {name: (raw_losses[name].detach() / self.initial_raw_losses[name] + EPS) ** self.gradnorm_alpha
                 for name in raw_losses}
        C = sum(grads.values()).detach() / len(grads)
        targets = {name: C * rates[name] for name in grads}
        loss = sum((grads[name] - targets[name]) ** 2 for name in grads)
        if mlflow.active_run():
            mlflow.log_metric("gradnorm_loss", loss.item(), step=self.current_epoch)
        return loss
    
    def _clamp_log_weights(self) -> None:
        """
        Clamp the log weights to a specified range.
        """
        for log_w in self.log_weights.values():
            log_w.data.clamp_(LOGW_CLAMP_MIN, LOGW_CLAMP_MAX)