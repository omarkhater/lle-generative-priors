
# evaluate_stage2.py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluation.metrics import (
    to_numpy,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_niqe,
    compute_pi
)
from evaluation.utils import validate_dataloader, aggregate_metrics

def compute_metrics_paired(enhanced: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute full-reference metrics for a single enhanced image and its ground truth.

    Args:
        enhanced (torch.Tensor): Enhanced image tensor of shape [3, H, W].
        gt (torch.Tensor): Ground truth image of shape [3, H, W].

    Returns:
        dict: Dictionary containing PSNR, SSIM, LPIPS, NIQE, and PI.
    """
    enhanced_np = to_numpy(enhanced)
    gt_np = to_numpy(gt)
    psnr_value = compute_psnr(gt_np, enhanced_np)
    ssim_value = compute_ssim(gt_np, enhanced_np)
    lpips_value = compute_lpips(gt.unsqueeze(0), enhanced.unsqueeze(0))
    niqe_value = compute_niqe(enhanced.unsqueeze(0))
    pi_value = compute_pi(lpips_value, niqe_value)
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "niqe": niqe_value,
        "pi": pi_value,
    }

def evaluate_stage2_metrics_avgfirst(model: torch.nn.Module,
                                     dataloader: DataLoader,
                                     device: str = "cuda") -> dict:
    """
    Evaluate Stage2 of the pipeline using the averaged reconstruction.
    
    For each sample, the model's predict() method is used to compute the final enhanced image
    (which internally aggregates multiple low-light inputs and applies reverse diffusion sampling).
    Full-reference metrics are then computed against the provided ground truth.

    Args:
        model (torch.nn.Module): The complete LightenDiffusionPipeline model.
        dataloader (DataLoader): Expects samples of (x, y) where:
                                   - x: low-light images, shape [B, m, C, H, W]
                                   - y: high-light images (ground truth), shape [B, C, H, W]
        device (str): Device for evaluation, e.g., "cuda" or "cpu".

    Returns:
        dict: Aggregated metrics over the dataset.
    """
    # Validate dataloader format (expecting dimension 5 for x)
    validate_dataloader(dataloader, expected_dims=5, paired=True)
    model.to(device)
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating Stage2 (Average-First)"):
            x, y = sample  # x: [B, m, C, H, W], y: [B, C, H, W]
            x = x.to(device)
            y = y.to(device)
            
            # Obtain the enhanced image using the pipeline's predict() method.
            # The predict() method already uses the aggregation and reverse diffusion sampling.
            enhanced_batch = model.predict(x)  # Expected shape: [B, 3, H, W]
            
            batch_size = enhanced_batch.size(0)
            for i in range(batch_size):
                sample_metrics = compute_metrics_paired(enhanced_batch[i], y[i])
                metrics_list.append(sample_metrics)
    
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics
