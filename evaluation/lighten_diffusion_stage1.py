import torch
import numpy as np
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


def compute_pairwise_lpips_consistency(recons: torch.Tensor) -> float:
    """
    Compute the average pairwise LPIPS distance among m reconstructions for one sample.
    
    Args:
        recons (torch.Tensor): Tensor of shape [m, 3, H, W].
    
    Returns:
        float: Average pairwise LPIPS distance as a consistency metric (lower is better).
    """
    m = recons.size(0)
    if m <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            lpips_val = compute_lpips(recons[i].unsqueeze(0), recons[j].unsqueeze(0))
            total += lpips_val
            count += 1
    return total / count if count > 0 else 0.0


# ---------- Metrics Computation on Averaged Output (Average-First Approach) ----------

def compute_metrics_paired_avg(avg_recon: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute full-reference metrics using the averaged reconstruction.

    Args:
        avg_recon (torch.Tensor): Averaged reconstruction of shape [3, H, W].
        gt (torch.Tensor): Ground truth image of shape [3, H, W].
    
    Returns:
        dict: Dictionary with PSNR, SSIM, LPIPS, NIQE, and PI.
    """
    recon_np = to_numpy(avg_recon)
    gt_np = to_numpy(gt)
    psnr_value = compute_psnr(gt_np, recon_np)
    ssim_value = compute_ssim(gt_np, recon_np)
    lpips_value = compute_lpips(gt.unsqueeze(0), avg_recon.unsqueeze(0))
    niqe_value = compute_niqe(avg_recon.unsqueeze(0))
    pi_value = compute_pi(lpips_value, niqe_value)
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "niqe": niqe_value,
        "pi": pi_value,
    }

def compute_metrics_unpaired_avg(avg_recon: torch.Tensor, sample_recons: torch.Tensor) -> dict:
    """
    Compute no-reference metrics using the averaged reconstruction for a sample.

    Args:
        avg_recon (torch.Tensor): Averaged reconstruction of shape [3, H, W].
        sample_recons (torch.Tensor): All m reconstructions of shape [m, 3, H, W].
    
    Returns:
        dict: Dictionary with NIQE and consistency (average pairwise LPIPS).
    """
    niqe_value = compute_niqe(avg_recon.unsqueeze(0))
    consistency = compute_pairwise_lpips_consistency(sample_recons)
    return {
        "niqe": niqe_value,
        "consistency_lpips": consistency
    }

# ---------- Metrics Computation on Individual Reconstructions (Per-Image Approach) ----------

def compute_metrics_paired_individual(recon: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute full-reference metrics for a single reconstruction (without prior averaging).

    Args:
        recon (torch.Tensor): A reconstruction image of shape [3, H, W].
        gt (torch.Tensor): Ground truth image of shape [3, H, W].
    
    Returns:
        dict: Dictionary with PSNR, SSIM, LPIPS, NIQE, and PI.
    """
    recon_np = to_numpy(recon)
    gt_np = to_numpy(gt)
    psnr_value = compute_psnr(gt_np, recon_np)
    ssim_value = compute_ssim(gt_np, recon_np)
    lpips_value = compute_lpips(gt.unsqueeze(0), recon.unsqueeze(0))
    niqe_value = compute_niqe(recon.unsqueeze(0))
    pi_value = compute_pi(lpips_value, niqe_value)
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "niqe": niqe_value,
        "pi": pi_value,
    }

def compute_metrics_unpaired_individual(recon: torch.Tensor) -> dict:
    """
    Compute no-reference metrics for a single reconstruction (no averaging with other samples).

    Args:
        recon (torch.Tensor): A reconstruction image of shape [3, H, W].
    
    Returns:
        dict: Dictionary with NIQE.
    """
    niqe_value = compute_niqe(recon.unsqueeze(0))
    return {
        "niqe": niqe_value
    }

# ---------- Evaluation Functions ----------

def evaluate_stage1_metrics_avgfirst(model: torch.nn.Module,
                                     dataloader: DataLoader,
                                     paired: bool = True,
                                     device: str = "cuda") -> dict:
    """
    Evaluate Stage1 using the "average-first" approach:
    Average the m reconstructions per sample first, then compute metrics.

    For paired evaluation, full-reference metrics are computed against the ground truth.
    For unpaired evaluation, no-reference NIQE and a consistency metric (on m reconstructions) are computed.

    Args:
        model (torch.nn.Module): Stage1 model.
        dataloader (DataLoader): Expects samples of (x, y) when paired (x: [B, m, C, H, W]; y: [B, C, H, W])
                                 or (x,) when unpaired.
        paired (bool): True if ground truth is available.
        device (str): Device, e.g., "cuda" or "cpu".
    
    Returns:
        dict: Aggregated metrics over the dataset.
    """
    validate_dataloader(dataloader, expected_dims=5, paired=paired)
    model.to(device)
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating (Average-First)"):
            if paired:
                x, y = sample
            else:
                x = sample[0]
            x = x.to(device)
            if paired:
                y = y.to(device)
            
            # Stage1 model outputs a list of m dictionaries; each dictionary contains "recon" of shape [B, C, H, W]
            stage1_outputs = model(x)
            all_recons = torch.stack([out["recon"] for out in stage1_outputs], dim=0)  # [m, B, C, H, W]
            avg_recons = all_recons.mean(dim=0)  # [B, C, H, W]
            
            batch_size = avg_recons.shape[0]
            for i in range(batch_size):
                if paired:
                    sample_metrics = compute_metrics_paired_avg(avg_recons[i], y[i])
                else:
                    sample_recons = all_recons[:, i, :, :, :]
                    sample_metrics = compute_metrics_unpaired_avg(avg_recons[i], sample_recons)
                metrics_list.append(sample_metrics)
    
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics

def evaluate_stage1_metrics_individual(model: torch.nn.Module,
                                       dataloader: DataLoader,
                                       paired: bool = True,
                                       device: str = "cuda") -> dict:
    """
    Evaluate Stage1 using the "individual" approach:
    Compute metrics for each of the m reconstructions per sample, then average the metric values.

    For paired evaluation, metrics for each reconstruction are computed against the ground truth,
    and then the values are averaged over m.
    For unpaired evaluation, no-reference metrics are computed per reconstruction, then averaged.
    
    Additionally, for unpaired evaluation, a consistency metric (average pairwise LPIPS)
    is computed over the m reconstructions.
    
    Args:
        model (torch.nn.Module): Stage1 model.
        dataloader (DataLoader): Expects samples of (x, y) when paired (x: [B, m, C, H, W]; y: [B, C, H, W])
                                 or (x,) when unpaired.
        paired (bool): True if a ground truth image is available.
        device (str): Device, e.g., "cuda" or "cpu".
    
    Returns:
        dict: Aggregated metrics over the dataset.
    """
    validate_dataloader(dataloader, expected_dims=5, paired=paired)
    model.to(device)
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating (Individual)"):
            if paired:
                x, y = sample
            else:
                x = sample[0]
            x = x.to(device)
            if paired:
                y = y.to(device)
            
            stage1_outputs = model(x)
            # all_recons shape: [m, B, C, H, W]
            all_recons = torch.stack([out["recon"] for out in stage1_outputs], dim=0)
            m, B, C, H, W = all_recons.shape
            
            batch_metrics = []
            for i in range(B):
                sample_metrics = {}
                # Collect per-reconstruction metric dictionaries for sample i.
                per_recon_metrics = []
                for j in range(m):
                    recon_j = all_recons[j, i]
                    if paired:
                        metrics_dict = compute_metrics_paired_individual(recon_j, y[i])
                    else:
                        metrics_dict = compute_metrics_unpaired_individual(recon_j)
                    per_recon_metrics.append(metrics_dict)
                
                # Average metrics over the m reconstructions.
                avg_metrics = {}
                keys = per_recon_metrics[0].keys()
                for key in keys:
                    avg_metrics[key] = sum(d[key] for d in per_recon_metrics) / m
                
                # For unpaired evaluation, still compute consistency across all m reconstructions.
                if not paired:
                    consistency = compute_pairwise_lpips_consistency(all_recons[:, i, :, :, :])
                    avg_metrics["consistency_lpips"] = consistency
                batch_metrics.append(avg_metrics)
            
            metrics_list.extend(batch_metrics)
    
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics