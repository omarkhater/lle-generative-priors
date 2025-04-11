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


def compute_metrics_paired(avg_recon: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute full-reference metrics for one sample.

    Args:
        avg_recon (torch.Tensor): Averaged reconstruction of shape [3, H, W].
        gt (torch.Tensor): Ground truth image of shape [3, H, W].
    
    Returns:
        dict: Dictionary containing PSNR, SSIM, LPIPS, NIQE, and PI metrics.
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

def compute_metrics_unpaired(avg_recon: torch.Tensor, sample_recons: torch.Tensor) -> dict:
    """
    Compute no-reference metrics for one sample.
    
    Args:
        avg_recon (torch.Tensor): Averaged reconstruction of shape [3, H, W].
        sample_recons (torch.Tensor): Tensor of shape [m, 3, H, W] representing the m reconstructions 
                                       for a given sample.
    
    Returns:
        dict: Dictionary containing NIQE and consistency metrics.
    """
    niqe_value = compute_niqe(avg_recon.unsqueeze(0))
    consistency = compute_pairwise_lpips_consistency(sample_recons)
    return {
        "niqe": niqe_value,
        "consistency_lpips": consistency
    }

def compute_pairwise_lpips_consistency(recons: torch.Tensor) -> float:
    """
    Compute the average pairwise LPIPS distance among m reconstructions from a single sample.
    
    Args:
        recons (torch.Tensor): Tensor of shape [m, 3, H, W] representing the m reconstructions 
                               for a given sample.
    
    Returns:
        float: The average pairwise LPIPS distance as a consistency metric (lower is better).
               If m == 1, returns 0.
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


def evaluate_stage1_metrics(model: torch.nn.Module,
                            dataloader: DataLoader,
                            paired: bool = True,
                            device: str = "cuda") -> dict:
    """
    Evaluate the Stage1 model quantitatively using reconstruction quality and/or consistency metrics.
    
    For each sample, the Stage1 model processes m low-quality images (with shape [B, m, 3, H, W])
    to produce m reconstructions via its encoder, decomposer, and decoder. In this evaluation, the
    m reconstructions are averaged (pixel-wise) to yield one representative output per sample.
    
    In the paired setting (if the data loader provides a ground truth reference image per sample),
    full-reference metrics are computed (PSNR, SSIM, LPIPS, NIQE, and the composite perceptual index).
    
    In the unpaired setting (without ground truth), the evaluation computes:
      - A no-reference NIQE metric on the averaged reconstruction.
      - A consistency metric computed as the average pairwise LPIPS distance among the 
        m reconstructions (to gauge whether different low-quality inputs yield similar outputs).
    
    Args:
        model (torch.nn.Module): The Stage1 model to be evaluated.
        dataloader (DataLoader): DataLoader providing inputs. If paired=True, each sample must be a tuple (x, y)
                                 where x has shape [B, m, 3, H, W] and y has shape [B, 3, H, W].
                                 If unpaired, sample[0] is used.
        paired (bool): Indicates if a reference ground truth image is available.
        device (str): Device for running evaluation (e.g., "cuda" or "cpu").
    
    Returns:
        dict: Aggregated metrics over the evaluation dataset.
    """
    validate_dataloader(dataloader, expected_dims=5, paired=paired)
    
    model.to(device)
    model.eval()
    metrics_list = [] 
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating Stage1"):
            if paired:
                x, y = sample  # x: [B, m, 3, H, W], y: [B, 3, H, W]
            else:
                x = sample[0]
            x = x.to(device)
            if paired:
                y = y.to(device)
            
            stage1_outputs = model(x)
            all_recons = torch.stack([out["recon"] for out in stage1_outputs], dim=0)
            avg_recons = all_recons.mean(dim=0)
            
            batch_size = avg_recons.shape[0]
            for i in range(batch_size):
                if paired:
                    sample_metrics = compute_metrics_paired(avg_recons[i], y[i])
                else:
                    sample_recons = all_recons[:, i, :, :, :]
                    sample_metrics = compute_metrics_unpaired(avg_recons[i], sample_recons)
                metrics_list.append(sample_metrics)
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics
