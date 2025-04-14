import torch
import torch.nn.functional as F
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
from frameworks.LightenDiffusion.visualization.visualization_utils import map_to_rgb

# -------------------------------------------------------------------------------
# Unsupervised metric: Total Variation (TV) for illumination smoothness
# -------------------------------------------------------------------------------
def compute_total_variation(img: torch.Tensor) -> float:
    """
    Compute the total variation of an illumination map as a measure of local smoothness.
    Lower values indicate a smoother (more locally consistent) illumination.
    
    Args:
        img (torch.Tensor): Illumination map of shape [1, H, W] or [H, W].
        
    Returns:
        float: The total variation value.
    """
    # If image is of shape [1, H, W], squeeze the channel dimension.
    if img.dim() == 3 and img.size(0) == 1:
        img = img.squeeze(0)
    dh = torch.abs(img[:, 1:] - img[:, :-1])
    dw = torch.abs(img[1:, :] - img[:-1, :])
    tv = torch.mean(dh) + torch.mean(dw)
    return tv.item()

# -------------------------------------------------------------------------------
# Unsupervised metric: Reflectance consistency
# -------------------------------------------------------------------------------
def compute_pairwise_reflectance_consistency(reflectances: torch.Tensor) -> float:
    """
    Compute the average pairwise LPIPS distance among m reflectance maps for one sample.
    Lower values indicate that the reflectance maps are invariant to input variations.
    
    Args:
        reflectances (torch.Tensor): Tensor of shape [m, C, H, W] for one sample.
    
    Returns:
        float: The average pairwise LPIPS distance.
    """
    m = reflectances.size(0)
    if m <= 1:
        return 0.0
    total = 0.0
    count = 0
    # Compute pairwise differences (LPIPS metric) between all unique pairs.
    for i in range(m):
        for j in range(i + 1, m):
            lpips_val = compute_lpips(reflectances[i].unsqueeze(0), reflectances[j].unsqueeze(0))
            total += lpips_val
            count += 1
    return total / count if count > 0 else 0.0

# -------------------------------------------------------------------------------
# Supervised metrics computed on each individual reconstruction.
# The input image (first low-quality input) is used as the proxy ground truth.
# -------------------------------------------------------------------------------
def compute_supervised_metrics_individual(recon: torch.Tensor, input_img: torch.Tensor) -> dict:
    """
    Compute supervised metrics by comparing a reconstructed image with the input image.
    This includes PSNR, SSIM, LPIPS, NIQE, and a perceptual index (PI).
    
    Args:
        recon (torch.Tensor): Reconstructed image of shape [3, H, W].
        input_img (torch.Tensor): Original input image (first low-quality image) of shape [3, H, W].
    
    Returns:
        dict: Dictionary with keys "psnr", "ssim", "lpips", "niqe", and "pi".
    """
    # Convert tensors to NumPy for PSNR/SSIM computations.
    recon_np = to_numpy(recon)
    input_np = to_numpy(input_img)
    
    psnr_value = compute_psnr(input_np, recon_np)
    ssim_value = compute_ssim(input_np, recon_np)
    lpips_value = compute_lpips(input_img.unsqueeze(0), recon.unsqueeze(0))
    niqe_value = compute_niqe(recon.unsqueeze(0))
    pi_value = compute_pi(lpips_value, niqe_value)
    
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "niqe": niqe_value,
        "pi": pi_value,
    }

# -------------------------------------------------------------------------------
# Evaluation: Compute metrics for each sample based on individual reconstructions.
# Supervised metrics are computed per reconstruction and then averaged.
# In addition, unsupervised metrics on the illumination and reflectance are computed.
# -------------------------------------------------------------------------------
def evaluate_stage1_metrics_individual(model: torch.nn.Module,
                                       dataloader: DataLoader,
                                       device: str = "cuda") -> dict:
    """
    Evaluate Stage1 performance using individual outputs rather than an averaged output.
    For each sample, supervised metrics (comparing each reconstruction with the input image)
    are computed and then averaged. Additionally, unsupervised metrics are computed:
      - Total variation (TV) for the illumination maps (to quantify local smoothness)
      - Pairwise LPIPS consistency for the reflectance maps (to quantify invariance)
    
    Args:
        model (torch.nn.Module): The Stage1 model.
        dataloader (DataLoader): DataLoader yielding samples with shape (x,) where x is [B, m, 3, H, W].
                                 (The first low-quality image in x is used as the input reference.)
        device (str): Computation device ("cuda", "cpu", etc.).
    
    Returns:
        dict: Aggregated metrics over the dataset.
    """
    validate_dataloader(dataloader, expected_dims=5, paired=False)
    model.to(device)
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating Stage1 (Individual)"):
            # Assume dataloader returns a tuple: (x,) where x has shape [B, m, 3, H, W]
            x = sample[0].to(device)
            B, m, C, H, W = x.shape

            # Run the Stage1 model: expects x of shape [B, m, 3, H, W] and returns a list with m dicts.
            # Each dict contains keys "recon", "R", and "L" (each of shape [B, 3, H, W] for "recon"
            # and [B, C, H', W'] for "R" and "L").
            stage1_outputs = model(x)
            
            all_recons = torch.stack([out["recon"] for out in stage1_outputs], dim=0)
            all_illum = torch.stack([out["L"] for out in stage1_outputs], dim=0)
            all_refl  = torch.stack([out["R"] for out in stage1_outputs], dim=0)
            all_refl_rgb = convert_maps_to_rgb(all_refl)
            
            for i in range(B):
                sample_supervised = []
                # Compute supervised metrics for each reconstruction compared with the input.
                # Use the first low-quality image as a proxy for ground truth.
                input_img = x[i, 0]  # shape [3, H, W]
                for j in range(m):
                    recon_j = all_recons[j, i]  # shape [3, H, W]
                    metrics_dict = compute_supervised_metrics_individual(recon_j, input_img)
                    sample_supervised.append(metrics_dict)
                
                # Average the supervised metrics over m reconstructions.
                avg_supervised = {}
                keys = sample_supervised[0].keys()
                for key in keys:
                    avg_supervised[key] = sum(d[key] for d in sample_supervised) / m

                # Compute unsupervised metric: total variation on illumination.
                # Average the total variation across the m illumination maps for this sample.
                tv_values = []
                for j in range(m):
                    illum_j = all_illum[j, i]  # may be shape [3, H, W] or [1, H, W]
                    tv_values.append(compute_total_variation(illum_j))
                avg_tv = sum(tv_values) / m

                # Compute unsupervised metric: reflectance consistency.
                # Pass all m reflectance maps for sample i (shape: [m, C, H, W]).
                refl_consistency = compute_pairwise_reflectance_consistency(all_refl_rgb[i, :, :, :, :])

                # Combine supervised and unsupervised metrics.
                sample_metrics = {
                    **avg_supervised,
                    "tv_illumination": avg_tv,
                    "reflectance_consistency": refl_consistency,
                }
                metrics_list.append(sample_metrics)
    
    # Aggregate metrics over the entire dataset.
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics


def convert_maps_to_rgb(all_maps):
    """
    Convert a batch of image maps to RGB images.

    This function expects an input tensor with shape [m, B, C, H, W] and returns a tensor
    of shape [B, m, 3, H, W]. The conversion is performed by transposing the input to shape
    [B, m, C, H, W] and then applying a map-to-RGB conversion (using the map_to_rgb function)
    for each sample in the batch.

    Args:
        all_maps (torch.Tensor): Input tensor with shape [m, B, C, H, W],
                                  where m is the number of maps per sample,
                                  B is the batch size, C is the number of channels,
                                  and H, W are spatial dimensions.

    Returns:
        torch.Tensor: Output tensor with RGB images, shape [B, m, 3, H, W].
    """
    all_maps = all_maps.transpose(0, 1)
    rgb_list = []
    for i in range(all_maps.shape[0]):  # iterate over B
        rgb = map_to_rgb(all_maps[i])  # output: [m, 3, H, W]
        rgb_list.append(rgb)
    rgb_batch = torch.stack(rgb_list, dim=0)
    return rgb_batch