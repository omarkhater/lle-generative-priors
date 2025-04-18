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

def evaluate_classic_enhancement(model: torch.nn.Module,
                               dataloader: DataLoader,
                               device: str = "cuda") -> dict:
    """
    Evaluate classic image enhancement model using the same metrics as ML-based models.
    
    For each sample, the model's predict() method is used to compute the enhanced image.
    Full-reference metrics are then computed against the provided ground truth.

    Args:
        model (torch.nn.Module): The classic enhancement model.
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
        for sample in tqdm(dataloader, desc="Evaluating Classic Enhancement"):
            x, y = sample  # x: [B, m, C, H, W], y: [B, C, H, W]
            x = x.to(device)
            y = y.to(device)
            
            # Obtain the enhanced image using the enhancer's predict() method
            enhanced_batch = model.predict(x)  # Expected shape: [B, 3, H, W]
            
            batch_size = enhanced_batch.size(0)
            for i in range(batch_size):
                sample_metrics = compute_metrics_paired(enhanced_batch[i], y[i])
                metrics_list.append(sample_metrics)
    
    aggregated_metrics = aggregate_metrics(metrics_list)
    return aggregated_metrics

def visualize_classic_results(model: torch.nn.Module,
                             dataloader: DataLoader,
                             num_samples: int = 4,
                             device: str = "cuda"):
    """
    Visualize the results of classic enhancement compared to ground truth.
    
    Args:
        model (torch.nn.Module): The classic enhancement model.
        dataloader (DataLoader): Expects samples of (x, y) where:
                                   - x: low-light images, shape [B, m, C, H, W]
                                   - y: high-light images (ground truth), shape [B, C, H, W]
        num_samples (int): Number of samples to visualize.
        device (str): Device for evaluation, e.g., "cuda" or "cpu".
        
    Returns:
        matplotlib Figure showing original, enhanced, and ground truth images.
    """
    import matplotlib.pyplot as plt
    
    model.to(device)
    model.eval()
    
    # Get a batch
    for sample in dataloader:
        x, y = sample
        x = x.to(device)
        y = y.to(device)
        break
    
    # Select only the first num_samples
    x = x[:num_samples]
    y = y[:num_samples]
    
    # Get enhanced images
    with torch.no_grad():
        enhanced = model.predict(x)
    
    # Create figure
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Display original low-light image (first exposure)
        low_light = x[i, 0].cpu()
        enhanced_img = enhanced[i].cpu()
        gt_img = y[i].cpu()
        
        # Convert to numpy for displaying
        low_light_np = to_numpy(low_light) / 255.0
        enhanced_np = to_numpy(enhanced_img) / 255.0
        gt_np = to_numpy(gt_img) / 255.0
        
        axs[i, 0].imshow(low_light_np)
        axs[i, 0].set_title("Low-Light Input")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(enhanced_np)
        axs[i, 1].set_title("Classic Enhancement")
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(gt_np)
        axs[i, 2].set_title("Ground Truth")
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    return fig