import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import numpy as np

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of shape [3, H, W] with values in [0, 1] 
    to a NumPy array of shape [H, W, 3] (uint8).
    """
    t = t.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t

def get_visualization_batch(data_loader: DataLoader, is_paired: bool, device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Retrieve one batch from the dataloader and move it to the device.
    
    Args:
        data_loader (DataLoader): DataLoader yielding batches.
        is_paired (bool): Indicates if ground truth is available.
        device (str): Computation device.
        
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: sample_batch (shape [B, m, 3, H, W])
            and gt_batch (shape [B, 3, H, W]) if available; otherwise None.
    """
    for batch in data_loader:
        if is_paired:
            x, y = batch
            return x.to(device), y.to(device)
        else:
            return batch[0].to(device), None
    raise ValueError("DataLoader is empty.")


# =========================== Average-First Approach ===========================
def visualize_stage1_results_avg(stage1: torch.nn.Module,
                                 data_loader: DataLoader,
                                 num_samples: int = 8,
                                 is_paired: bool = True,
                                 seed: int = 42) -> None:
    torch.manual_seed(seed)
    """
    Visualize Stage1 outputs using the averaged output approach.
    
    For each sample, the m reconstructions (and components) are averaged over the m inputs.
    Columns (per sample) include:
      - Source low image (if m==2, display sub-image 0; otherwise average over m)
      - Target low image (if m==2)
      - Estimated Reflectance (averaged)
      - Estimated Illumination (averaged)
      - Reconstruction (averaged)
      - Ground Truth (if available)
    
    Args:
        stage1 (torch.nn.Module): The Stage1 model.
        data_loader (DataLoader): Loader yielding (x, y) in paired mode; otherwise, (x,).
        num_samples (int): Number of samples (rows) to visualize.
        is_paired (bool): Whether ground truth is available.
    """
    stage1.eval()
    device = next(stage1.parameters()).device
    sample_batch, gt_batch = get_visualization_batch(data_loader, is_paired, device)
    B, m, _, H, W = sample_batch.shape

    # Run the Stage1 model: outputs is a list (length m) of dicts with keys "R", "L", "recon"
    with torch.no_grad():
        outputs = stage1(sample_batch)
    
    # Aggregate outputs by averaging over m low-quality images.
    R_agg = torch.stack([out["R_rgb"] for out in outputs], dim=1).mean(dim=1)      # [B, 3, H, W]
    L_agg = torch.stack([out["L_rgb"] for out in outputs], dim=1).mean(dim=1)      # [B, 3, H, W]
    recon_agg = torch.stack([out["recon"] for out in outputs], dim=1).mean(dim=1)  # [B, 3, H, W]
    RL_agg = R_agg * L_agg
    
    # Determine source (and possibly target) low image.
    if m == 2:
        source_low = sample_batch[:, 0, :, :, :]  # [B, 3, H, W]
        target_low = sample_batch[:, 1, :, :, :]  # [B, 3, H, W]
    else:
        source_low = sample_batch.mean(dim=1)

    num_to_vis = min(num_samples, B)
    # Determine number of columns.
    if m == 2:
        num_cols = 7 if is_paired else 6
    else:
        num_cols = 6 if is_paired else 5

    fig, axes = plt.subplots(num_to_vis, num_cols, figsize=(4 * num_cols, 4 * num_to_vis))
    if num_to_vis == 1:
        axes = axes[None, :]
    
    for i in range(num_to_vis):
        col = 0
        # Column 1: Source Low.
        axes[i, col].imshow(tensor_to_image(source_low[i]))
        axes[i, col].set_title("Source Low")
        axes[i, col].axis("off")
        col += 1

        # Column 2: If m == 2, show Target Low.
        if m == 2:
            axes[i, col].imshow(tensor_to_image(target_low[i]))
            axes[i, col].set_title("Target Low")
            axes[i, col].axis("off")
            col += 1

        # Column 3: Aggregated Reflectance.
        axes[i, col].imshow(tensor_to_image(R_agg[i]))
        axes[i, col].set_title("Reflectance")
        axes[i, col].axis("off")
        col += 1

        # Column 4: Aggregated Illumination.
        axes[i, col].imshow(tensor_to_image(L_agg[i]))
        axes[i, col].set_title("Illumination")
        axes[i, col].axis("off")
        col += 1

        # Column 5: (R x L)
        axes[i, col].imshow(tensor_to_image(RL_agg[i]))
        axes[i, col].set_title("R x L")
        axes[i, col].axis("off")
        col += 1

        # Column 6: Aggregated Reconstruction.
        axes[i, col].imshow(tensor_to_image(recon_agg[i]))
        axes[i, col].set_title("Reconstruction")
        axes[i, col].axis("off")
        col += 1

        # Column 7: Ground Truth (if paired).
        if is_paired:
            axes[i, col].imshow(tensor_to_image(gt_batch[i]))
            axes[i, col].set_title("Ground Truth")
            axes[i, col].axis("off")
    
    plt.tight_layout()
    plt.show()


# =========================== Individual Approach ===========================
def visualize_stage1_results_individual(stage1: torch.nn.Module,
                                        data_loader: DataLoader,
                                        num_samples: int = 8,
                                        is_paired: bool = True,
                                        seed: int = 42) -> None:
    """
    Visualize Stage1 outputs by displaying individual reconstructions.
    
    For each sample, instead of averaging over the m low-quality inputs,
    each individual reconstruction is visualized. The layout per sample:
      - First column: Source low image (using the first sub-image or average over m).
      - Next m columns: Each of the m individual reconstructions.
      - Last column: Ground Truth (if available).
    
    Args:
        stage1 (torch.nn.Module): The Stage1 model.
        data_loader (DataLoader): Loader yielding (x, y) in paired mode; otherwise, (x,).
        num_samples (int): Number of samples to visualize.
        is_paired (bool): Whether ground truth is available.
    """
    torch.manual_seed(seed)
    stage1.eval()
    device = next(stage1.parameters()).device
    sample_batch, gt_batch = get_visualization_batch(data_loader, is_paired, device)
    B, m, _, H, W = sample_batch.shape

    # Run the Stage1 model; outputs: list of length m, each with keys "recon", "R", "L"
    with torch.no_grad():
        outputs = stage1(sample_batch)
    
    # Get individual reconstructions.
    # all_recons shape: [m, B, 3, H, W]
    all_recons = torch.stack([out["recon"] for out in outputs], dim=0)
    all_R = torch.stack([out["R_rgb"] for out in outputs], dim=0)  # shape [m,B,3,H,W]
    all_L = torch.stack([out["L_rgb"] for out in outputs], dim=0)

    # Use source low as the first low-quality image (or average if desired)
    source_low = sample_batch[:, 0, :, :, :]

    num_to_vis = min(num_samples, B)
    # Layout: 1 (source) + m (individual reconstructions) + (1 if paired ground truth)
    num_cols = 1 + 2*m + (1 if is_paired else 0)

    fig, axes = plt.subplots(num_to_vis, num_cols, figsize=(4 * num_cols, 4 * num_to_vis))
    if num_to_vis == 1:
        axes = axes[None, :]
    
    for i in range(num_to_vis):
        col = 0
        # Column 1: Source Low image.
        axes[i, col].imshow(tensor_to_image(source_low[i]))
        axes[i, col].set_title("Source Low")
        axes[i, col].axis("off")
        col += 1
        
        # Columns 2 to (m+1): Individual Reconstructions.
        for j in range(m):
            recon = all_recons[j, i]
            axes[i, col].imshow(tensor_to_image(recon))
            axes[i, col].set_title(f"Reconstruction {j+1}")
            axes[i, col].axis("off")
            col += 1

        # Next m columns: each direct R x L
        for j in range(m):
            R_ij = all_R[j, i]
            L_ij = all_L[j, i]
            RL_ij = R_ij * L_ij
            axes[i, col].imshow(tensor_to_image(RL_ij))
            axes[i, col].set_title(f"R x L {j+1}")
            axes[i, col].axis("off")
            col += 1
            
        # Last column: Ground Truth (if available).
        if is_paired:
            axes[i, col].imshow(tensor_to_image(gt_batch[i]))
            axes[i, col].set_title("Ground Truth")
            axes[i, col].axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_stage1_results(stage1: torch.nn.Module,
                             data_loader: DataLoader,
                             num_samples: int = 8,
                             is_paired: bool = True,
                             approach: str = "avg") -> None:
    """
    Top-level visualization function for Stage1 results.
    
    Supports two approaches:
      - "avg": Averaged Output Approach (average m reconstructions before visualization).
      - "individual": Individual Approach (display each of the m reconstructions separately).
    
    Args:
        stage1 (torch.nn.Module): The Stage1 model.
        data_loader (DataLoader): DataLoader yielding (x, y) in paired mode; otherwise, (x,).
        num_samples (int): Number of samples to visualize.
        is_paired (bool): Indicates if ground truth is provided.
        approach (str): "avg" for averaged output; "individual" for per-image visualization.
    """
    if approach.lower() == "avg":
        visualize_stage1_results_avg(stage1, data_loader, num_samples, is_paired)
    elif approach.lower() == "individual":
        visualize_stage1_results_individual(stage1, data_loader, num_samples, is_paired)
    else:
        raise ValueError("Invalid approach. Use 'avg' or 'individual'.")

