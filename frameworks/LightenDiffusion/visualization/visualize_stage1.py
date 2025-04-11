import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Dict
import numpy as np

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of shape [3, H, W] with values in [0, 1] to a NumPy array
    of shape [H, W, 3] (uint8).
    """
    t = t.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t

def visualize_stage1_results(
    stage1: torch.nn.Module,
    data_loader: DataLoader,
    num_samples: int = 8,
    is_paired: bool = True
) -> None:
    """
    Visualize Stage1 outputs in one figure with multiple rows.
    
    For each sample, the following columns are displayed:
      1. Source low image (if m == 2, use sub-image 0; otherwise, average over m).
      2. Target low image (if m == 2; otherwise, this column is omitted).
      3. Estimated reflectance (aggregated across sub-images).
      4. Estimated illumination (aggregated across sub-images).
      5. Reconstructed image (aggregated across sub-images).
      6. Ground truth (if available).
    
    Args:
        stage1 (torch.nn.Module): The Stage1 model.
        data_loader (DataLoader): Yields batches of images with shape [B, m, 3, H, W]. In paired mode,
                                  the loader should yield a tuple (x, y) with y shaped [B, 3, H, W].
        num_samples (int): Number of samples (rows) to visualize.
        is_paired (bool): Whether ground truth is provided.
    """
    stage1.eval()
    device = next(stage1.parameters()).device

    # Get one batch for visualization.
    # In paired mode, assume data_loader yields (x, y); else just x.
    sample_batch = None
    gt_batch = None
    for batch in data_loader:
        if is_paired:
            x, y = batch
            sample_batch = x.to(device)  # shape: [B, m, 3, H, W]
            gt_batch = y.to(device)       # shape: [B, 3, H, W]
        else:
            sample_batch = batch[0].to(device)  # shape: [B, m, 3, H, W]
        break

    B, m, _, H, W = sample_batch.shape

    # Obtain Stage1 outputs for the entire batch.
    with torch.no_grad():
        outputs = stage1(sample_batch)  # returns a list (length=m) of dictionaries.
    
    # Aggregate across sub-images (i.e. over m) for reflectance, illumination, and reconstructed image.
    R_agg = torch.stack([out["R"] for out in outputs], dim=1).mean(dim=1)      # [B, 3, H, W]
    L_agg = torch.stack([out["L"] for out in outputs], dim=1).mean(dim=1)      # [B, 3, H, W]
    recon_agg = torch.stack([out["recon"] for out in outputs], dim=1).mean(dim=1)  # [B, 3, H, W]
    
    # Determine low images to display.
    if m == 2:
        source_low = sample_batch[:, 0, :, :, :]  # [B, 3, H, W]
        target_low = sample_batch[:, 1, :, :, :]  # [B, 3, H, W]
    else:
        source_low = sample_batch.mean(dim=1)       # aggregated low image
    
    num_to_vis = min(num_samples, B)
    # Determine number of columns:
    # If m==2 and paired: 6 columns; if m==2 and unpaired: 5 columns; 
    # If m!=2 and paired: 5 columns; if m!=2 and unpaired: 4 columns.
    if m == 2:
        num_cols = 6 if is_paired else 5
    else:
        num_cols = 5 if is_paired else 4

    fig, axes = plt.subplots(num_to_vis, num_cols, figsize=(4 * num_cols, 4 * num_to_vis))
    if num_to_vis == 1:
        axes = axes[None, :]  # ensure 2D indexing

    for i in range(num_to_vis):
        col = 0
        # Column 1: Source Low Image.
        axes[i, col].imshow(tensor_to_image(source_low[i]))
        axes[i, col].set_title("Source Low")
        axes[i, col].axis("off")
        col += 1

        # Column 2: If m == 2, display target low image; otherwise skip this column.
        if m == 2:
            axes[i, col].imshow(tensor_to_image(target_low[i]))
            axes[i, col].set_title("Target Low")
            axes[i, col].axis("off")
            col += 1

        # Column 3: Estimated Reflectance.
        axes[i, col].imshow(tensor_to_image(R_agg[i]))
        axes[i, col].set_title("Reflectance")
        axes[i, col].axis("off")
        col += 1

        # Column 4: Estimated Illumination.
        axes[i, col].imshow(tensor_to_image(L_agg[i]))
        axes[i, col].set_title("Illumination")
        axes[i, col].axis("off")
        col += 1

        # Column 5: Reconstructed Image.
        axes[i, col].imshow(tensor_to_image(recon_agg[i]))
        axes[i, col].set_title("Reconstruction")
        axes[i, col].axis("off")
        col += 1

        # Column 6: Ground Truth (if paired).
        if is_paired:
            axes[i, col].imshow(tensor_to_image(gt_batch[i]))
            axes[i, col].set_title("Ground Truth")
            axes[i, col].axis("off")

    plt.tight_layout()
    plt.show()