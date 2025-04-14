import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple

def map_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    Maps an input tensor of shape [B, C, H, W] to an RGB tensor [B, 3, H, W]
    by projecting the C channels onto the top three principal components using PCA.
    For each sample, the PCA projection is then min–max normalized to [0, 1].

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor: Output RGB tensor of shape [B, 3, H, W].
    """
    B, C, H, W = tensor.shape
    output = torch.zeros((B, 3, H, W), device=tensor.device)
    
    for b in range(B):
        # Reshape sample to shape (C, H*W)
        sample = tensor[b]  # shape: [C, H, W]
        X = sample.reshape(C, -1)  # shape: [C, N] where N = H*W
        
        # Center the data
        mean = X.mean(dim=1, keepdim=True)
        X_centered = X - mean

        # Perform singular value decomposition
        # U: (C, C), S: (min(C, N)), Vh: (min(C, N), N)
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Take the top 3 principal components from U
        W_pca = U[:, :3]  # shape: [C, 3]
        
        # Project the centered data onto the top 3 components
        X_proj = torch.matmul(W_pca.T, X_centered)  # shape: [3, N]
        X_proj = X_proj.reshape(3, H, W)
        
        # Normalize the projected output to [0, 1]
        X_min = X_proj.min()
        X_max = X_proj.max()
        X_norm = (X_proj - X_min) / (X_max - X_min + 1e-6)
        
        output[b] = X_norm

    return output

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of shape [3, H, W] with values in [0, 1] 
    to a NumPy array of shape [H, W, 3] (uint8).
    """
    t = t.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
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