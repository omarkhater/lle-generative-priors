"""
    This module implements losses mentioned in the paper.

    "Content-Transfer Decomposition Network for Low-Light Image Enhancement" by Yifan Zhang, Yujie Wang, and Zhaoyang Lv.

    We could not find any implementation in the original implementation available at:
    https://github.com/JianghaiSCU/LightenDiffusion 

"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def reconstruction_loss(
    reflectances: torch.Tensor,    
    illuminations: torch.Tensor,     
    input_images: torch.Tensor       
) -> torch.Tensor:
    """
    Implements Eq. (8) in the paper for cross reconstruction:
       L_rec = sum_{i=1}^m sum_{j=1}^m || R^i * L^j - I^i ||_1.
    Averages the total over (m*m) for stability.
    
    Args:
        reflectances: Stacked reflectance maps for each of the m low-light frames. Expected shape [B, m, 3, H, W].
        illuminations: Stacked illumination maps for each of the m low-light frames. Expected shape [B, m, 3, H, W].
        input_images: The original low-light images (the 'targets' for each frame). Expected shape [B, m, 3, H, W].
    
    Returns:
        A scalar (mean) reconstruction loss.
    """
    _, m, _, _, _ = reflectances.shape
    loss_sum = 0.0
    count = 0
    for i in range(m):
        R_i = reflectances[:, i]  
        I_i = input_images[:, i]
        for j in range(m):
            L_j = illuminations[:, j]
            recon_ij = R_i * L_j
            loss_sum += F.l1_loss(recon_ij, I_i)
            count += 1
    return loss_sum / count


def reflectance_consistency_loss(
    reflectances: torch.Tensor     
) -> torch.Tensor:
    """
    Implements the reflectance consistency term from Eq. (9):
       || R^1 - R^2 ||_1  (if m=2), or pairwise for m>2.
    We average across all unique pairs (i<j).
    
    Args:
        reflectances: Stacked reflectance maps for each of the m frames. Expected shape [B, m, 3, H, W].
    
    Returns:
        A scalar L1 loss penalizing differences between each pair of reflectances.
    """
    B, m, C, H, W = reflectances.shape
    if m < 2:
        return torch.tensor(0.0, device=reflectances.device, dtype=reflectances.dtype)
    
    loss_sum = 0.0
    pair_count = 0
    for i in range(m):
        for j in range(i+1, m):
            R_i = reflectances[:, i]
            R_j = reflectances[:, j]
            loss_sum += F.l1_loss(R_i, R_j)
            pair_count += 1
    return loss_sum / max(pair_count, 1)



def illumination_smoothness_loss(
    illuminations: torch.Tensor, 
    reflectances: torch.Tensor,     
    lambda_g: float = 0.2
) -> torch.Tensor:
    """
    Implements illumination smoothness from Eq. (9) with exponential weighting:
        || ∇L * exp(-lambda_g * ∇R) ||_1
    
    Args:
        illuminations: The stacked illumination maps for each of m frames. Expected shape [B, m, 3, H, W].
        reflectances: The corresponding reflectance maps for each of m frames. Expected shape [B, m, 3, H, W].
        lambda_g: The exponential weighting parameter.
    
    Returns:
        A scalar (mean) loss over all m frames in the batch.
    """
    B, m, C, H, W = illuminations.shape
    loss_sum = 0.0
    for i in range(m):
        L_i = illuminations[:, i]  # [B,3,H,W]
        R_i = reflectances[:, i]   # [B,3,H,W]
        grad_x_L = L_i[:, :, 1:, :] - L_i[:, :, :-1, :]
        grad_y_L = L_i[:, :, :, 1:] - L_i[:, :, :, :-1]
        grad_x_R = R_i[:, :, 1:, :] - R_i[:, :, :-1, :]
        grad_y_R = R_i[:, :, :, 1:] - R_i[:, :, :, :-1]
        
        # Weight = exp( -lambda_g * |∇R| ) or sometimes directly with ∇R (paper’s eq. has ∇R).
        # We'll interpret eq. (9) as requiring the magnitude of ∇R in the exponent:
        weight_x = torch.exp(-lambda_g * grad_x_R.abs())
        weight_y = torch.exp(-lambda_g * grad_y_R.abs())
        weighted_grad_x = grad_x_L * weight_x
        weighted_grad_y = grad_y_L * weight_y
        smoothness_i = weighted_grad_x.abs().mean() + weighted_grad_y.abs().mean()
        loss_sum += smoothness_i
    
    return loss_sum / m



def ctdn_loss(
    reflectances: torch.Tensor,   
    illuminations: torch.Tensor,    
    low_images: torch.Tensor,       
    weight_rec: float = 1.0,
    weight_ref: float = 0.1,
    weight_ill: float = 0.1,
    lambda_g: float = 0.2
) -> torch.Tensor:
    """
    Stage-1 CTDN loss combining Eqs. (7)–(9) in an unsupervised manner:
      L = L_rec + L_ref + L_ill,
    where:
      L_rec = sum_{i,j} || R^i * L^j - I^i ||,
      L_ref = sum_{i<j} || R^i - R^j ||,
      L_ill = sum_i || ∇L^i * exp(-lambda_g * ∇R^i) ||.

    Args:
        reflectances: Stacked reflectances for each of m frames. [B,m,3,H,W]
        illuminations: Stacked illuminations for each of m frames. [B,m,3,H,W]
        low_images: The original input frames. [B,m,3,H,W]
        weight_rec: Weight for cross-reconstruction term.
        weight_ref: Weight for reflectance consistency term.
        weight_ill: Weight for illumination smoothness term.
        lambda_g: Exponential weighting factor for the smoothness.

    Returns:
        A scalar, the total Stage-1 CTDN loss.
    """
    loss_rec = reconstruction_loss(reflectances, illuminations, low_images)
    loss_ref = reflectance_consistency_loss(reflectances)
    loss_ill = illumination_smoothness_loss(illuminations, reflectances, lambda_g)
    
    total_loss = (weight_rec * loss_rec
                  + weight_ref * loss_ref
                  + weight_ill * loss_ill)
    return total_loss


def noise_loss(
    predicted_noise: torch.Tensor,
    noise_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute the noise loss as the L1 loss between the predicted noise and the target noise.
    
    Args:
        predicted_noise (torch.Tensor): Predicted noise tensor (B, C, H, W).
        noise_target (torch.Tensor): Target noise tensor (B, C, H, W).
    
    Returns:
        torch.Tensor: Noise loss as a scalar tensor.
    """
    return F.l1_loss(predicted_noise, noise_target)

def self_constrained_consistency_loss(
    f_low: torch.Tensor,
    f_low_hat: torch.Tensor
) -> torch.Tensor:
    """
    Compute the self-constrained consistency loss as the L1 loss between the original feature
    and the updated feature.
    
    Args:
        f_low (torch.Tensor): Original feature tensor (B, F, ...).
        f_low_hat (torch.Tensor): Updated feature tensor (B, F, ...).
    
    Returns:
        torch.Tensor: Self-constrained consistency loss as a scalar tensor.
    """
    return F.l1_loss(f_low_hat, f_low)


def stage2_loss_wrapper(
    noise_loss: torch.Tensor,
    self_constrained_consistency_loss: torch.Tensor,
    lambda_scc: float,
) -> torch.Tensor:
    """
    Compute the Stage 2 diffusion loss with self-constrained consistency:
    
        L = L_diff + lambda_scc * L_scc
    
    Args:
        noise_loss (torch.Tensor): The noise loss term, comes from the diffusion process. 
        self_constrained_consistency_loss (torch.Tensor): The self-constrained consistency loss term.
        lambda_scc (float): The weight for the self-constrained consistency loss.
    
    Returns:
        torch.Tensor: A single scalar representing the total Stage 2 loss.
    """
    
    total_loss = noise_loss + lambda_scc * self_constrained_consistency_loss
    return total_loss