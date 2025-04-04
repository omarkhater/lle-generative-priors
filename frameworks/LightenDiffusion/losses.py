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
    estimated_reflectance: torch.Tensor,
    estimated_illumination: torch.Tensor,
    target_raw: torch.Tensor
) -> torch.Tensor:
    """
    Compute the reconstruction loss as the L1 loss between the reconstructed image
    (obtained by multiplying estimated reflectance and illumination) and the target raw image.
    
    Args:
        estimated_reflectance (torch.Tensor): Estimated reflectance tensor (B, 3, H, W).
        estimated_illumination (torch.Tensor): Estimated illumination tensor (B, 3, H, W).
        target_raw (torch.Tensor): Target raw/normal-light image tensor (B, 3, H, W).
    
    Returns:
        torch.Tensor: Reconstruction loss as a scalar tensor.
    """
    reconstructed_image: torch.Tensor = estimated_reflectance * estimated_illumination
    return F.l1_loss(reconstructed_image, target_raw)


def reflectance_consistency_loss(
    estimated_reflectance: torch.Tensor,
    target_reflectance: torch.Tensor
) -> torch.Tensor:
    """
    Compute the reflectance consistency loss as the L1 loss between the estimated reflectance
    and the ground-truth reflectance.
    
    Args:
        estimated_reflectance (torch.Tensor): Estimated reflectance tensor (B, 3, H, W).
        target_reflectance (torch.Tensor): Ground-truth reflectance tensor (B, 3, H, W).
        
    Returns:
        torch.Tensor: Reflectance consistency loss as a scalar tensor.
    """
    return F.l1_loss(estimated_reflectance, target_reflectance)


def illumination_smoothness_loss(
    estimated_illumination: torch.Tensor
) -> torch.Tensor:
    """
    Compute the illumination smoothness loss as the mean absolute differences of gradients
    in both x and y directions.
    
    Args:
        estimated_illumination (torch.Tensor): Estimated illumination tensor (B, 3, H, W).
        
    Returns:
        torch.Tensor: Illumination smoothness loss as a scalar tensor.
    """
    def compute_gradient(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x: torch.Tensor = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        grad_y: torch.Tensor = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return grad_x, grad_y
    grad_x, grad_y = compute_gradient(estimated_illumination)
    return grad_x.abs().mean() + grad_y.abs().mean()


def ctdn_loss(
    estimated_reflectance: torch.Tensor,
    estimated_illumination: torch.Tensor,
    target_raw: torch.Tensor,
    target_reflectance: Optional[torch.Tensor] = None,
    weight_rec: float = 1.0,
    weight_ref: float = 0.1,
    weight_lit: float = 0.1
) -> torch.Tensor:
    """
    Compute the total CTDN loss as a weighted sum of reconstruction loss, reflectance consistency loss,
    and illumination smoothness loss.
    
    Args:
        estimated_reflectance (torch.Tensor): Estimated reflectance tensor (B, 3, H, W).
        estimated_illumination (torch.Tensor): Estimated illumination tensor (B, 3, H, W).
        target_raw (torch.Tensor): Target raw/normal-light image tensor (B, 3, H, W).
        target_reflectance (Optional[torch.Tensor]): Ground-truth reflectance tensor (B, 3, H, W), if available.
        weight_rec (float): Weight for the reconstruction loss.
        weight_ref (float): Weight for the reflectance consistency loss.
        weight_lit (float): Weight for the illumination smoothness loss.
    
    Returns:
        torch.Tensor: Total CTDN loss as a scalar tensor.
    
    Raises:
        ValueError: If the spatial dimensions of the tensors do not match.
    """
    if target_raw.shape[2:] != estimated_reflectance.shape[2:]:
        raise ValueError(
            f"Target raw image shape {target_raw.shape[2:]} does not match estimated reflectance shape {estimated_reflectance.shape[2:]}."
        )
    
    if target_reflectance is not None and target_reflectance.shape[2:] != estimated_reflectance.shape[2:]:
        raise ValueError(
            f"Target reflectance shape {target_reflectance.shape[2:]} does not match estimated reflectance shape {estimated_reflectance.shape[2:]}."
        )
    
    loss_reconstruction: torch.Tensor = reconstruction_loss(estimated_reflectance, estimated_illumination, target_raw)
    
    if target_reflectance is not None:
        loss_reflectance: torch.Tensor = reflectance_consistency_loss(estimated_reflectance, target_reflectance)
    else:
        loss_reflectance: torch.Tensor = torch.tensor(0.0, device=target_raw.device, dtype=target_raw.dtype)
    
    loss_illumination: torch.Tensor = illumination_smoothness_loss(estimated_illumination)
    
    total_loss: torch.Tensor = (
        weight_rec * loss_reconstruction +
        weight_ref * loss_reflectance +
        weight_lit * loss_illumination
    )
    return total_loss


def ctdn_loss_wrapper(
    outputs: Tuple[torch.Tensor, torch.Tensor],
    target_raw: torch.Tensor
) -> torch.Tensor:
    """
    Wrapper for the CTDN loss function. Ensures that model outputs are resized to match the target,
    then computes the total CTDN loss.
    
    Args:
        outputs (Tuple[torch.Tensor, torch.Tensor]): A tuple containing (estimated_reflectance, estimated_illumination).
        target_raw (torch.Tensor): Target raw/normal-light image tensor.
    
    Returns:
        torch.Tensor: Total CTDN loss.
    """
    estimated_reflectance, estimated_illumination = outputs

    if estimated_reflectance.shape[2:] != target_raw.shape[2:]:
        estimated_reflectance = F.interpolate(
            estimated_reflectance, size=target_raw.shape[2:], mode='bilinear', align_corners=False
        )
        estimated_illumination = F.interpolate(
            estimated_illumination, size=target_raw.shape[2:], mode='bilinear', align_corners=False
        )
    
    return ctdn_loss(estimated_reflectance, estimated_illumination, target_raw)
