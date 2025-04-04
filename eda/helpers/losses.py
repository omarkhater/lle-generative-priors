import torch
import torch.nn.functional as F
from typing import Optional, Tuple

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
    Compute the Content-Transfer Decomposition Network (CTDN) loss per the description in section 3.4.

        @InProceedings{Jiang_2024_ECCV,
        author    = {Jiang, Hai and Luo, Ao and Liu, Xiaohong and Han, Songchen and Liu, Shuaicheng},
        title     = {LightenDiffusion: Unsupervised Low-Light Image Enhancement with Latent-Retinex Diffusion Models},
        booktitle = {European Conference on Computer Vision},
        year      = {2024},
        pages     = {}
        }

    Parameters:
        estimated_reflectance: Estimated reflectance tensor (B, 3, H, W).
        estimated_illumination: Estimated illumination tensor (B, 3, H, W).
        target_raw: Target raw/normal-light image tensor (B, 3, H, W).
        target_reflectance: Optional ground-truth reflectance tensor (B, 3, H, W).
        weight_rec: Weight for the reconstruction loss.
        weight_ref: Weight for the reflectance consistency loss.
        weight_lit: Weight for the illumination smoothness loss.

    Returns:
        A scalar tensor representing the total loss.
    """

    if target_raw.shape[2:] != estimated_reflectance.shape[2:]:
        raise ValueError(
            f"Target raw image shape {target_raw.shape[2:]} does not match estimated reflectance shape {estimated_reflectance.shape[2:]}."
        )
    
    if target_reflectance is not None and target_reflectance.shape[2:] != estimated_reflectance.shape[2:]:
        raise ValueError(
            f"Target reflectance shape {target_reflectance.shape[2:]} does not match estimated reflectance shape {estimated_reflectance.shape[2:]}."
        )

    reconstructed_image: torch.Tensor = estimated_reflectance * estimated_illumination
    loss_reconstruction: torch.Tensor = F.l1_loss(reconstructed_image, target_raw)

    if target_reflectance is not None:
        loss_reflectance: torch.Tensor = F.l1_loss(estimated_reflectance, target_reflectance)
    else:
        loss_reflectance: torch.Tensor = torch.tensor(0.0, device=target_raw.device, dtype=target_raw.dtype)

    def compute_gradient(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x: torch.Tensor = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        grad_y: torch.Tensor = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return grad_x, grad_y

    grad_x, grad_y = compute_gradient(estimated_illumination)
    loss_illumination: torch.Tensor = grad_x.abs().mean() + grad_y.abs().mean()

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
    Wrapper for ctdn_loss function to handle model outputs.
    
    Args:
        outputs: A tuple of (estimated_reflectance, estimated_illumination).
        target_raw: Target raw/normal-light image tensor.
        
    Returns:
        A scalar tensor representing the total loss.
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
