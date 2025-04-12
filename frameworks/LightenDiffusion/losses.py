"""
    This module implements losses mentioned in the paper.

    "Content-Transfer Decomposition Network for Low-Light Image Enhancement" by Yifan Zhang, Yujie Wang, and Zhaoyang Lv.

    We could not find any implementation in the original implementation available at:
    https://github.com/JianghaiSCU/LightenDiffusion 

"""

import torch
import torch.nn.functional as F

def reconstruction_loss(
    reflectances: torch.Tensor,    
    illuminations: torch.Tensor,     
    features: torch.Tensor       
) -> torch.Tensor:
    """
    Implements Eq. (8) in the paper for reconstruction loss:
       L_rec = sum_{i=1}^m sum_{j=1}^m || F^j - R^i * L^j ||_1.
    
    It aims to guarantee the decomposed components can reconstruct the encoded features.
    
    Args:
        reflectances: Stacked reflectance maps for each of the m low-light frames. Expected shape [B, m, 3, H, W].
        illuminations: Stacked illumination maps for each of the m low-light frames. Expected shape [B, m, 3, H, W].
        features: Encoded features for each of the m frames. Expected shape [B, m, C, H, W] or similar.
    
    Returns:
        A scalar (mean) reconstruction loss.
    """
    _, m, _, H, W = reflectances.shape
    loss_sum = 0.0
    for i in range(m):
        R_i = reflectances[:, i]  
        for j in range(m):
            L_j = illuminations[:, j]
            F_i = features[:, j]
            recon_ij = R_i * L_j
            loss_sum += F.l1_loss(F_i, recon_ij)
    return loss_sum


def reflectance_consistency_loss(
    reflectances: torch.Tensor     
) -> torch.Tensor:
    """
    Implements the reflectance consistency loss in a multi–image setting by comparing each
    reflectance to the mean reflectance:
    
        L_ref = 1/m * sum_{i=1}^{m} || R^i - \bar{R} ||_1,
    
    where \(\bar{R}\) is the average reflectance over m images.
    
    Args:
        reflectances: Stacked reflectance maps for each of the m frames. Expected shape [B, m, 3, H, W].
    
    Returns:
        A scalar L1 loss.
    """
    B, m, C, H, W = reflectances.shape
    # Compute the mean reflectance over the m frames
    R_mean = reflectances.mean(dim=1, keepdim=True)  # shape [B, 1, 3, H, W]
    R_mean_expanded = R_mean.expand_as(reflectances)  # shape [B, m, 3, H, W]
    # Compute the average L1 difference from the mean for each image in the batch
    loss = F.l1_loss(reflectances, R_mean_expanded, reduction='mean')
    return loss



def content_loss(
    reconstructions: torch.Tensor,
    input_images: torch.Tensor
) -> torch.Tensor:
    """
    Implements Eq. (7) from the paper, "Content-Transfer Decomposition Network for Low-Light Image Enhancement":
        L_con = sum_{i=1}^m || I_low^i - D( E(I_low^i) ) ||_2

    Args:
        Args:
        reconstructions: Tensor of decoder outputs for each of the m sub-images, shape [B, m, 3, H, W].
        input_images: The original low-light inputs, shape [B, m, 3, H, W].

    Returns:
        torch.Tensor: A scalar tensor representing the average L2 content loss.
    """
    return F.mse_loss(reconstructions, input_images, reduction='mean')


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
    encoded_features: torch.Tensor,    
    weight_rec: float = .1,
    weight_ref: float = 0.1,
    weight_ill: float = 0.01,
    lambda_g: float = 10
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
        encoded_features: The encoded features for each of m frames. [B,m,C,H,W]
        weight_rec: Weight for cross-reconstruction term.
        weight_ref: Weight for reflectance consistency term.
        weight_ill: Weight for illumination smoothness term.
        lambda_g: Exponential weighting factor for the smoothness.

    Returns:
        A scalar, the total Stage-1 CTDN loss.
    """
    loss_rec = reconstruction_loss(reflectances, illuminations, encoded_features)
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