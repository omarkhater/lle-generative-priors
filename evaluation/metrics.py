import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
import pyiqa

def to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        image_tensor: Input tensor of shape [C, H, W] with values in range [0, 1]
        
    Returns:
        NumPy array of shape [H, W, C] with values in range [0, 255] as uint8
    """
    image_tensor = image_tensor.detach().cpu().clamp(0, 1)
    image_np = image_tensor.mul(255).byte().permute(1,2,0).numpy()
    return image_np

def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        gt: Ground truth image as numpy array of shape [H, W, C] in uint8
        pred: Predicted image as numpy array of shape [H, W, C] in uint8
        
    Returns:
        PSNR value as a float
    """
    return peak_signal_noise_ratio(gt, pred, data_range=255)

def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        gt: Ground truth image as numpy array of shape [H, W, C] in uint8
        pred: Predicted image as numpy array of shape [H, W, C] in uint8
        
    Returns:
        SSIM value as a float
    """
    return structural_similarity(gt, pred, data_range=255, channel_axis=-1, win_size=7)

def compute_lpips(gt: torch.Tensor, pred: torch.Tensor, net_type: str = 'alex') -> float:
    """
    Compute Learned Perceptual Image Patch Similarity between two images.
    
    Args:
        gt: Ground truth image as tensor of shape [C, H, W] or [B, C, H, W] with values in [0, 1]
        pred: Predicted image as tensor of shape [C, H, W] or [B, C, H, W] with values in [0, 1]
        net_type: Network backbone to use ('alex', 'vgg', or 'squeeze'), defaults to 'alex'
        
    Returns:
        LPIPS distance as a float
    """
    lpips_model = lpips.LPIPS(net=net_type)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    
    gt_norm = gt * 2 - 1
    pred_norm = pred * 2 - 1
    
    device = next(lpips_model.parameters()).device
    gt_norm = gt_norm.to(device)
    pred_norm = pred_norm.to(device)
    
    with torch.no_grad():
        dist = lpips_model(gt_norm, pred_norm)
    return dist.item()

def compute_niqe(image_tensor: torch.Tensor) -> float:
    """
    Compute Natural Image Quality Evaluator score for an image.
    
    Args:
        image_tensor: Input image as tensor of shape [C, H, W] or [1, C, H, W] with values in [0, 1]
        
    Returns:
        NIQE score as a float (lower is better)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    niqe_metric = pyiqa.create_metric('niqe_matlab')
    return niqe_metric(image_tensor).item()

def compute_pi(lpips_val: float, niqe_val: float) -> float:
    """
    Compute Perceptual Index as the average of LPIPS and NIQE scores.
    
    Args:
        lpips_val: LPIPS distance value
        niqe_val: NIQE score value
        
    Returns:
        Perceptual Index (PI) as a float
    """
    return 0.5 * (lpips_val + niqe_val)
