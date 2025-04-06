import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def get_decomposed_images(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        device: str ="cuda"
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the estimated reflectance and illumination images from the model.

    Args:
        model (torch.nn.Module): The model to use for estimation.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the input images. 
            Expected to be 5D (N, 2, C, H, W). where N is the data size,
        device (str): Device to run the model on. Default is "cuda".
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Estimated reflectance and illumination images.
            Each is a numpy array of shape (N, C, H, W).
    
    """
    model.eval()
    estimated_reflectance_list = []
    estimated_illumination_list = []

    with torch.no_grad():
        for low_imgs, _ in data_loader:
            low_imgs = low_imgs.to(device)
            if low_imgs.dim() == 5:
                input_imgs = low_imgs[:, 0, ...]
            else:
                raise ValueError(f"Unsupported input shape: {low_imgs.shape}")
            
            estimated_reflectance, estimated_illumination = model(input_imgs)
            estimated_reflectance_list.append(estimated_reflectance.cpu())
            estimated_illumination_list.append(estimated_illumination.cpu())
    
    estimated_reflectance = torch.cat(estimated_reflectance_list, dim=0)
    estimated_illumination = torch.cat(estimated_illumination_list, dim=0)
    estimated_reflectance = estimated_reflectance.numpy()
    estimated_illumination = estimated_illumination.numpy()
    return estimated_reflectance, estimated_illumination