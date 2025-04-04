"""
This module implements image restoration using a diffusion model.
The DiffusiveRestoration class provides methods to restore low-light images
by leveraging a pre-trained diffusion model.

This implementation is based on the paper:
    "Content-Transfer Decomposition Network for Low-Light Image Enhancement" by Yifan Zhang, Yujie Wang, and Zhaoyang Lv.

Codes are adapted from the original implementation available at:
https://github.com/JianghaiSCU/LightenDiffusion 

Some modifications have been made to improve the code structure and readability.
"""

import torch
import numpy as np
import os
import time
import torch.nn.functional as F
from typing import Any, Optional
import frameworks.LightenDiffusion.utils as utils

class DiffusiveRestoration:
    """
    Performs image restoration using a diffusion model.
    """
    def __init__(self, 
                 diffusion: Any,
                 device: str = 'cuda',
                 resume_path: Optional[str] = None,
                 image_folder: str = 'results',
                 val_dataset: str = 'val') -> None:
        """
        Initialize the diffusive restoration model.
        
        Args:
            diffusion (Any): The diffusion model.
            device (str): Device to use ('cuda' or 'cpu').
            resume_path (Optional[str]): Path to resume from checkpoint.
            image_folder (str): Folder to save results.
            val_dataset (str): Name of the validation dataset.
        """
        super(DiffusiveRestoration, self).__init__()
        self.device = device
        self.resume_path = resume_path
        self.image_folder = image_folder
        self.val_dataset = val_dataset
        self.diffusion = diffusion

        if resume_path and os.path.isfile(resume_path):
            self.diffusion.load_ddm_ckpt(resume_path, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def forward_sample(self, low_light: torch.Tensor) -> torch.Tensor:
        """
        Process a single batch of low-light images and return the restored images.
        
        Args:
            low_light (torch.Tensor): Input tensor of shape [B, C, H, W].
        
        Returns:
            torch.Tensor: Restored image tensor of shape [B, C, H, W] with values in [0, 1].
        """
        low_light = low_light.to(self.device)
        _, _, h, w = low_light.shape
        img_h_64 = int(64 * np.ceil(h / 64.0))
        img_w_64 = int(64 * np.ceil(w / 64.0))
        x_padded = F.pad(low_light, (0, img_w_64 - w, 0, img_h_64 - h), mode='reflect')
        
        output_dict = self.diffusion.model(x_padded)
        if "pred_x" not in output_dict:
            raise ValueError("Model output does not contain 'pred_x'")
        
        pred_img = output_dict["pred_x"][:, :, :h, :w]
        pred_img = torch.clamp(pred_img, 0, 1)
        return pred_img

    def restore(self, val_loader: Any) -> None:
        """
        Restore images from a validation DataLoader by processing each sample and saving the results.
        
        Args:
            val_loader (Any): DataLoader providing validation samples.
        """
        output_folder = os.path.join(self.image_folder, self.val_dataset)
        os.makedirs(output_folder, exist_ok=True)
        
        with torch.no_grad():
            for i, (low_light, filename) in enumerate(val_loader):
                t1 = time.time()
                pred_x = self.forward_sample(low_light)
                t2 = time.time()
                
                img_name = filename[0] if isinstance(filename, (list, tuple)) else filename
                utils.logging.save_image(pred_x, os.path.join(output_folder, f"{img_name}"))
                print(f"Processing image {img_name}, time={t2 - t1:.3f}")
