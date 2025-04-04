import torch
import numpy as np
import frameworks.LightenDiffusion.utils as utils
import os
import time
import torch.nn.functional as F

class DiffusiveRestoration:
    def __init__(self, 
                 diffusion,
                 device='cuda',
                 resume_path=None,
                 image_folder='results',
                 val_dataset='val'):
        """
        Initialize the diffusive restoration model.
        
        Args:
            diffusion: The diffusion model
            device (str): Device to use ('cuda' or 'cpu')
            resume_path (str): Path to resume from checkpoint
            image_folder (str): Folder to save results
            val_dataset (str): Name of validation dataset
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

    def forward_sample(self, low_light):
        """
        Process a single batch of low-light images
        
        Args:
            low_light (torch.Tensor): Low-light input tensor of shape [B, C, H, W]
        
        Returns:
            pred_img (torch.Tensor): Predicted restoration, shape [B, C, H, W] with values in [0, 1]
        """
        low_light = low_light.to(self.device)
        b, c, h, w = low_light.shape
        
        # Pad to multiple of 64
        img_h_64 = int(64 * np.ceil(h / 64.0))
        img_w_64 = int(64 * np.ceil(w / 64.0))
        x_padded = F.pad(low_light, (0, img_w_64 - w, 0, img_h_64 - h), mode='reflect')
        
        # Forward pass through the diffusion model
        output_dict = self.diffusion.model(x_padded)
        if "pred_x" not in output_dict:
            raise ValueError("Model output does not contain 'pred_x'")
        
        # Crop output back to original size and clamp values
        pred_img = output_dict["pred_x"][:, :, :h, :w]
        pred_img = torch.clamp(pred_img, 0, 1)
        return pred_img

    def restore(self, val_loader):
        """
        Restore images from a validation DataLoader by processing each sample via forward_sample(),
        saving the output images.
        
        Args:
            val_loader: DataLoader for validation data
        """
        output_folder = os.path.join(self.image_folder, self.val_dataset)
        os.makedirs(output_folder, exist_ok=True)
        
        with torch.no_grad():
            for i, (low_light, filename) in enumerate(val_loader):
                t1 = time.time()
                pred_x = self.forward_sample(low_light)
                t2 = time.time()
                
                # Handle filename as string or list of strings
                img_name = filename[0] if isinstance(filename, (list, tuple)) else filename
                utils.logging.save_image(pred_x, os.path.join(output_folder, f"{img_name}"))
                print(f"Processing image {img_name}, time={t2 - t1:.3f}")
