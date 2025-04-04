import torch
import numpy as np
import frameworks.LightenDiffusion.utils as utils
import os
import time
import torch.nn.functional as F

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
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
        low_light = low_light.to(self.diffusion.device)
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
        """
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)
        with torch.no_grad():
            for i, (low_light, filename) in enumerate(val_loader):
                t1 = time.time()
                pred_x = self.forward_sample(low_light)
                t2 = time.time()
                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{filename[0]}"))
                print(f"Processing image {filename[0]}, time={t2 - t1:.3f}")
