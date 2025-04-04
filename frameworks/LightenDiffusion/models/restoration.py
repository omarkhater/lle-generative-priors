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

    def forward_sample(self, x):
        """
        Process a single batch: extract low image, pad it, concatenate with itself,
        run the forward pass, crop the prediction, and clamp to [0,1].

        Args:
            x (torch.Tensor): Input tensor of shape [B, 6, H, W] (even if unpaired, low image is in the first 3 channels).
        
        Returns:
            pred_img (torch.Tensor): Predicted restoration, shape [B, C, H, W] with values in [0, 1].
        """
        # Extract low image (first 3 channels)
        x_cond = x[:, :3, :, :].to(self.diffusion.device)
        b, c, h, w = x_cond.shape
        
        # Pad to multiple of 64
        img_h_64 = int(64 * np.ceil(h / 64.0))
        img_w_64 = int(64 * np.ceil(w / 64.0))
        x_padded = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), mode='reflect')
        
        # Concatenate low image with itself to form 6-channel input
        model_input = torch.cat((x_padded, x_padded), dim=1)
        
        # Forward pass through the diffusion model
        output_dict = self.diffusion.model(model_input)
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
            for i, (x, y) in enumerate(val_loader):
                t1 = time.time()
                pred_x = self.forward_sample(x)
                t2 = time.time()
                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{y[0]}"))
                print(f"Processing image {y[0]}, time={t2 - t1:.3f}")
