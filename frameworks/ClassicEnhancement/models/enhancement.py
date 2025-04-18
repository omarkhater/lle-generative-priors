import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ClassicImageEnhancer(nn.Module):
    """
    Classic image enhancement techniques implemented as a PyTorch module
    to maintain API compatibility with ML models like LightenDiffusionPipeline.
    """
    def __init__(self, 
                 brightness: float = 2.0, 
                 contrast: float = 1.5, 
                 sharpness: float = 1.2,
                 saturation: float = 1.1,
                 gamma: float = 0.8,
                 use_clahe: bool = False,
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: int = 8,
                 aggregation_mode: str = "mean"):
        """
        Initialize the classic image enhancement model.
        
        Args:
            brightness: Brightness enhancement factor (>1 increases brightness)
            contrast: Contrast enhancement factor (>1 increases contrast)
            sharpness: Sharpness enhancement factor (>1 increases sharpness)
            saturation: Color saturation factor (>1 increases saturation)
            gamma: Gamma correction value (<1 brightens shadows, >1 darkens)
            use_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe_clip_limit: Clip limit for CLAHE
            clahe_grid_size: Grid size for CLAHE
            aggregation_mode: How to aggregate multiple exposures ("mean", "max", or "first")
        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.saturation = saturation
        self.gamma = gamma
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.aggregation_mode = aggregation_mode
        
        # For CLAHE if needed
        if self.use_clahe:
            try:
                import cv2
                self.cv2 = cv2
            except ImportError:
                print("OpenCV not found, CLAHE will be disabled")
                self.use_clahe = False

    def _apply_clahe(self, img_tensor):
        """Apply CLAHE using OpenCV"""
        # Convert to numpy for OpenCV processing
        img_np = img_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        
        try:
            # Apply CLAHE to L channel in LAB space
            lab = self.cv2.cvtColor(img_np, self.cv2.COLOR_RGB2LAB)
            l, a, b = self.cv2.split(lab)
            
            clahe = self.cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, 
                tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
            )
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            clahe_lab = self.cv2.merge((cl, a, b))
            clahe_rgb = self.cv2.cvtColor(clahe_lab, self.cv2.COLOR_LAB2RGB)
            
            # Convert back to tensor
            return torch.from_numpy(clahe_rgb).float().div(255).permute(2, 0, 1).to(img_tensor.device)
        except Exception as e:
            print(f"CLAHE error: {e}. Returning original image.")
            return img_tensor
    
    def _enhance_single(self, img_tensor):
        """
        Apply enhancements to a single image tensor [C, H, W]
        
        Args:
            img_tensor: Input tensor of shape [C, H, W] with values in range [0, 1]
            
        Returns:
            Enhanced tensor with the same shape as input
        """
        img_tensor = img_tensor.clamp(0, 1)
        device = img_tensor.device
        
        # Apply gamma correction (directly on tensor for better precision)
        img_tensor = torch.pow(img_tensor, 1/self.gamma)
        
        # Convert to PIL for enhancement operations
        img_pil = TF.to_pil_image(img_tensor.cpu())
        
        # Apply PIL-based enhancements
        img_pil = TF.adjust_brightness(img_pil, self.brightness)
        img_pil = TF.adjust_contrast(img_pil, self.contrast)
        img_pil = TF.adjust_saturation(img_pil, self.saturation)
        img_pil = TF.adjust_sharpness(img_pil, self.sharpness)
        
        # Convert back to tensor
        enhanced = TF.to_tensor(img_pil).to(device)
        
        # Apply CLAHE if enabled
        if self.use_clahe:
            enhanced = self._apply_clahe(enhanced)
            
        return enhanced.clamp(0, 1)
        
    def forward(self, x):
        """
        Apply classic enhancement to input images.
        Accepts either a single tensor or a batch of tensors.
        
        Args:
            x: Input tensor of shape [C, H, W], [B, C, H, W], or [B, m, C, H, W]
            
        Returns:
            Enhanced tensor with the appropriate output shape
        """
        if x.dim() == 3:  # [C, H, W]
            return self._enhance_single(x)
        elif x.dim() == 4:  # [B, C, H, W]
            return torch.stack([self._enhance_single(img) for img in x])
        elif x.dim() == 5:  # [B, m, C, H, W]
            # Process each exposure separately, then aggregate
            B, m, C, H, W = x.shape
            processed_batch = torch.zeros((B, m, C, H, W), device=x.device)
            
            for b in range(B):
                for i in range(m):
                    processed_batch[b, i] = self._enhance_single(x[b, i])
            
            # Use the aggregation method to combine the processed exposures
            return self.aggregate_exposures(processed_batch)
        else:
            raise ValueError(f"Input tensor must be 3D, 4D, or 5D, got {x.dim()}D")

    def aggregate_exposures(self, inputs_batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple exposures for each image in the batch.
        
        Args:
            inputs_batch: Tensor of shape [B, m, C, H, W] where m is the number of exposures
            
        Returns:
            Tensor of shape [B, C, H, W] with aggregated exposures
        """
        if self.aggregation_mode == "mean":
            return inputs_batch.mean(dim=1)
        elif self.aggregation_mode == "max":
            return inputs_batch.max(dim=1)[0]
        else:  # "first"
            return inputs_batch[:, 0]

    def predict(self, input_low: torch.Tensor) -> torch.Tensor:
        """
        Enhance a set of low-light images and return the final enhanced output.
        This method matches the LightenDiffusionPipeline interface.
        
        Args:
            input_low (torch.Tensor): Low-light images, shape [B, m, 3, H, W].
            
        Returns:
            torch.Tensor: Final enhanced image, shape [B, 3, H, W].
        """
        self.eval()
        with torch.no_grad():
            # Apply enhancement to each exposure, then aggregate
            # We can simply call the forward method as it handles 5D tensors properly
            return self.forward(input_low)