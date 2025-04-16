import torch
import torch.nn as nn
from typing import Dict, List, Any
from tqdm import tqdm
from .decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from torch.utils.data import DataLoader

class Stage1(nn.Module):
    """
    Processes a batch of m images per sample ([B, m, 3, H, W]) by encoding,
    decomposing, and decoding each image. 
    """
    def __init__(
        self,
        encoder: ImageEncoder,
        decomposer: RetinexDecomposition,
        decoder: ImageDecoder
    ) -> None:
        """
        Args:
            encoder (ImageEncoder): Module to encode an input image.
            decomposer (RetinexDecomposition): Module to perform Retinex decomposition.
            decoder (ImageDecoder): Module to decode features back to image space.
        """
        super().__init__()
        self.encoder = encoder
        self.decomposer = decomposer
        self.decoder = decoder

    def _process_sample(self, img: torch.Tensor) -> Dict[str, Any]:
        """
        Processes a single image through encoding, decomposition, and decoding.
        Returns a dictionary of internal outputs.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys:
                - "f": Compressed latent features from the encoder.
                - "R": Decomposed reflectance.
                - "L": Decomposed illumination.
                - "recon": Reconstructed image.
                - "features": Tuple of multi-scale skip features from the encoder.
        """
        features = self.encoder(img)
        f = features[-1] # Last element is the encoded feature
        skip_features = features[:-1] # from low to high resolution
        R, L = self.decomposer(f)
        recon = self.decoder(R, *skip_features)
        
        
        return {
            "f": f,
            "R": R,
            "L": L,
            "recon": recon,
            "features": skip_features
        }

    def forward(self, imgs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Processes a batch containing m images per sample.
        
        Args:
            imgs (torch.Tensor): Image tensor of shape [B, m, 3, H, W].
            
        Returns:
            List[Dict[str, torch.Tensor]]: A list (length m) of dictionaries corresponding to each
            image in the batch.
        """
        if imgs.ndim != 5:
            raise ValueError(f"Expected input shape [B, m, 3, H, W], but got {imgs.shape}")
        _, num_images, _, _, _ = imgs.shape
        outputs = []
        for j in range(num_images):
            img = imgs[:, j, :, :, :]
            outputs.append(self._process_sample(img))
        return outputs
    
    def get_decomposed_images(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Processes the data loader to return aggregated reflectance and illumination components.
        
        Args:
            data_loader (DataLoader): Yields images of shape [B, m, 3, H, W].
            
        Returns:
            Dict[str, torch.Tensor]:
                - "R": Tensor [Total_N, m, C, H/scale, W/scale] containing reflectance.
                - "L": Tensor [Total_N, m, C, H/scale, W/scale] containing illumination.
        """
        R_list, L_list = [], []
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch, _ in tqdm(data_loader, desc="Decomposing images"):
                batch = batch.to(device)
                outputs = self._process_batch(batch)
                R_batch = torch.stack([out["R"] for out in outputs], dim=1)
                L_batch = torch.stack([out["L"] for out in outputs], dim=1)
                R_list.append(R_batch)
                L_list.append(L_batch)
        R_all = torch.cat(R_list, dim=0)
        L_all = torch.cat(L_list, dim=0)
        return {"R": R_all, "L": L_all}

    def get_reconstructed_images(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Processes the data loader to return reconstructed images.
        
        Args:
            data_loader (DataLoader): Yields images of shape [B, m, 3, H, W].
            
        Returns:
            torch.Tensor: Reconstructed images of shape [Total_N, m, 3, H, W].
        """
        recon_list = []
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch, _ in tqdm(data_loader, desc="Reconstructing images"):
                batch = batch.to(device)
                outputs = self._process_batch(batch)
                recon_batch = torch.stack([out["recon"] for out in outputs], dim=1)
                recon_list.append(recon_batch)
        return torch.cat(recon_list, dim=0)
    
