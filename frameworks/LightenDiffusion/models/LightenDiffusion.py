import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
from .decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from .unet import DiffusionUNet

class Stage1(nn.Module):
    """
    Stage 1 processor that independently processes an arbitrary number of input images
    through an encoder, Retinex decomposer, and decoder. The outputs for each image are
    returned as a dictionary, and the forward method returns a list of these dictionaries.
    """

    def __init__(
        self,
        encoder: ImageEncoder,
        decomposer: RetinexDecomposition,
        decoder: ImageDecoder
    ) -> None:
        """
        Initialize Stage1 with required modules.
        
        Args:
            encoder (ImageEncoder): Module to encode an input image.
            decomposer (RetinexDecomposition): Module to perform Retinex decomposition on encoded features.
            decoder (ImageDecoder): Module to decode processed features back to image space.
        """
        super().__init__()
        self.encoder = encoder
        self.decomposer = decomposer
        self.decoder = decoder

    def forward(
            self, 
            imgs: torch.Tensor
        ) -> List[Dict[str, torch.Tensor]]:
        """
        Process an arbitrary number of images independently and return a list of dictionaries.
        
        Args:
            imgs (torch.Tensor): Image tensor [B, m, 3, H, W].
        
        Returns:
            List[Dict[str, torch.Tensor]]: A list where each element is a dictionary containing:
                - "f": Encoded features.
                - "R": Reflectance extracted from the features.
                - "L": Illumination extracted from the features.
                - "recon": Reconstructed image produced by the decoder.
        """
        if imgs.ndim != 5:
            raise ValueError(f"Expected input shape [B, m, 3, H, W], but got {imgs.shape}")
        _ , num_images, _ , _ , _ = imgs.shape
        outputs = []
        for j in range(num_images):
            img = imgs[:, j, :, :, :]
            lv2, lv4, lv8, f = self.encoder(img)
            R, L = self.decomposer(f)
            recon = self.decoder(R, lv2, lv4, lv8)
            outputs.append({
                "f": f,
                "R": R,
                "L": L,
                "recon": recon,
            })

        return outputs
    

class Stage2(nn.Module):
    """
    Stage2 isolates the diffusion process by accepting a reflectance map (R) and an illumination map (L).
    It computes the conditioning input as R * L and passes it through the diffusion UNet to output a noise estimation.
    """
    
    def __init__(
        self,
        diffusion_unet: DiffusionUNet
    ) -> None:
        """
        Initialize Stage2 with the diffusion UNet.
        
        Args:
            diffusion_unet (DiffusionUNet): The diffusion model used to estimate noise.
        """
        super().__init__()
        self.diffusion_unet = diffusion_unet

    def forward(
        self,
        R: torch.Tensor,
        L: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process the conditioning input through the diffusion UNet to get noise estimation.
        
        Args:
            R (torch.Tensor): Aggregated reflectance map (B, C, H, W).
            L (torch.Tensor): Aggregated illumination map (B, C, H, W).
            timestep (Optional[torch.Tensor]): Timestep tensor for conditioning the diffusion UNet.
                If not provided, a zero tensor is used.
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary with:
                - "noise_est": Noise estimation from the diffusion UNet.
                - "x_condition": The conditioning input computed as R * L.
        """
        if timestep is None:
            timestep = torch.zeros(R.size(0), device=R.device)
            
        x_condition = R * L
        noise_est = self.diffusion_unet(x_condition, timestep)
        
        return {
            "noise_est": noise_est,
            "x_condition": x_condition
        }



class LightenDiffusionPipeline(nn.Module):
    """
    Full pipeline combining Stage1 and Stage2.
    
    This pipeline accepts two branches:
      - imgs_R: images used to generate the aggregated reflectance (R) map.
      - imgs_L: images used to generate the aggregated illumination (L) map.
    
    Stage1 processes each branch independently (returning a list of dictionaries).
    An aggregation step then selects (and aggregates) selected outputs from each branch
    according to configurable indices and aggregation modes.
    The aggregated R and L maps are passed to Stage2 (the diffusion part) to output the noise estimation.
    In addition, the aggregated R map is decoded (using Stage1’s ImageDecoder) with skip features
    extracted from the first image of the R branch.
    """
    def __init__(
        self,
        stage1: Stage1,
        stage2: Stage2,
        r_indices: List[int] = None,
        r_agg_mode: str = "first",   # Options: "first", "average"
        l_indices: List[int] = None,
        l_agg_mode: str = "first"    # Options: "first", "average"
    ) -> None:
        """
        Initialize the pipeline.
        
        Args:
            stage1 (Stage1): An instance of the Stage1 processor.
            stage2 (Stage2): An instance of the Stage2 (diffusion) processor.
            r_indices (List[int]): List of indices from Stage1 outputs (for the R branch) to use for aggregation.
                                   Defaults to [0] if not provided.
            r_agg_mode (str): Aggregation mode for R ("first" selects the first map; "average" computes the elementwise average).
            l_indices (List[int]): List of indices from Stage1 outputs (for the L branch) to use for aggregation.
                                   Defaults to [0] if not provided.
            l_agg_mode (str): Aggregation mode for L.
        """
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.r_indices = r_indices if r_indices is not None else [0]
        self.r_agg_mode = r_agg_mode
        self.l_indices = l_indices if l_indices is not None else [0]
        self.l_agg_mode = l_agg_mode

    def _aggregate(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        key: str,
        indices: List[int],
        mode: str
    ) -> torch.Tensor:
        """
        Aggregate maps (e.g., R or L) from Stage1 outputs.
        
        Args:
            outputs (List[Dict[str, torch.Tensor]]): List of dictionaries from Stage1.
            key (str): The key ("R" or "L") to aggregate.
            indices (List[int]): Which entries from outputs to use.
            mode (str): Aggregation mode ("first" or "average").
        
        Returns:
            torch.Tensor: Aggregated map with shape [B, C, H, W].
        """
        maps = [outputs[i][key] for i in indices if i < len(outputs)]
        if mode == "first":
            return maps[0]
        elif mode == "average":
            return torch.stack(maps, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Aggregation mode {mode} not supported for {key} maps.")

    def forward(
        self,
        imgs_R: torch.Tensor,
        imgs_L: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline:
          1. Process imgs_R and imgs_L independently via Stage1.
          2. Aggregate selected outputs to form the R and L maps.
          3. Pass the aggregated maps to Stage2 (diffusion) to obtain a noise estimation.
          4. Extract skip features and decode the aggregated R map using the decoder.
        
        Args:
            imgs_R (torch.Tensor): Tensor for the R branch, shape [B, m, 3, H, W].
            imgs_L (torch.Tensor): Tensor for the L branch, shape [B, m, 3, H, W].
            timestep (Optional[torch.Tensor]): Timestep tensor for conditioning the diffusion UNet.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "stage1_R": List of dictionaries from Stage1 (R branch).
                - "stage1_L": List of dictionaries from Stage1 (L branch).
                - "stage2": Output dictionary from Stage2 (including noise estimation and x_condition).
                - "decoded": Enhanced image produced by the decoder using the aggregated R map and skip features.
        """

        stage1_out_R = self.stage1(imgs_R)
        stage1_out_L = self.stage1(imgs_L)
        aggregated_R = self._aggregate(stage1_out_R, "R", self.r_indices, self.r_agg_mode)
        aggregated_L = self._aggregate(stage1_out_L, "L", self.l_indices, self.l_agg_mode)
        stage2_out = self.stage2(aggregated_R, aggregated_L, timestep)
        lv2, lv4, lv8, _ = self.stage1.encoder(aggregated_R)
        decoded = self.stage1.decoder(aggregated_R, lv2, lv4, lv8)
        
        return {
            "stage1_R": stage1_out_R,
            "stage1_L": stage1_out_L,
            "stage2": stage2_out,
            "decoded": decoded
        }