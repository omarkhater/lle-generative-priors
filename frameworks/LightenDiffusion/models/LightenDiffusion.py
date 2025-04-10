import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
from .decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from .unet import DiffusionUNet

class Stage1(nn.Module):
    """
    Stage1: Processes a batch of m images per sample (shape [B, m, 3, H, W]) using an encoder,
    a Retinex decomposition, and a decoder. Returns a list of dictionaries (one per image) containing:
      - "f": Encoded features.
      - "R": Extracted reflectance.
      - "L": Extracted illumination.
      - "recon": Decoded reconstruction.
      - "lv2", "lv4", "lv8": Intermediate features for skip connections.
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
                - "f": Encoded features (Latent Image Representation).
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
                "lv2": lv2,
                "lv4": lv4,
                "lv8": lv8
            })

        return outputs
    

class Stage2(nn.Module):
    """
    Stage2 handles the diffusion process. It computes a noised sample x_t from a clean input x_0
    (obtained as the product of reflectance and high-light illumination) and predicts the noise using a diffusion UNet.
    It also provides a reverse sampling method.
    """
    def __init__(
            self, 
            diffusion_unet: DiffusionUNet, 
            betas: torch.Tensor = None, 
            num_diffusion_timesteps: int = 1000,
            num_sampling_steps: int = 50,
            gamma: float = 0.2
            ) -> None:
        """
        Args:
            diffusion_unet (DiffusionUNet): The diffusion UNet used for noise prediction.
            betas (torch.Tensor): 1D tensor of beta values with shape [T] for the diffusion process.
            gamma (float): Illumination correction factor.
            num_sampling_steps (int): Number of steps for reverse sampling.
            num_diffusion_timesteps (int): Total number of diffusion timesteps.
        """
        super().__init__()
        self.diffusion_unet = diffusion_unet
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.gamma = gamma
        self.num_sampling_steps = num_sampling_steps
        if betas is None:
            betas = torch.linspace(0.0001, 0.02, steps=num_diffusion_timesteps)
        # Register betas as a buffer so they are moved with the model.
        self.register_buffer("betas", betas)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward diffusion sample: 
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        where alpha_bar_t is the cumulative product of (1 - beta) up to time t.
        
        Args:
            x0 (torch.Tensor): Clean input tensor (e.g., R_low * L_high), shape [B, 3, H, W].
            t (torch.Tensor): Timestep tensor of shape [B].
            noise (torch.Tensor): Noise tensor of the same shape as x0.
        
        Returns:
            torch.Tensor: Noised sample x_t with shape [B, 3, H, W].
        """
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def forward(
        self,
        R_low: torch.Tensor,
        L_high: torch.Tensor,
        R_condition: torch.Tensor,
        L_for_scc: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes the forward diffusion process:
          1. Forms x0 = R_low * L_high.
          2. Samples a random timestep t and noise e.
          3. Computes x_t via forward diffusion.
          4. Predicts the noise using the diffusion UNet.
          5. Computes the reference feature for self-constrained consistency (SCC).

        Args:
            R_low (torch.Tensor): Reflectance from low-light input, shape [B, 3, H, W].
            L_high (torch.Tensor): Illumination from high-light input, shape [B, 3, H, W].
            R_condition (torch.Tensor): Condition for the UNet (e.g., R_low or encoded feature), shape [B, 3, H, W].
            L_for_scc (torch.Tensor): Illumination used for SCC, shape [B, 3, H, W].
            t (torch.Tensor, optional): Timestep tensor of shape [B]. If None, random timesteps are sampled.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "x0": Clean composite image.
                - "x_t": Noised sample.
                - "noise": The noise used in forward diffusion.
                - "noise_pred": Predicted noise by the diffusion UNet.
                - "reference_feature": Reference feature computed as R_low * L^gamma (gamma corrected illumination).
                - "R_low": Input reflectance.
                - "L_high": Input high illumination.
        """
        device = R_low.device
        batch_size = R_low.shape[0]
        x0 = R_low * L_high
        if t is None:
            t = torch.randint(0, self.num_diffusion_timesteps, (batch_size,), device=device).long()
        e = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, e)
        noise_pred = self.diffusion_unet(x_t, t, R_condition)
        reference_feature = R_low * (L_for_scc ** self.gamma)
        return {
            "x0": x0,
            "x_t": x_t,
            "noise": e,
            "noise_pred": noise_pred,
            "reference_feature": reference_feature,
            "R_low": R_low,
            "L_high": L_high
        }

    def sample_reverse(
            self, 
            conditioning: torch.Tensor, 
            eta: float = 0.0
            ) -> torch.Tensor:
        """
        Performs reverse diffusion sampling from x_T ~ N(0, I) to x_0,
        conditioned on the provided conditioning tensor.
        
        Args:
            conditioning (torch.Tensor): Conditioning tensor (e.g., R_low) of shape [B, 3, H, W].
            eta (float): Noise scale parameter.
        
        Returns:
            torch.Tensor: The final restored feature x_0 with shape [B, 3, H, W].
        """
        device = conditioning.device
        B, C, H, W = conditioning.shape
        x = torch.randn(B, C, H, W, device=device)
        skip = self.num_diffusion_timesteps // self.num_sampling_steps
        seq = list(range(0, self.num_diffusion_timesteps, skip))
        seq_next = [-1] + seq[:-1]
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((B,), i, dtype=torch.long, device=device)
            next_t = torch.full((B,), j, dtype=torch.long, device=device)
            at = alphas_cumprod[t].view(B, 1, 1, 1)
            at_next = alphas_cumprod[next_t].view(B, 1, 1, 1) if j >= 0 else torch.tensor(0.0, device=device)
            unet_input = torch.cat([conditioning, x], dim=1)
            et = self.diffusion_unet(unet_input, t)
            x0_t = (x - et * (1.0 - at).sqrt()) / at.sqrt()
            if j >= 0:
                c1 = eta * ((1.0 - at / at_next) * (1.0 - at_next) / (1.0 - at)).sqrt()
                c2 = ((1.0 - at_next) - c1 ** 2).sqrt()
                x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            else:
                x = x0_t
        return x


class LightenDiffusionPipeline(nn.Module):
    """
    LightenDiffusionPipeline orchestrates the two-stage process:
      Stage1 (decomposition) and Stage2 (diffusion).
    It allows flexible usage of multi-image inputs for low-light (and high-light) and
    produces the outputs necessary for training and evaluation.
    """
    def __init__(self, stage1: Stage1, stage2: Stage2) -> None:
        """
        Args:
            stage1 (Stage): Module for decomposing images.
            stage2 (Stage2): Module for the diffusion process.
        """
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(
        self,
        inputs_low: torch.Tensor,
        inputs_high: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Performs a full forward pass of the pipeline.
          - Decomposes low-light images using Stage1.
          - If high-light images are provided, decomposes them as well.
          - Uses one representative sub-image from low and high to run Stage2 diffusion.

        Args:
            inputs_low (torch.Tensor): Tensor with shape [B, m, 3, H, W] for low-light images.
            inputs_high (torch.Tensor, optional): Tensor with shape [B, m, 3, H, W] for high-light images.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "stage1_low": Decomposition outputs from low-light images.
                - "stage1_high": (Optional) Decomposition outputs from high-light images.
                - "stage2": Diffusion outputs.
        """
        outputs: Dict[str, Any] = {}
        low_decomp = self.stage1(inputs_low)
        outputs["stage1_low"] = low_decomp

        if inputs_high is None:
            return outputs

        high_decomp = self.stage1(inputs_high)
        outputs["stage1_high"] = high_decomp
        R_low = low_decomp[0]["R"]
        L_low = low_decomp[0]["L"]
        R_high = high_decomp[0]["R"]
        L_high = high_decomp[0]["L"]

        diffusion_output = self.stage2(
            R_low=R_low,
            L_high=L_high,
            R_condition=R_low,
            L_for_scc=L_low
        )
        outputs["stage2"] = diffusion_output
        return outputs

    def sample_reverse(
        self, 
        conditioning: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Invokes the reverse diffusion sampling given a conditioning tensor.
        
        Args:
            conditioning (torch.Tensor): Conditioning tensor (e.g., R_low) of shape [B, 3, H, W].
            steps (int): Number of reverse diffusion steps.
            eta (float): Noise scale parameter.
        
        Returns:
            torch.Tensor: Restored feature x_0 with shape [B, 3, H, W].
        """
        return self.stage2.sample_reverse(
            conditioning, 
            eta
        )