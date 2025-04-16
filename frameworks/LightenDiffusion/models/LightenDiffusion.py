import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
from frameworks.LightenDiffusion.training.tqdm_configuration import TqdmManager
from .unet import DiffusionUNet
from ..utils.sampling import data_transform, inverse_data_transform
from .stage1 import Stage1

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

        sampling_bar = TqdmManager(
        total=len(seq), 
        desc="Reverse Sampling", 
        leave=False,
        unit="step"
    )
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
            sampling_bar.update(1)
        sampling_bar.close()
        return x


class LightenDiffusionPipeline(nn.Module):
    """
    LightenDiffusionPipeline orchestrates the two-stage process:
      Stage1 (decomposition) and Stage2 (diffusion).
    It allows flexible usage of multi-image inputs for low-light (and high-light) and
    produces the outputs necessary for training and evaluation.
    """
    def __init__(
            self, 
            stage1: Stage1, 
            stage2: Stage2,
            aggregation_mode: str = "mean"
        ) -> None:
        """
        Args:
            stage1 (Stage): Module for decomposing images.
            stage2 (Stage2): Module for the diffusion process.
            VisualizationMapper (Optional[VisualizationMapper]): Optional visualization mapper for output images.
        """
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.aggregation_mode = aggregation_mode

        for param in self.stage1.parameters():
            param.requires_grad = False
    
    def aggregate_decompositions(self, decomp_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregates a list of Stage1 output dictionaries (for each sub-image) into a single dictionary.
        Keys 'f', 'R', and 'L' are aggregated along the sub-image dimension according to the selected mode.
        The aggregation modes are:
            - "mean": Average across sub-images.
            - "max": Maximum across sub-images.
            - "first": Use the first sub-image as is.
        
        Args:
            decomp_list (List[Dict[str, Tensor]]): List of decompositions for m sub-images.
        
        Returns:
            Dict[str, Tensor]: Aggregated outputs for keys 'f', 'R', and 'L'.
        """
        aggregated = {}
        for key in ['f', 'R', 'L', "features"]:
            if key == "features":
                # Handle tuple of multi-scale skip features
                num_scales = len(decomp_list[0]["features"])
                aggregated_features = []
                for i in range(num_scales):
                    scale_list = [out["features"][i] for out in decomp_list]
                    stacked = torch.stack(scale_list, dim=1)  # Shape: [B, m, ...]
                    if self.aggregation_mode == "mean":
                        aggregated_features.append(stacked.mean(dim=1))
                    elif self.aggregation_mode == "max":
                        aggregated_features.append(stacked.max(dim=1)[0])
                    else:
                        aggregated_features.append(stacked[:, 0])
                aggregated[key] = tuple(aggregated_features)
            else:
                tensor_list = [out[key] for out in decomp_list]
                stacked = torch.stack(tensor_list, dim=1)
                if self.aggregation_mode == "mean":
                    aggregated[key] = stacked.mean(dim=1)
                elif self.aggregation_mode == "max":
                    aggregated[key] = stacked.max(dim=1)[0]
                else:
                    aggregated[key] = stacked[:, 0]
        return aggregated

    def forward(
        self,
        inputs_low: torch.Tensor,
        inputs_high: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Performs a full forward pass of the pipeline.
          - Decomposes low-light & high-light images using Stage1.
          - Aggregates the decompositions for low-light images.
          - Computes the diffusion process using Stage2.


        Args:
            inputs_low (torch.Tensor): Tensor with shape [B, m, 3, H, W] for low-light images.
            inputs_high (torch.Tensor, optional): Tensor with shape [B, m, 3, H, W] for high-light images.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "stage1_low": Decomposition outputs from low-light images.
                - "stage1_high": (Optional) Decomposition outputs from high-light images.
                - "stage2": Diffusion outputs.
        """
        outputs = {}
        with torch.no_grad():
            low_decomp_list = self.stage1(inputs_low)  # List of m dicts (each with keys 'f', 'R', 'L', etc.)
            aggregated_low = self.aggregate_decompositions(low_decomp_list)
        outputs["stage1_low"] = aggregated_low

        if inputs_high is not None:
            inputs_high_expanded = inputs_high.unsqueeze(1) # [B, C, H, W] --> [B, 1, C, H, W]
            with torch.no_grad():
                high_decomp_list = self.stage1(inputs_high_expanded)
                # Since m == 1, take the first (and only) dictionary
                high_decomp = high_decomp_list[0]
            outputs["stage1_high"] = high_decomp

            # Stage2 (Diffusion): use aggregated low reflectance and high illumination.
            diffusion_output = self.stage2(
                R_low=aggregated_low["R"],
                L_high=high_decomp["L"],
                R_condition=aggregated_low["R"],
                L_for_scc=aggregated_low["L"]
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
    
    def predict(
            self, 
            input_low: torch.Tensor,
        ) -> torch.Tensor:
        """
        Enhance a set of low-light images and return the final enhanced output.
        Instead of selecting only the first sub-image, this method aggregates across all
        sub-images (dimension 1) by averaging their latent representations and skip connection features.
        
        Args:
            input_low (torch.Tensor): Low-light images, shape [B, m, 3, H, W].
            
        Returns:
            torch.Tensor: Final enhanced image, shape [B, 3, H, W].
        """
        self.eval()
        with torch.no_grad():
            low_decomp_list = self.stage1(input_low)
            aggregated_low = self.aggregate_decompositions(low_decomp_list)
            f_trans = data_transform(aggregated_low["f"])
            restored_latent = self.stage2.sample_reverse(f_trans)
            restored_latent = inverse_data_transform(restored_latent)
            enhanced_image = self.stage1.decoder(restored_latent, *aggregated_low.get("features"))
            return enhanced_image