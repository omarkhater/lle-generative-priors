"""
    This module implements Content-Transfer Decomposition Network (CTDN) for low-light image enhancement.

    This module includes a decomposition network and a reconstruction network:
    - DecompositionNet: Decomposes low-light images into reflectance and illumination components.
    - ReconstructionNet: Reconstructs enhanced images using encoded features and skip connections.

    This implementation is based on the paper:
    "Content-Transfer Decomposition Network for Low-Light Image Enhancement" by Yifan Zhang, Yujie Wang, and Zhaoyang Lv.

    Codes are adapted from the original implementation available at:
    https://github.com/JianghaiSCU/LightenDiffusion 

    Some modifications have been made to improve the code structure and readability.

"""

import torch
import torch.nn as nn
from typing import Tuple
from .attention_mechanisms import Cross_Attention, Self_Attention
from .backbones import Res_block, feature_pyramid
from .operators import channel_down, channel_up, upsampling
import numpy as np


class ReconNet(nn.Module):
    """
    Reconstructs an image or provides low-level features for decomposition.
    Combines a feature pyramid with upsampling modules.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pyramid = feature_pyramid(channels)
        self.channel_down = channel_down(channels)
        self.channel_up = channel_up(channels)
        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)
        self.block_up4 = Res_block(channels, channels)
        self.block_up5 = Res_block(channels, channels)
        self.up_sampling2 = upsampling(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract low-level features for decomposition.
        """
        _, _, low_fea_down8 = self.pyramid(x)
        return self.channel_down(low_fea_down8)


    def reconstruct_image(self, x: torch.Tensor, pred_fea: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image from features.
        """
        low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x)
        low_fea_down8 = self.channel_down(low_fea_down8)
        pred_fea = self.channel_up(pred_fea)

        pred_fea_up2 = self.up_sampling0(self.block_up1(self.block_up0(pred_fea) + low_fea_down8))
        pred_fea_up4 = self.up_sampling1(self.block_up3(self.block_up2(pred_fea_up2) + low_fea_down4))
        pred_fea_up8 = self.up_sampling2(self.block_up5(self.block_up4(pred_fea_up4) + low_fea_down2))

        pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))
        return pred_img


class ImageEncoder(nn.Module):
    """
    Encodes an image into low-dimensional features using a feature pyramid and channel reduction.
    Output includes intermediate features for skip connections and compressed features for downstream tasks.
    """
    def __init__(self, channels: int) -> None:
        """
        Args:
            channels (int): Base number of channels used in the pyramid.
        """
        super().__init__()
        self.feature_pyramid = feature_pyramid(channels)
        self.channel_down = channel_down(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - level2, level4, level8: intermediate features for skip connections
                - encoded_features: channel-reduced version of level 3
        """
        level2, level4, level8 = self.feature_pyramid(x)
        encoded = self.channel_down(level8)
        return level2, level4, level8, encoded

class ImageDecoder(nn.Module):
    """
    Decodes low-dimensional features back to a full-resolution RGB image.
    Uses skip connections from encoder stages.
    """
    def __init__(self, channels: int) -> None:
        """
        Args:
            channels (int): Base number of channels used in the pyramid.
        
        """
        super().__init__()
        self.channel_up = channel_up(channels)

        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)

        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)

        self.block_up4 = Res_block(channels, channels)
        self.block_up5 = Res_block(channels, channels)
        self.up_sampling2 = upsampling(channels, channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self,
        encoded: torch.Tensor,
        level2: torch.Tensor,
        level4: torch.Tensor,
        level8: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct an image from encoded features and intermediate pyramid features.

        Args:
            encoded (torch.Tensor): Encoded features from the encoder.
            level2 (torch.Tensor): Intermediate feature from level 2 of the pyramid.
            level4 (torch.Tensor): Intermediate feature from level 4 of the pyramid.
            level8 (torch.Tensor): Intermediate feature from level 8 of the pyramid.

        Returns:
            torch.Tensor: Reconstructed image (B, 3, H, W)
        """
        up = self.channel_up(encoded)
        up2 = self.up_sampling0(self.block_up1(self.block_up0(up) + level8))
        up4 = self.up_sampling1(self.block_up3(self.block_up2(up2) + level4))
        up8 = self.up_sampling2(self.block_up5(self.block_up4(up4) + level2))
        return self.final_conv(up8)

class RetinexDecomposition(nn.Module):
    """
    Retinex-based decomposition module using cross- and self-attention to estimate reflectance and illumination.
    """
    def __init__(
            self, 
            channels: int = 64, 
            num_cross_attention_heads: int = 8,
            num_self_attention_heads: int = 8
            ) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.blocks0 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels)
        )
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1)
        self.blocks1 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels)
        )
        self.cross_attention = Cross_Attention(dim=channels, num_heads=num_cross_attention_heads)
        self.self_attention = Self_Attention(dim=channels, num_heads=num_self_attention_heads, bias=True)
        self.conv0_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
        )
        self.conv1_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose the 3-channel input image x into reflectance (R) and illumination (L).
        The decomposition is performed using a combination of convolutional layers, residual blocks,
        and attention mechanisms. The input image is first processed to estimate the illumination,
        and then the reflectance is computed. The final reflectance and illumination are obtained
        through a series of transformations and attention mechanisms.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                Reflectance (R): Tensor of shape (B, 3, H, W) 
                Illumination (L) Tensor of shape (B, 1, H, W).
        """
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]
        init_reflectance = x / (init_illumination + 1e-6)  # Avoid division by zero.
        Reflectance = self.blocks0(self.conv0(init_reflectance))
        Illumination = self.blocks1(self.conv1(init_illumination))
        Reflectance_final = self.cross_attention(Illumination, Reflectance)
        Illumination_content = self.self_attention(Illumination)
        Reflectance_final = self.conv0_1(Reflectance_final + Illumination_content)
        Illumination_final = self.conv1_1(Illumination - Illumination_content)
        Reflectance = torch.sigmoid(Reflectance_final)
        Illumination = torch.sigmoid(Illumination_final)
        Illumination = torch.cat([Illumination] * 3, dim=1)
        return Reflectance, Illumination


class DecompositionNet(nn.Module):
    """
        Decomposition Network for low-light image enhancement.
        This network decomposes an input image into two components: reflectance and illumination.

        This decomposision is based on the Retinex model, which assumes that the observed image is a product of
        the reflectance and illumination components. The model can be represented as:
        observed_image = reflectance * illumination where * is the Hadamard product.
    """

    def __init__(
            self, 
            channels: int = 64,
            num_cross_attention_heads: int = 8,
            num_self_attention_heads: int = 8
            ) -> None:
        """
        Initialize the DecompositionNet. 

        Args:
            channels (int): Number of channels for the network. Default is 64.
        
        Returns: 
            None

        """
        super().__init__()
        self.encoder = ImageEncoder(channels)
        self.retinex = RetinexDecomposition(
            channels,
            num_cross_attention_heads=num_cross_attention_heads,
            num_self_attention_heads=num_self_attention_heads
            )
        return

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the DecompositionNet.

        It decomposes the input image into reflectance and illumination components by 
        """
        _, _, _, encoded = self.encoder(image)
        return self.retinex(encoded)
    

    def get_decomposed_images(
            self, 
            data_loader: torch.utils.data.DataLoader, 
            device: str ="cuda"
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated reflectance and illumination images from the model.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing the input images. 
                Expected to be 5D (N, 2, C, H, W). where N is the data size,
            device (str): Device to run the model on. Default is "cuda".
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated reflectance and illumination images.
                Each is a numpy array of shape (N, C, H, W).
        
        """
        self.to(device)
        self.eval()
        estimated_reflectance_list = []
        estimated_illumination_list = []

        with torch.no_grad():
            for low_imgs, _ in data_loader:
                low_imgs = low_imgs.to(device)
                if low_imgs.dim() == 5:
                    input_imgs = low_imgs[:, 0, ...]
                else:
                    raise ValueError(f"Unsupported input shape: {low_imgs.shape}")
                
                _, _, _, encoded = self.encoder(input_imgs)
                estimated_reflectance, estimated_illumination = self.retinex(encoded)
                estimated_reflectance_list.append(estimated_reflectance.cpu())
                estimated_illumination_list.append(estimated_illumination.cpu())
        
        estimated_reflectance = torch.cat(estimated_reflectance_list, dim=0)
        estimated_illumination = torch.cat(estimated_illumination_list, dim=0)
        estimated_reflectance = estimated_reflectance.numpy()
        estimated_illumination = estimated_illumination.numpy()
        return estimated_reflectance, estimated_illumination

class ReconstructionNet(nn.Module):

    """
    Full reconstruction pipeline: Encodes image and decodes it using external features (e.g., reflectance).
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = ImageEncoder(channels)
        self.decoder = ImageDecoder(channels)

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ReconstructionNet.

        Args:
            image (torch.Tensor): Input image tensor.
            features (torch.Tensor): Features extracted from the image.
        """
        level2, level4, level8, _ = self.encoder(image)
        return self.decoder(features, level2, level4, level8)

    def get_reconstructed_images(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> np.ndarray:
        """
        Get the reconstructed images from the model.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing the input images.
                Expected to be 5D (N, 2, C, H, W). where N is the data size,
            device (str): Device to run the model on. Default is "cuda".
        Returns:
            np.ndarray: Reconstructed images.
                Each is a numpy array of shape (N, C, H, W).

        """
        self.to(device)
        self.eval()

        reconstructed_images = []
        with torch.no_grad():
            for low_imgs, _ in data_loader:
                low_imgs = low_imgs.to(device)
                if low_imgs.dim() == 5:
                    input_imgs = low_imgs[:, 0, ...]
                    features = low_imgs[:, 1, ...]
                else:
                    raise ValueError(f"Expected shape (B, 2, C, H, W), got {low_imgs.shape}")
                output = self(input_imgs, features)
                reconstructed_images.append(output.cpu())

        return torch.cat(reconstructed_images, dim=0).numpy()
