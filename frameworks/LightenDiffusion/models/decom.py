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
from typing import Tuple, List
from .attention_mechanisms import Cross_Attention, Self_Attention
from .backbones import Res_block, FeaturePyramid
from .operators import ChannelDown, ChannelUp, upsampling


class FeatureExtractor(nn.Module):
    """
    Extracts a reduced feature representation from an input image using a feature pyramid
    followed by channel reduction.
    """
    def __init__(self, base_channels: int, pyramid_factors: List[int] = [1, 2, 4], out_channels: int = 3) -> None:
        """
        Args:
            base_channels (int): Base number of channels for the pyramid.
            pyramid_factors (List[int], optional): Multipliers for the pyramid levels.
            out_channels (int, optional): Number of output channels after reduction.
        """
        super().__init__()
        self.pyramid = FeaturePyramid(base_channels, channel_factors=pyramid_factors)
        self.channel_down = ChannelDown(in_channels=base_channels * pyramid_factors[-1], out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Reduced feature representation of shape (B, out_channels, H/8, W/8) when using default factors.
        """
        _, _, final_feature = self.pyramid(x)
        return self.channel_down(final_feature)


class ImageReconstructor(nn.Module):
    """
    Reconstructs an image from encoded features using a feature pyramid, channel operations, and upsampling stages.
    """
    def __init__(self, base_channels: int, pyramid_factors: List[int] = [1, 2, 4]) -> None:
        """
        Args:
            base_channels (int): Base number of channels for the pyramid.
            pyramid_factors (List[int], optional): Multipliers for the pyramid levels.
        """
        super().__init__()
        self.pyramid = FeaturePyramid(base_channels, channel_factors=pyramid_factors)
        final_channels = base_channels * pyramid_factors[-1]
        self.channel_down = ChannelDown(in_channels=final_channels, out_channels=3)
        self.channel_up = ChannelUp(in_channels=3, out_channels=final_channels)

        def make_stage(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                Res_block(in_ch, in_ch),
                Res_block(in_ch, in_ch),
                upsampling(in_ch, out_ch)
            )

        self.stage0 = make_stage(final_channels, base_channels * pyramid_factors[-2])
        self.stage1 = make_stage(base_channels * pyramid_factors[-2], base_channels * pyramid_factors[-3])
        self.stage2 = nn.Sequential(
            Res_block(base_channels * pyramid_factors[-3], base_channels * pyramid_factors[-3]),
            Res_block(base_channels * pyramid_factors[-3], base_channels * pyramid_factors[-3])
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * pyramid_factors[-3], base_channels * pyramid_factors[-3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels * pyramid_factors[-3], 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor, pred_fea: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            pred_fea (torch.Tensor): Encoded features of shape (B, 3, H/8, W/8).
            
        Returns:
            torch.Tensor: Reconstructed image of shape (B, 3, H, W).
        """
        low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x)
        low_fea_down8 = self.channel_down(low_fea_down8)
        pred_fea = self.channel_up(pred_fea)

        pred_stage0 = self.stage0(pred_fea) + low_fea_down8
        pred_stage1 = self.stage1(pred_stage0) + low_fea_down4
        pred_stage2 = self.stage2(pred_stage1) + low_fea_down2
        return self.final_conv(pred_stage2)


class ImageEncoder(nn.Module):
    """
    Encodes an image into latent features using a configurable feature pyramid and
    channel reduction. It returns both all multi-scale features (for skip connections)
    and a compressed representation from the last stage.
    """
    def __init__(
        self,
        base_channels: int,
        channel_factors: List[int] = [1, 2, 4],
        in_channels: int = 3,
        encoded_channels: int = 64
    ) -> None:
        """
        Args:
            base_channels (int): Base number of channels.
            channel_factors (List[int]): List of multipliers for pyramid levels.
            in_channels (int, optional): Number of input image channels.
            encoded_channels (int, optional): Number of channels for the compressed representation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_factors = channel_factors
        self.encoded_channels = encoded_channels
        self.pyramid = FeaturePyramid(base_channels, channel_factors, in_channels)
        final_channels = base_channels * channel_factors[-1]
        self.channel_down = ChannelDown(in_channels=final_channels, out_channels=encoded_channels)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input image tensor (B, in_channels, H, W).
            
        Returns:
            Tuple containing:
              - A tuple of feature maps from the pyramid (all stages).
              - A compressed representation from the final (lowest resolution) stage.
        """
        if x.min() < 0 or x.max() > 1:
            x = (x - x.min()) / (x.max() - x.min())
        features = self.pyramid(x)
        encoded = self.channel_down(features[-1])
        return (*features, encoded)

class ImageDecoder(nn.Module):
    """
    Reconstructs a full-resolution image from latent features using iterative upsampling
    and skip connections. The number of upsampling stages is derived from the pyramid levels.
    """
    def __init__(
        self,
        base_channels: int = 64,
        channel_factors: List[int] = [1, 2, 4],
        out_channels: int = 3,
        encoded_channels: int = 64
    ) -> None:
        """
        Args:
            base_channels (int): Base number of channels used in the pyramid.
            channel_factors (List[int]): Multipliers used during encoding.
            out_channels (int, optional): Number of output image channels.
            encoded_channels (int, optional): Number of channels in the compressed latent feature.
        """
        super().__init__()
        self.channel_factors = channel_factors
        final_channels = base_channels * channel_factors[-1]
        self.channel_up = ChannelUp(in_channels=encoded_channels, out_channels=final_channels)
        self.up_blocks = nn.ModuleList()
        L = len(channel_factors)
        for i in range(L - 1, -1, -1):
            in_ch  = base_channels * channel_factors[i]
            # for the last block (i=0), we upsample back to base_channels*channel_factors[0]
            # which is the same as in_ch, so out_ch=in_ch is fine.
            out_ch = base_channels * channel_factors[i - 1] if i > 0 else in_ch
            block = nn.Sequential(
                Res_block(in_ch, in_ch),
                Res_block(in_ch, in_ch),
                upsampling(in_ch, out_ch)
            )
            self.up_blocks.append(block)
            
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        latent: torch.Tensor,
        *features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: latent feature of shape [B, encoded_channels, H/2^(L), W/2^(L)]
            *features: skip‐features in order from lower to higher resolution
                      [(B, base*1, H/2, W/2),
                    (B, base*2, H/4, W/4),
                    (B, base*4, H/8, W/8), …]
            
        Returns:
            torch.Tensor: Reconstructed image tensor (B, out_channels, H, W).
        """
        up = self.channel_up(latent)
        L = len(self.up_blocks)
        if len(features) < L:
            raise ValueError(f"Expected {L} skip features, but got {len(features)}")
        
        skips = list(features[:-1])[::-1]
        for block, skip in zip(self.up_blocks[:2], skips):
            up = block(up) + skip

        up = self.up_blocks[2](up)

        return self.final_conv(up)

class RetinexDecomposition(nn.Module):
    """
    Decomposes a latent feature map into reflectance and illumination components using cross- and self-attention.
    The model is fully parameterized by the channel dimension, making it flexible for any feature space.
    
    Args:
        channels (int): Number of channels in the input feature map.
        num_cross_attention_heads (int): Number of attention heads for the cross-attention module.
        num_self_attention_heads (int): Number of attention heads for the self-attention module.
    """
    def __init__(
        self,
        channels: int = 64,
        num_cross_attention_heads: int = 8,
        num_self_attention_heads: int = 8
    ) -> None:
        super().__init__()
        self.channels = channels
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.blocks0 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels)
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.blocks1 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels)
        )
        self.cross_attention = Cross_Attention(dim=channels, num_heads=num_cross_attention_heads)
        self.self_attention = Self_Attention(dim=channels, num_heads=num_self_attention_heads, bias=True)
        self.conv0_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv1_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.init_illum_transform = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes the input latent feature map into reflectance and illumination.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, channels, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reflectance and illumination maps, each with shape (B, channels, H, W).
        """
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]
        init_illumination = self.init_illum_transform(init_illumination)
        
        reflectance_features = self.blocks0(self.conv0(x))
        illumination_features = self.blocks1(self.conv1(init_illumination))
        
        reflectance_modulated = self.cross_attention(illumination_features, reflectance_features)
        illum_content = self.self_attention(illumination_features)
        
        reflectance_final = self.conv0_1(reflectance_modulated + illum_content)
        illumination_final = self.conv1_1(illumination_features - illum_content)
        
        Reflectance = torch.sigmoid(reflectance_final)
        Illumination = torch.sigmoid(illumination_final)
        
        return Reflectance, Illumination
    

if __name__ == "__main__":
    import torch
    from decom import ImageEncoder, ImageDecoder

    # ---- Configuration ----
    B = 1                   # batch size
    H = W = 256             # input spatial size (must be divisible by 2**len(channel_factors))
    base_channels = 64      # same as in your model defaults
    channel_factors = [1, 2, 4]
    in_channels = 3
    encoded_channels = 64
    out_channels = 3

    # ---- Instantiate ----
    encoder = ImageEncoder(
        base_channels=base_channels,
        channel_factors=channel_factors,
        in_channels=in_channels,
        encoded_channels=encoded_channels
    )
    decoder = ImageDecoder(
        base_channels=base_channels,
        channel_factors=channel_factors,
        out_channels=out_channels,
        encoded_channels=encoded_channels
    )

    # Move to CPU (or .cuda() if you prefer)
    device = torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    # ---- Forward pass through encoder ----
    x = torch.randn(B, in_channels, H, W, device=device)
    # Should return a flat tuple: (feat0, feat1, ..., encoded)
    out = encoder(x)
    assert isinstance(out, tuple), "Encoder must return a tuple"
    *features, encoded = out

    # Check number of pyramid levels
    L = len(channel_factors)
    assert len(features) == L, f"Expected {L} pyramid levels, got {len(features)}"

    # Check each feature map’s shape
    for i, feat in enumerate(features):
        expected_c = base_channels * channel_factors[i]
        expected_h = H // (2 ** (i+1))
        expected_w = W // (2 ** (i+1))
        actual = feat.shape
        assert actual == (B, expected_c, expected_h, expected_w), (
            f"features[{i}] shape {actual} != expected {(B, expected_c, expected_h, expected_w)}"
        )

    # Check encoded shape
    down_factor = 2 ** L
    expected_encoded_shape = (B, encoded_channels, H // down_factor, W // down_factor)
    assert encoded.shape == expected_encoded_shape, (
        f"Encoded shape {encoded.shape} != expected {expected_encoded_shape}"
    )

    print("✅ Encoder shapes OK")

    # ---- Forward pass through decoder ----
    # Pass latent first, then each skip feature in the same order
    recon = decoder(encoded, *features)

    # Check reconstruction shape
    expected_recon_shape = (B, out_channels, H, W)
    assert recon.shape == expected_recon_shape, (
        f"Reconstruction shape {recon.shape} != expected {expected_recon_shape}"
    )

    print("✅ Decoder shapes OK")
    print("All shape checks passed.")
