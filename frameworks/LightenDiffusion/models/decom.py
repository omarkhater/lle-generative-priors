"""
    This module implements Content-Transfer Decomposition Network (CTDN) for low-light image enhancement.
    The CTDN consists of a feature pyramid for multi-scale feature extraction, a Retinex decomposition
    module for separating reflectance and illumination, and a reconstruction network for image restoration.
    The model is designed to work with low-light images and can be used for both decomposition and reconstruction tasks.

    This implementation is based on the paper:
    "Content-Transfer Decomposition Network for Low-Light Image Enhancement" by Yifan Zhang, Yujie Wang, and Zhaoyang Lv.

    Codes are adapted from the original implementation available at:
    https://github.com/JianghaiSCU/LightenDiffusion 

    Some modifications have been made to improve the code structure and readability.

"""

import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Union
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DepthConv(nn.Module):
    """
    Performs depthwise separable convolution.
    Applies a depthwise 3x3 convolution followed by a 1x1 pointwise convolution.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the DepthConv module.

        Parameters:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            None
        """
        super(DepthConv, self).__init__()
        self.depth_conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels
        )
        self.point_conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for depthwise separable convolution.
        
        Parameters:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            A tensor of shape (B, out_channels, H, W) after applying depthwise separable convolution.
        """

        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class Res_block(nn.Module):
    """
    Implements a residual block with two convolutional layers.
    Combines the learned transformation with a shortcut path.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Res_block, self).__init__()
        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    """
    Increases the spatial resolution of features using transposed convolution followed by LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upsampling module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Upsampled tensor.
        """
        out = self.relu(self.conv(x))
        return out


class channel_down(nn.Module):
    """
    Reduces the number of channels through sequential convolution operations and activation.
    """
    def __init__(self, channels: int) -> None:
        super(channel_down, self).__init__()

        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel reduction.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Tensor with reduced channels.
        """
        out = torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))

        return out


class channel_up(nn.Module):
    """
    Expands the feature channels via successive convolution layers and activation functions.
    """
    def __init__(self, channels: int) -> None:
        super(channel_up, self).__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel expansion.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Tensor with expanded channels.
        """
        out = self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))

        return out


class feature_pyramid(nn.Module):
    """
    Extracts multi-scale feature maps using a sequence of convolutions and residual blocks.
    """
    def __init__(self, channels: int) -> None:
        super(feature_pyramid, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
                                   nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2))

        self.block0 = Res_block(channels, channels)

        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = Res_block(channels, channels * 2)

        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = Res_block(channels * 2, channels * 4)

        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature pyramid extraction.
        
        Args:
            x (torch.Tensor): Input image tensor.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Multi-scale feature maps (level0, level1, level2).
        """
        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))

        return level0, level1, level2


class ReconNet(nn.Module):
    """
    Reconstructs an image or provides low-level features for decomposition.
    Combines a feature pyramid with upsampling modules.
    """
    def __init__(self, channels: int) -> None:
        super(ReconNet, self).__init__()
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

    def forward(self, x: torch.Tensor, pred_fea: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the reconstruction network.
        
        In decomposition mode, returns low-level features for decomposition.
        In reconstruction mode, returns the reconstructed image.
        
        Args:
            x (torch.Tensor): Input image tensor.
            pred_fea (Optional[torch.Tensor]): Feature tensor for conditioning reconstruction.
        
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - A tuple of (reflectance, illumination) in decomposition mode.
                - A reconstructed image tensor in reconstruction mode.
        """
        if pred_fea is None:
            # Process x through the pyramid and channel-down layers.
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x)
            low_fea_down8 = self.channel_down(low_fea_down8)
            # Return the same features for both branches.
            return low_fea_down8, low_fea_down8
        else:
            # Reconstruction branch.
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x)
            low_fea_down8 = self.channel_down(low_fea_down8)
            pred_fea = self.channel_up(pred_fea)
            pred_fea_up2 = self.up_sampling0(self.block_up1(self.block_up0(pred_fea) + low_fea_down8))
            pred_fea_up4 = self.up_sampling1(self.block_up3(self.block_up2(pred_fea_up2) + low_fea_down4))
            pred_fea_up8 = self.up_sampling2(self.block_up5(self.block_up4(pred_fea_up4) + low_fea_down2))
            pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))
            return pred_img


class Self_Attention(nn.Module):
    """
    Implements self-attention on spatial features using convolutions and normalization.
    """
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Cross_Attention(nn.Module):
    """
    Computes cross-attention between a query and a context using depthwise separable convolutions.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.) -> None:
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = DepthConv(in_channels=dim, out_channels=dim)
        self.key = DepthConv(in_channels=dim, out_channels=dim)
        self.value = DepthConv(in_channels=dim, out_channels=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape and permute tensor for attention score computation.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Transposed tensor.
        """
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention mechanism.
        
        Args:
            hidden_states (torch.Tensor): Query tensor.
            ctx (torch.Tensor): Context tensor for key and value.
        
        Returns:
            torch.Tensor: Output tensor after cross-attention.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Retinex_decom(nn.Module):
    """
    Decomposes an image into its reflectance and illumination components.
    Uses residual blocks and both self and cross-attention mechanisms.
    """
    def __init__(self, channels: int) -> None:
        super(Retinex_decom, self).__init__()
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
        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)
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
        """
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]
        init_reflectance = x / (init_illumination + 1e-6)  # Avoid division by zero.
        Reflectance = self.blocks0(self.conv0(init_reflectance))
        Illumination = self.blocks1(self.conv1(init_illumination))
        Reflectance_final = self.cross_attention(Illumination, Reflectance)
        Illumination_content = self.self_attention(Illumination)
        Reflectance_final = self.conv0_1(Reflectance_final + Illumination_content)
        Illumination_final = self.conv1_1(Illumination - Illumination_content)
        R = torch.sigmoid(Reflectance_final)
        L = torch.sigmoid(Illumination_final)
        L = torch.cat([L] * 3, dim=1)
        return R, L

class CTDN(nn.Module):
    """
    Constructs the Content-Transfer Decomposition Network (CTDN) for low-light image enhancement.
    Integrates decomposition (Retinex_decom) and reconstruction (ReconNet) modules.
    """
    def __init__(self, channels: int = 64) -> None:
        super(CTDN, self).__init__()
        self.ReconNet = ReconNet(channels)
        self.retinex = Retinex_decom(channels)

    def forward(self, images: torch.Tensor, pred_fea: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the CTDN model.
        
        In decomposition mode, returns the estimated reflectance and illumination.
        In reconstruction mode, returns the reconstructed image.
        
        Args:
            images (torch.Tensor): Input low-light image tensor.
            pred_fea (Optional[torch.Tensor]): Optional feature tensor to trigger reconstruction.
        
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - A tuple (estimated reflectance, estimated illumination) in decomposition mode.
                - A reconstructed image tensor in reconstruction mode.
        """
        if pred_fea is None:
            low_features, _ = self.ReconNet(images, pred_fea=None)
            estimated_reflectance, estimated_illumination = self.retinex(low_features)
            return estimated_reflectance, estimated_illumination
        else:
            pred_img = self.ReconNet(images, pred_fea=pred_fea)
            return pred_img

