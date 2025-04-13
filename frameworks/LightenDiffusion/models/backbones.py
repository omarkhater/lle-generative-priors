import torch
import torch.nn as nn
from typing import Tuple, List


class Res_block(nn.Module):
    """
    Implements a residual block with two convolutional layers.
    Combines the learned transformation with a shortcut path.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
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


class FeaturePyramid(nn.Module):
    """
    Extracts a hierarchy of feature maps at progressively coarser spatial resolutions
    and higher channel depths. It repeatedly extracts and down‐samples feature maps at multiple scales
    using a series of convolutional layers and residual blocks.
    The first two convolutional layers are used to process the input image.
    The subsequent layers are organized into stages, each consisting of a residual block followed 
    by a down-sampling convolution.
    """
    def __init__(
            self, 
            base_channels: int, 
            channel_factors: List[int] = [1, 2, 4], 
            in_channels: int = 3
        ) -> None:
        """
        Args:
            base_channels (int): Base number of channels for the pyramid.
            channel_factors (List[int], optional): Multipliers defining the channels at each stage.
            in_channels (int, optional): Number of input channels (default is 3).
        """
        super().__init__()
        self.initial_convs = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(base_channels, base_channels, kernel_size=5, stride=1, padding=2)
        )
        self.levels = nn.ModuleList()
        current_channels = base_channels
        for factor in channel_factors:
            target_channels = base_channels * factor
            stage = nn.Sequential(
                Res_block(current_channels, target_channels),
                nn.Conv2d(target_channels, target_channels, kernel_size=3, stride=2, padding=1)
            )
            self.levels.append(stage)
            current_channels = target_channels
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            Tuple[torch.Tensor, ...]: A tuple of feature maps from each pyramid stage.
        """
        x = self.initial_convs(x)
        features = []
        for stage in self.levels:
            x = stage(x)
            features.append(x)
        return tuple(features)