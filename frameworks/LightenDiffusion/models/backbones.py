import torch
import torch.nn as nn
from typing import Tuple


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