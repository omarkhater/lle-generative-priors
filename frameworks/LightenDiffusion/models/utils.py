from typing import Optional
import torch
import torch.nn as nn

class VisualizationMapper(nn.Module):
    """
    A lightweight convolutional mapping branch to convert features into a displayable RGB image.
    """
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2
        self.mapper = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mapper.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, 3, H, W) after mapping.
        """
        out = self.mapper(x)
        return torch.sigmoid(out)