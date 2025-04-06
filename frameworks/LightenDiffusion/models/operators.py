import torch
import torch.nn as nn

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