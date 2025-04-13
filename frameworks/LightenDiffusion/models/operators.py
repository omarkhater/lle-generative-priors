import torch
import torch.nn as nn
from typing import List, Optional
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
        super().__init__()
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


def conv_sequence(channels: List[int],
                  activations: Optional[List[Optional[nn.Module]]] = None,
                  final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """
    Helper function to build a sequential convolution block.
    Args:
        channels (List[int]): List of channel sizes; a conv layer is added for each adjacent pair.
        activations (List[Optional[nn.Module]], optional): List of activations to apply after each conv.
            If None, LeakyReLU is used for every conv except the last.
        final_activation (nn.Module, optional): Activation applied after the last conv.
    Returns:
        nn.Sequential: The sequential block.
    """
    layers = []
    num_convs = len(channels) - 1
    for i in range(num_convs):
        layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1))
        if activations is not None and activations[i] is not None:
            layers.append(activations[i])
        elif i < num_convs - 1:
            layers.append(nn.LeakyReLU(inplace=True))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class upsampling(nn.Module):
    """
    Increases the spatial resolution of features using transposed convolution followed by LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

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
    
class ChannelDown(nn.Module):
    """
    Reduces the number of channels via a configurable sequence of convolution operations.
    By default, if the input is expected to have (base * 4) channels and you want to reduce
    to the final number (e.g. 3), then the progression is:
      [in_channels, base*2, base, out_channels]
    where base = in_channels // 4.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channel_progression: Optional[List[int]] = None,
                 final_activation: Optional[nn.Module] = nn.Sigmoid()):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channel_progression (List[int], optional): List (excluding the input) of intermediate channel sizes.
                If not provided, we assume in_channels = base * 4 and set progression to [base*2, base].
            final_activation (nn.Module, optional): Activation function to apply at the end (e.g. Sigmoid).
        """
        super().__init__()
        if channel_progression is None:
            base = in_channels // 4
            # Default progression: in_channels -> base*2 -> base -> out_channels.
            channels = [in_channels, base * 2, base, out_channels]
        else:
            channels = [in_channels] + channel_progression + [out_channels]

        self.down = conv_sequence(channels, final_activation=final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class ChannelUp(nn.Module):
    """
    Expands the channels via a configurable sequence of convolution operations.
    By default, if you want to go from a low-dimensional representation (e.g. 3 channels)
    back to a high-dimensional feature representation (e.g. base * 4 channels), then
    the progression is:
      [in_channels, base, base*2, out_channels]
    where base = out_channels // 4.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channel_progression: Optional[List[int]] = None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channel_progression (List[int], optional): List (excluding the input) of intermediate channel sizes.
                If not provided, we assume out_channels = base * 4 and set progression to [base, base*2].
        """
        super().__init__()
        if channel_progression is None:
            base = out_channels // 4
            # Default progression: in_channels -> base -> base*2 -> out_channels.
            channels = [in_channels, base, base * 2, out_channels]
        else:
            channels = [in_channels] + channel_progression + [out_channels]

        self.up = conv_sequence(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

if __name__ == "__main__":

    in_channels_down = 64
    out_channels_down = 10
    out_channels_up = 128

    down_module = ChannelDown(in_channels=in_channels_down, out_channels=out_channels_down)
    up_module = ChannelUp(in_channels=out_channels_down, out_channels=out_channels_up)

    input_tensor = torch.randn(1, in_channels_down, 32, 32)
    input_tensor_up = torch.randn(1, out_channels_down, 32, 32)
    
    
    output_down = down_module(input_tensor)
    output_up = up_module(input_tensor_up)


    assert output_down.shape == (1, out_channels_down, 32, 32), "Channel down output shape mismatch"
    assert output_up.shape == (1, out_channels_up, 32, 32), "Channel up output shape mismatch" 
    