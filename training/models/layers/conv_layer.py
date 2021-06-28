import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    
    """
    Basic Convolutional Layer. The activation function
    and the Batch Normalization layers are swapped compared
    to the original paper based on empirical evidence.

    1. Convolution
    2. ReLU
    3. Batch Normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, t):

        t = self.conv(t)
        t = F.relu(t, inplace=True)
        t = self.bn(t)

        return t