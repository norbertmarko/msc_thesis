import torch
import torch.nn as nn
import torch.nn.functional as F

#from conv_layer import ConvLayer

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


class StemBlock(nn.Module):

    """
    input: (H x W x 3) --> (H/2 x W/2 x C)
    left:  (H/2 x W/2 x C) --> (H/2 x W/2 x C/2) --> (H/4 x W/4 x C)
    right: (H/2 x W/2 x C) --> (H/4 x W/4 x C)
    fuse: (H/4 x W/4 x C)
    """

    def __init__(self):
        super(StemBlock, self).__init__()

        self.conv = ConvLayer(3, 16, kernel_size=3, stride=2) 
        self.left = nn.Sequential(
                    ConvLayer(16, 8, kernel_size=1, padding=0),
                    ConvLayer(8, 16, kernel_size=3, stride=2)
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvLayer(32, 16, kernel_size=3, stride=1)

    def forward(self, t):

        t = self.conv(t)
        t_left = self.left(t)
        t_right = self.right(t)
        t = torch.cat([t_left, t_right], dim=1)
        t = self.fuse(t)

        return t

class GELayerS1(nn.Module):

    """
    Gather-and-Expansion Layer (stride=1 variation)
    """

    def __init__(self, in_channels, out_channels, exp_ratio=6):
        super(GELayerS1, self).__init__()

        exp_channels = in_channels * exp_ratio

        self.conv = ConvLayer(
            in_channels, in_channels, kernel_size=3, stride=1
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, exp_channels, kernel_size=3, stride=1,
                padding=1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(exp_channels)
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                exp_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.pointwise_conv[1].last_bn = True

    def forward(self, t):
        
        x = self.conv(t)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = torch.add(x, t)
        x = F.relu(x, inplace=True)

        return x


class GELayerS2(nn.Module):

    """
    Gather-and-Expansion Layer (stride=2 variation)
    """

    def __init__(self, in_channels, out_channels, exp_ratio=6):
        super(GELayerS2, self).__init__()

        exp_channels = in_channels * exp_ratio

        self.conv = ConvLayer(
            in_channels, in_channels, kernel_size=3, stride=1
        )

        self.depthwise_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, exp_channels, kernel_size=3, stride=2,
                padding=1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(exp_channels)
        )

        self.depthwise_conv2 = nn.Sequential(
            nn.Conv2d(
                exp_channels, exp_channels, kernel_size=3, stride=1,
                padding=1, groups=exp_channels, bias=False
            ),
            nn.BatchNorm2d(exp_channels)
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                exp_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.pointwise_conv[1].last_bn = True        

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2,
                padding=1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, t):
        
        x = self.conv(t)
        x = self.depthwise_conv1(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv(x)

        shortcut = self.shortcut(t)

        x = torch.add(x, shortcut)
        x = F.relu(x, inplace=True)

        return x


class CEBlock(nn.Module):

    """
    Context Embedding Block
    """
    
    def __init__(self):
        super(CEBlock, self).__init__()

        self.bn = nn.BatchNorm2d(128)

        self.pointwise_conv = ConvLayer(
            128, 128, kernel_size=1, stride=1, padding=0
        )

        self.conv = nn.Conv2d(128, 128, 3, stride=1, padding=1)

    def forward(self, t):
        
        x = torch.mean(t, dim=(2, 3), keepdim=True)
        x = self.bn(x)
        x = self.pointwise_conv(x)
        x = torch.add(x, t)
        x = self.conv(x)

        return x


class SemanticBranch(nn.Module):
    
    """
    Backbone of the model

    S: Stage of the Branch
    """

    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class DetailBranch(nn.Module):
    
    """
    S: Stage of the branch
    """

    def __init__(self):
        super(DetailBranch, self).__init__()

        self.S1 = nn.Sequential(
            ConvLayer(3, 64, kernel_size=3, stride=2),
            ConvLayer(64, 64, kernel_size=3, stride=1)
        )

        self.S2 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=2),
            ConvLayer(64, 64, kernel_size=3, stride=1),
            ConvLayer(64, 64, kernel_size=3, stride=1)
        )

        self.S3 = nn.Sequential(
            ConvLayer(64, 128, kernel_size=3, stride=2),
            ConvLayer(128, 128, kernel_size=3, stride=1),
            ConvLayer(128, 128, kernel_size=3, stride=1)
        )
    
    def forward(self, t):
        
        t = self.S1(t)
        t = self.S2(t)
        t = self.S3(t)

        return t