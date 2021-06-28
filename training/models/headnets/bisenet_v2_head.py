import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys
# sys.path.append('../../..')

from training.models.layers.conv_layer import ConvLayer
from training.models.backbones.bisenet_v2_backbone import SemanticBranch, DetailBranch

class BGALayer(nn.Module):

    """
    Bilateral Guided Aggregation Layer
    """

    def __init__(self):
        super(BGALayer, self).__init__()

        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False
            )
        )

        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(
                kernel_size=3, stride=2, padding=1,
                ceil_mode=False
            )
        )

        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(128),          
        )

        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False
            ),
        )

        self.up1 = nn.Upsample(scale_factor=4)

        self.up2 = nn.Upsample(scale_factor=4)
        
        self.conv = ConvLayer(
            128, 128, kernel_size=3, stride=1,
            padding=1, bias=False
        )

    def forward(self, t_det, t_sem):
        
        det_size = t_det.size()[2:] #??
        
        left1 = self.left1(t_det)
        left2 = self.left2(t_det)

        right1 = self.right1(t_sem)
        right2 = self.right2(t_sem)

        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        output = self.conv(left + right)
        
        return output


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes, upscale_factor=8, aux=True):
        super(SegHead, self).__init__()

        self.conv = ConvLayer(in_channels, mid_channels, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.1)
        self.upscale_factor = upscale_factor

        out_channels = num_classes * upscale_factor * upscale_factor
        
        if aux:
            self.conv_out = nn.Sequential(
                ConvLayer(
                    mid_channels, upscale_factor * upscale_factor, kernel_size=3, stride=1
                ),
                nn.Conv2d(
                    upscale_factor * upscale_factor, out_channels, kernel_size=1, stride=1,
                    padding=0
                ),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.PixelShuffle(upscale_factor)
            )
    
    def forward(self, t):
        
        t = self.conv(t)
        t = self.dropout(t)
        t = self.conv_out(t)

        return t

class BiSeNetV2(nn.Module):

    def __init__(self, num_classes, output_aux=True):
        super(BiSeNetV2, self).__init__()
        self.output_aux = output_aux
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.head = SegHead(128, 1024, num_classes, upscale_factor=8, aux=False)
        if self.output_aux:
            self.aux2 = SegHead(16, 128, num_classes, upscale_factor=4)
            self.aux3 = SegHead(32, 128, num_classes, upscale_factor=8)
            self.aux4 = SegHead(64, 128, num_classes, upscale_factor=16)
            self.aux5_4 = SegHead(128, 128, num_classes, upscale_factor=32)

        self.init_weights()

    def forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head)
        if self.output_aux:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        return logits
        #pred = logits.argmax(dim=1)
        #return pred

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

#TODO: Booster