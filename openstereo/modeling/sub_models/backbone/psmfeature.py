import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, BasicBlock

# utils
from utils.logger import Logger as Log


class PSMFeature(nn.Module):
    def __init__(self, cfg):
        super(PSMFeature, self).__init__()
        # about all config
        model_cfg = cfg['net']
        backbone_cfg = model_cfg['backbone']  # basic backbone config
        
        self.batch_norm = backbone_cfg.get('batch_norm', True)
        self.inchannels = backbone_cfg.get('channels', 32)  
        self.feature_channels = [32]  # 1/4

        Log.info("Using the PSMNet-type backbone, with Spatial Pyramid Pooling Module")

        self.firstconv = nn.Sequential(
            BasicConv(3, 32, bn=self.batch_norm, kernel_size=3, stride=2, padding=1, dilation=1),
            BasicConv(32, 32, bn=self.batch_norm, kernel_size=3, stride=1, padding=1, dilation=1),
            BasicConv(32, 32, bn=self.batch_norm, kernel_size=3, stride=1, padding=1, dilation=1),
        )
        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(self.batch_norm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(self.batch_norm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(self.batch_norm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(self.batch_norm, BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            BasicConv(128, 32, bn=self.batch_norm, kernel_size=1, stride=1, padding=0, dilation=1),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            BasicConv(128, 32, bn=self.batch_norm, kernel_size=1, stride=1, padding=0, dilation=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            BasicConv(128, 32, bn=self.batch_norm, kernel_size=1, stride=1, padding=0, dilation=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            BasicConv(128, 32, bn=self.batch_norm, kernel_size=1, stride=1, padding=0, dilation=1),
        )
        self.lastconv = nn.Sequential(
            BasicConv(320, 128, bn=self.batch_norm, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = BasicConv(self.in_planes, out_planes * block.expansion, bn=True, relu=False,
                                   kernel_size=1, stride=stride, padding=0, dilation=1)

        layers = []
        layers.append(
            block(self.in_planes, out_planes, stride, downsample, padding, dilation, batch_norm)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.in_planes, out_planes, 1, None, padding, dilation, batch_norm)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        output_feature = self.lastconv(output_feature)
        # [B, 32, H//4, W//4]

        return output_feature
    
    def channels_recal(self):
        return self.inchannels