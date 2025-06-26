import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, SubModule
from openstereo.modeling.sub_models.sub_models.attention import ChannelAtt, attention_block3d

from utils.logger import Logger as Log
from utils.check import isNum


class Hourglass(SubModule):
    # basic Hourglass module
    def __init__(self, in_channels):
        super(Hourglass, self).__init__()
        self.conv1 = BasicConv(in_channels, in_channels * 2, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv(in_channels * 2, in_channels * 4, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv(in_channels * 4, in_channels * 4, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)

        self.conv5 = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True,
                               kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv6 = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True, relu=True,
                               kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.redir1 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=False, relu=False,
                               kernel_size=1, stride=1, padding=0)
        self.redir2 = BasicConv(in_channels, in_channels, deconv=False, is_3d=True, bn=False, relu=False,
                                kernel_size=1, stride=1, padding=0)

        self.acf = nn.LeakyReLU()
        # SubModule initialization
        # self.weight_init()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4) + self.redir1(conv2)
        conv5 = self.acf(conv5)
        conv6 = self.conv6(conv5) + self.redir2(x)
        conv6 = self.acf(conv6)
        return conv6


class Att_HourglassFit(SubModule):
    def __init__(self, cfg, model_cfg, agg_cfg=None, disp=None, **kwargs):
        # self, in_channels, img_channels, final_one=False, is_cat=True
        """
        used in FastACV series
        :param in_channels:
        :param img_channels:
        :param final_one: if is true, channel of final output is set to 1
        """
        super(Att_HourglassFit, self).__init__()
        # configuration
        if disp is None:
            max_disparity = cfg.get('max_disparity', 192)
            self.D = int(max_disparity // 4)
        else:
            assert isNum(disp)
            self.D = disp

        if agg_cfg is None:
            aggregation_cfg = model_cfg['aggregation']
        else:
            aggregation_cfg = agg_cfg
        
        # basic cfg
        self.is_cat = aggregation_cfg.get('is_cat', True)
        self.final_one = aggregation_cfg.get('final_one', False)
        self.init_beta = aggregation_cfg.get('init_group', 8)
        in_channels = aggregation_cfg.get('init_group', 8)

        # img channels
        img_channels_index = aggregation_cfg.get('img_index', [2, 5])
        backbone_cfg = model_cfg['backbone']
        fea_channels = backbone_cfg['feature_channels']  # [1/2, 1/4, 1/8, 1/16, 1/32]
        img_channels = fea_channels[img_channels_index[0]:img_channels_index[1]]

        Log.info("Using Att_HourglassFit module: final_one is set {}, in_channels is set {}, and the img channels are {}"
                 .format(self.final_one, in_channels, img_channels))

        self.hrglass_length = len(img_channels)
        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_agg = nn.ModuleList()
        self.channelAttDown = nn.ModuleList()
        self.channelAttUp = nn.ModuleList()

        for i in range(self.hrglass_length):
            k1 = i*2 if i > 0 else 1
            k2 = (i+1)*2
            self.conv_down.append(
                nn.Sequential(
                    BasicConv(in_channels * k1, in_channels * k2, is_3d=True, bn=True, relu=True, kernel_size=3,
                              padding=1, stride=2, dilation=1),
                    BasicConv(in_channels * k2, in_channels * k2, is_3d=True, bn=True, relu=True, kernel_size=3,
                              padding=1, stride=1, dilation=1)
            ))
            self.channelAttDown.append(ChannelAtt(in_channels*k2, img_channels[i]))

            if i == 0 and self.final_one:
                out_channels = 1
                bn_set, relu_set = False, False
            else:
                out_channels = in_channels * k1
                bn_set, relu_set = True, True
            self.conv_up.append(
                BasicConv(in_channels * k2, out_channels, deconv=True, is_3d=True, bn=bn_set,
                          relu=relu_set, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
            )
            if i < self.hrglass_length-1:
                mul = 2 if self.is_cat else 1
                self.conv_agg.append(
                    nn.Sequential(
                        BasicConv(in_channels * k2 * mul, in_channels * k2, is_3d=True, kernel_size=1, padding=0, stride=1),
                        BasicConv(in_channels * k2, in_channels * k2, is_3d=True, kernel_size=3, padding=1, stride=1),
                        BasicConv(in_channels * k2, in_channels * k2, is_3d=True, kernel_size=3, padding=1, stride=1),
                    )
                )
                self.channelAttUp.append((ChannelAtt(in_channels*k2, img_channels[i])))

        # self.weight_init()

    def forward(self, volume, fea):
        assert len(fea) == self.hrglass_length

        volume_list = [volume]
        volume_down = volume

        for i in range(self.hrglass_length):
            conv_down = self.conv_down[i](volume_down)
            conv_down = self.channelAttDown[i](conv_down, fea[i])
            volume_list.append(conv_down)
            volume_down = conv_down

        volume_up = volume_list[-1]

        for i in range(self.hrglass_length-1):
            conv_up = self.conv_up[-i-1](volume_up)
            if self.is_cat:
                conv_up = torch.cat((conv_up, volume_list[-i-2]), dim=1)
            else:
                conv_up = conv_up + volume_list[-i-2]
            conv_up = self.conv_agg[-i-1](conv_up)
            volume_up = self.channelAttUp[-i-1](conv_up, fea[-i-2])

        volume_final = self.conv_up[0](volume_up)
        return volume_final
    

class CrossHourglass(nn.Module):
    """
    An implementation of hourglass module proposed in PSMNet.
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True
    Inputs:
        x, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        presqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        postsqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
    Outputs:
        out, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        post, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout

    """
    def __init__(self, in_channels, batch_norm=True):
        super(CrossHourglass, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = BasicConv(in_channels, in_channels * 2, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=self.batch_norm, relu=False,
                               kernel_size=3, stride=1, padding=1)

        self.conv3 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                               kernel_size=3, stride=1, padding=1)

        self.conv5 = BasicConv(in_channels * 2, in_channels * 2, deconv=True, is_3d=True, bn=self.batch_norm, relu=False,
                               kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv6 = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=self.batch_norm, relu=False,
                               kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.acf = nn.LeakyReLU()

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, D, H, W], out: [B, 2C, D, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = self.acf(pre + postsqu)
        else:
            pre = self.acf(pre)

        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/4, W/4]
        out = self.conv4(out)
        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/2, W/2]
        if presqu is not None:
            post = self.acf(self.conv5(out) + presqu)
        else:
            post = self.acf(self.conv5(out) + pre)

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(post)

        return out, pre, post
    

# used in ACV
class Hourglass_with3DAtt(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(Hourglass_with3DAtt, self).__init__()
        use_bn = use_bn
        # downsampling
        self.conv1 = nn.Sequential(BasicConv(in_channels, 2 * in_channels, bn=use_bn, is_3d=True, kernel_size=3, stride=2, padding=1, dilation=1),
                                   BasicConv(in_channels * 2, 2 * in_channels, bn=use_bn, is_3d=True, kernel_size=3, stride=1, padding=1, dilation=1),)

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, 4 * in_channels, bn=use_bn, is_3d=True, kernel_size=3, stride=2, padding=1,
                      dilation=1),
            BasicConv(in_channels * 4, 4 * in_channels, bn=use_bn, is_3d=True, kernel_size=3, stride=1, padding=1,
                      dilation=1), )

        self.attention3d = attention_block3d(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        # upsampling
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=False,
                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)))
        self.conv4 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True, relu=False,
                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)))

        self.redir1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=1, stride=1, padding=0)
        self.redir2 = BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=False, kernel_size=1, stride=1, padding=0)
        self.acf = nn.LeakyReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2 = self.attention3d(conv2)
        conv3 = self.acf(self.conv3(conv2) + self.redir2(conv1))
        conv4 = self.acf(self.conv4(conv3) + self.redir1(x))
        return conv4