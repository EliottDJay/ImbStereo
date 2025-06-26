import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, SubModule
from openstereo.modeling.sub_models.sub_models.attention import ChannelAtt

from utils.logger import Logger as Log
from utils.check import isNum

class GCE(SubModule):
    def __init__(self, cfg, model_cfg, agg_cfg=None, disp=None, **kwargs):
        super(GCE, self).__init__()
        # configuration
        if disp is None:
            max_disparity = cfg.get('max_disparity', 192)
            self.D = int(max_disparity // 4)
        else:
            assert isNum(disp)
            self.D = disp

        aggregation_cfg = model_cfg['aggregation']
        volume_cfg = model_cfg['volume']
        self.gce = aggregation_cfg.get('gce', True)

        volume_group = volume_cfg.get('group', 1)
        init_group = aggregation_cfg.get('init_group', 8)
        s_disp = 2
        block_n = aggregation_cfg.get('block_num', [2, 2, 2])
        channels = aggregation_cfg.get('volume_channels', [16, 32, 48])

        backbone_cfg = model_cfg['backbone']
        fea_channels = backbone_cfg['feature_channels']

        final_cost_channel = 1

        Log.info("Using GCE as the aggregation module")
        Log.info("When using GCE u will get a 5 dim cost volume with disp dim equal to {}, "
                 "so if you directly put regression on the cost from GCE you need squeeze the cost".format(final_cost_channel))
        # structure

        self.conv_stem = BasicConv(volume_group, init_group, is_3d=True, kernel_size=3, stride=1, padding=1)

        if self.gce:
            self.channelAttStem = ChannelAtt(init_group, fea_channels[1])
            self.channelAtt = nn.ModuleList()
            self.channelAttDown = nn.ModuleList()

        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_skip = nn.ModuleList()
        self.conv_agg = nn.ModuleList()

        volume_channels = [init_group] + (channels)
        input_channels = volume_channels[0]
        for i in range(3):
            conv = nn.ModuleList()
            for n in range(block_n[i]):
                stride = (s_disp, 2, 2) if n == 0 else 1
                dilation, kernel_size, padding, bn, relu = 1, 3, 1, True, True
                conv.append(BasicConv(input_channels, volume_channels[i + 1], is_3d=True, bn=bn,
                        relu=relu, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
                input_channels = volume_channels[i + 1]
            self.conv_down.append(nn.Sequential(*conv))

            if self.gce:
                self.channelAttDown.append(ChannelAtt(volume_channels[i + 1], fea_channels[i+2], self.D // (2 ** (i + 1))))

            if i == 0:
                out_chan, bn, relu = final_cost_channel, True, True  # False, False
            else:
                out_chan, bn, relu = volume_channels[i], True, True

            # if i != 0:
            self.conv_up.append(BasicConv(volume_channels[i + 1], out_chan, deconv=True, is_3d=True, bn=bn,
                        relu=relu, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))
                
            if i != 0:
                self.conv_agg.append(nn.Sequential(
                    BasicConv(
                    volume_channels[i], volume_channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                    BasicConv(
                    volume_channels[i], volume_channels[i], is_3d=True, kernel_size=3, padding=1, stride=1), ))
                
            if i != 0:
                self.conv_skip.append(BasicConv(2 * volume_channels[i], volume_channels[i], is_3d=True,
                                                kernel_size=1, padding=0, stride=1))
                
            if self.gce and i != 0:
                self.channelAtt.append(ChannelAtt(volume_channels[i], fea_channels[i+1], self.D // (2 ** (i)),))
                
        #self.convup3 = BasicConv(16, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        
        self.weight_init()

    def forward(self, img, cost, **kwargs):

        b, c, h, w = img[0].shape

        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)
        if self.gce:
            cost = self.channelAttStem(cost, img[0])

        cost_feat = [cost]

        cost_up = cost
        for i in range(3):
            cost_ = self.conv_down[i](cost_up)
            if self.gce:
                cost_ = self.channelAttDown[i](cost_, img[i + 1])

            cost_feat.append(cost_)
            cost_up = cost_

        cost_ = cost_feat[-1]

        for i in range(3):
            cost_ = self.conv_up[-i - 1](cost_)
            if cost_.shape != cost_feat[-i - 2].shape:
                target_d, target_h, target_w = cost_feat[-i - 2].shape[-3:]
                cost_ = F.interpolate(
                    cost_,
                    size=(target_d, target_h, target_w),
                    mode='nearest')
            if i == 2:
                break

            cost_ = torch.cat([cost_, cost_feat[-i - 2]], 1)
            cost_ = self.conv_skip[-i-1](cost_)
            cost_ = self.conv_agg[-i-1](cost_)

            if self.gce:
                cost_ = self.channelAtt[-i-1](cost_, img[-i - 2])

        cost = cost_

        return cost