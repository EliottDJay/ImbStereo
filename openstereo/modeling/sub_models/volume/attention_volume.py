import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, SubModule
from utils.check import isNum        


class AttentionCostVolume(SubModule):
    def __init__(self, model_cfg, disp=None):
        super(AttentionCostVolume, self).__init__()
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', 192)
        else:
            assert isNum(disp)
            self.max_disp = disp
        volume_cfg = model_cfg['volume']
        group = volume_cfg.get('group', 1)
        norm = volume_cfg.get('norm', True)
        weighted = volume_cfg.get('weighted', False)
  
        backbone_cfg = model_cfg['backbone']
        in_chan = backbone_cfg['feature_channels'][1] 
        hidden_chan = volume_cfg.get('hidden_chan', 48)

        self.costVolume = GwcVolume(self.max_disp, group=group, norm=norm, glue=False)
        self.conv = BasicConv(in_chan, hidden_chan, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=1, padding=0, stride=1)
        self.group = group
        self.weighted = weighted
        if weighted:
            self.weights = nn.Parameter(
                torch.randn(hidden_chan).reshape(1, hidden_chan, 1, 1))
        
        self.weight_init() 

        # output_keys
        self.output_keys = ['init_cost']

    def forward(self, fea_L, fea_R):

        b, _, h, w = fea_L.shape
        x = self.conv(fea_L)
        y = self.conv(fea_R)

        x_ = self.desc(x)
        y_ = self.desc(y)

        if self.weighted:
            weights = torch.sigmoid(self.weights)
            x_ = x_ * weights
            y_ = y_ * weights
        cost = self.costVolume(x_ , y_ )

        return cost


class CostVolume(nn.Module):
    def __init__(self, maxdisp, group=1, norm=False, glue=False):
        super(CostVolume, self).__init__()
        self.maxdisp = maxdisp + 1
        self.glue = glue
        self.group = group
        self.norm = norm
        self.unfold = nn.Unfold((1, maxdisp + 1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y, v=None):
        b, c, h, w = x.shape
        x = x / (torch.norm(x, 2, 1, True) + 1e-05)
        y = y / (torch.norm(y, 2, 1, True) + 1e-05)

        unfolded_y = self.unfold(self.left_pad(y)).reshape(
            b, self.group, c // self.group, self.maxdisp, h, w)
        x = x.reshape(b, self.group, c // self.group, 1, h, w)

        cost = (x * unfolded_y).sum(2)
        cost = torch.flip(cost, [2])

        if self.glue:
            cross = self.unfold(self.left_pad(v)).reshape(
                b, c, self.maxdisp, h, w)
            cross = torch.flip(cross, [2])
            return cost, cross
        else:
            return cost
        

class GwcVolume(nn.Module):
    def __init__(self, maxdisp, group=1, norm=False, glue=False):
        super(GwcVolume, self).__init__()
        self.group = group
        self.norm = norm
        self.maxdisp = maxdisp

    def _groupwise_correlation(self, fea1, fea2, cpg):
        # gwc_paras = [B, C, H, W, self.group, channels_per_group]
        b, c, h, w = fea1.shape
        if not self.norm:
            cost = (fea1 * fea2).view([b, self.group, cpg, h, w]).mean(dim=2)
        elif self.norm:
            fea1 = fea1.view([b, self.group, cpg, h, w])
            fea2 = fea2.view([b, self.group, cpg, h, w])
            cost = ((fea1 / (torch.norm(fea1, 2, 2, True) + 1e-05)) * (
                        fea2 / (torch.norm(fea2, 2, 2, True) + 1e-05))).mean(dim=2)
        assert cost.shape == (b, self.group, h, w)  # (B, num_groups, H, W)
        return cost

    def forward(self, left_feature, right_feature, **kwargs):
        b, c, h, w = left_feature.shape
        assert c % self.group == 0
        cpg = c // self.group
        volume = left_feature.new_zeros([b, self.group, self.maxdisp, h, w])
        for i in range(self.maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = self._groupwise_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i],
                                                               cpg)
            else:
                volume[:, :, i, :, :] = self._groupwise_correlation(left_feature, right_feature, cpg)
        volume = volume.contiguous()
        return volume