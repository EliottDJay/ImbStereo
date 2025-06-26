import torch
import torch.nn as nn

from openstereo.modeling.sub_models.sub_models.spatial_trans import SpatialTransformer_grid

from utils.logger import Logger as Log
from utils.check import isNum, is_list

class ConcatVolume(nn.Module):

    def __init__(self, model_cfg, vol_cfg=None, disp=None, feature_channels=None, **kwargs):
        super(ConcatVolume, self).__init__()
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None) 
        else:
            assert isNum(disp)
            self.max_disp = disp
        if vol_cfg is None:
            volume_cfg = model_cfg['volume']
        else:
            volume_cfg = vol_cfg

        # feature channels and disp channels
        if feature_channels is None:
            self.feature_channels = model_cfg['backbone']['feature_channels']
        if is_list(self.feature_channels):
            self.feature_channels = self.feature_channels[0]
        self.beta_channels = self.feature_channels * 2


    def forward(self, left_feature, right_feature, **kwargs):
        B, C, H, W = left_feature.shape
        # volume = left_feature.new_zeros([B, 2 * C, max_disp, H, W])
        device = left_feature.device
        volume = torch.zeros([B, 2 * C, self.max_disp, H, W]).to(device)
        # FastACV
        for i in range(self.max_disp):
            if i > 0:
                volume[:, :C, i, :, i:] = left_feature[:, :, :, i:]
                volume[:, C:, i, :, i:] = right_feature[:, :, :, :-i]
            elif i == 0:
                volume[:, :C, i, :, :] = left_feature
                volume[:, C:, i, :, :] = right_feature
            else:
                volume[:, :C, i, :, :i] = left_feature[:, :, :, :i]
                volume[:, C:, i, :, :i] = right_feature[:, :, :, abs(i):]
        volume = volume.contiguous()
        return volume
    
    def get_beta_channels(self):
        return self.beta_channels
    

class SparseConcatVolume(nn.Module):
    def __init__(self, model_cfg, vol_cfg=None, disp=None, **kwargs):
        super(SparseConcatVolume, self).__init__()
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None)  
        else:
            assert isNum(disp)
            self.max_disp = disp
        if vol_cfg is None:
            volume_cfg = model_cfg['volume']
        else:
            volume_cfg = vol_cfg
        volume_cfg = model_cfg['volume']

    def forward(self, left_feature, right_feature, coarse_disparity=None, **kwargs):
        assert coarse_disparity is not None
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_feature,
                                                                      right_feature, coarse_disparity)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume
    

class CorVolume(nn.Module):
    """
    Output of CorVolume is 3D
    to obtain 4D CorVolume, use GwcVolume and set group as 1
    """
    def __init__(self, model_cfg, vol_cfg=None, disp=None, **kwargs):
        super(CorVolume, self).__init__()
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None)  
        else:
            assert isNum(disp)
            self.max_disp = disp

        if vol_cfg is None:
            volume_cfg = model_cfg['volume']
        else:
            volume_cfg = vol_cfg
        self.norm = volume_cfg.get('norm', False)

    def _norm_correlation(self, fea1, fea2):
        cvolume = torch.mean(
            ((fea1 / (torch.norm(fea1, 2, 1, True) + 1e-06)) * (fea2 / (torch.norm(fea2, 2, 1, True) + 1e-06))), dim=1,
            keepdim=True)
        return cvolume

    def forward(self, left_feature, right_feature, disp=None, **kwargs):
        b, c, h, w = left_feature.size()

        disp_here = disp if disp is not None else self.max_disp  

        volume = left_feature.new_zeros(b, disp_here, h, w)
        if not self.norm:
            for i in range(disp_here):
                if i > 0:
                    volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                            right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        elif self.norm:
            for i in range(disp_here):
                if i > 0:
                    volume[:, :, i, :, i:] = self._norm_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i])
                else:
                    volume[:, :, i, :, :] = self._norm_correlation(left_feature, right_feature)
        volume = volume.contiguous()
        return volume
    

class GwcVolume(nn.Module):
    def __init__(self, model_cfg, vol_cfg=None, disp=None, **kwargs):
        super(GwcVolume, self).__init__()
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None)  
        else:
            assert isNum(disp)
            self.max_disp = disp
        if vol_cfg is None:
            volume_cfg = model_cfg['volume']
        else:
            volume_cfg = vol_cfg
        self.norm = volume_cfg.get('norm', False)
        self.group = volume_cfg.get('group', 1)
        self.beta_channels = self.group

        Log.info("Using norm {} group cor volume with group: {}".format(self.norm, self.group))

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
        volume = left_feature.new_zeros([b, self.group, self.max_disp, h, w])
        for i in range(self.max_disp):
            if i > 0:
                volume[:, :, i, :, i:] = self._groupwise_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i],
                                                               cpg)
            else:
                volume[:, :, i, :, :] = self._groupwise_correlation(left_feature, right_feature, cpg)
        volume = volume.contiguous()
        return volume
    
    def get_beta_channels(self):
        return self.beta_channels
    

class DiffVolume(nn.Module):
    def __init__(self, model_cfg, disp=None):
        super(DiffVolume, self).__init__()
        # self.group = cfg.get('group')  # group
        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None) 
        else:
            assert isNum(disp)
            self.max_disp = disp

    def forward(self, left_feature, right_feature, **kwargs):
        b, c, h, w = left_feature.size()
        volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

        for i in range(self.max_disp):
            if i > 0:
                volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
            else:
                volume[:, :, i, :, :] = left_feature - right_feature

        volume = volume.contiguous()
        return volume
    

    

