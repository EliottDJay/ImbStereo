import torch
import torch.nn as nn

from .attention_volume import AttentionCostVolume  # used in CoEx
from .volume_helper import ConcatVolume, SparseConcatVolume, CorVolume, GwcVolume, DiffVolume, DiffVolumeV2  # used in CoEx


from utils.logger import Logger as Log
from utils.check import isNum, is_list, get_attr_from


class VolumePyramid(nn.Module):
    def __init__(self, model_cfg, vol_cfg=None, disp=None, feature_channels=None, **kwargs):
        super(VolumePyramid, self).__init__()

        if disp is None:
            self.max_disp = model_cfg.get('max_disparity', None)
        else:
            assert isNum(disp)
            self.max_disp = disp
        if vol_cfg is None:
            volume_cfg = model_cfg['volume']
        else:
            volume_cfg = vol_cfg
        
        volume_unit = volume_cfg['volume_unit']
        Volume_Structure = globals()[volume_unit]

        self.volume_constructor = Volume_Structure(model_cfg, vol_cfg=None, disp=None, feature_channels=None, **kwargs)

    def forward(self, fl, fr, **kwargs):
        num_scales = len(fl)
        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s)  
            cost_volume = self.volume_constructor(fl[s], fr[s], max_disp)
            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid
    
    def get_beta_channels(self):
        raise NotImplementedError
