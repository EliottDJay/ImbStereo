import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger as Log
from utils.check import isNum

class DispRegression(nn.Module):
    def __init__(self, cfg, disp=None):
        super(DispRegression, self).__init__()
        #regression_cfg = cfg['regression']
        # max_disparity = cfg.get('max_disparity', 192)  
        model_set = cfg['net']
        regression_cfg = model_set['regression']

        if disp is None:
            max_disparity = cfg.get('max_disparity', 192)
            self.D = int(max_disparity // 4)
        else:
            assert isNum(disp)
            self.D = disp

        self.top_k = regression_cfg.get('top_k', 2) 
        self.ind_init = False

        # temperature set
        self.temperature_set = regression_cfg.get('temperature_set', None)
        if self.temperature_set is not None:
            self.use_temperature = True
            self.temperature_type = self.temperature_set.get('type', 'const')
            if self.temperature_type == 'const':
                self.temperature = self.temperature_set.get('temperature', 1.0)
            elif self.temperature_type == 'learned':
                self.temperature = nn.Parameter(torch.ones((1, int(self.D), 1, 1)))  # 3D Volume B D H W
    
    def forward(self, cost, **kwargs):

        assert len(cost.shape) == 4

        if hasattr(self, 'use_temperature') and self.use_temperature:
            cost = cost/self.temperature

        disp_values = torch.arange(0, self.D, dtype=cost.dtype, device=cost.device)
        disp_values = disp_values.view(1, self.D, 1, 1)

        disp_prob = F.softmax(cost, 1)

        return torch.sum(disp_prob * disp_values, 1, keepdim=True)
    

class SparseRegression(nn.Module):
    def __init__(self, cfg, max_disparity=192, disp=None):
        super(SparseRegression, self).__init__()
        regression_cfg = cfg['regression']
        # max_disparity = cfg.get('max_disparity', 192)  

        if disp is None:
            max_disparity = cfg.get('max_disparity', 192)
            self.D = int(max_disparity // 4)
        else:
            assert isNum(disp)
            self.D = disp

        self.top_k = regression_cfg.get('top_k', 2)
        self.sparse_token = 24  

        # temperature set
        self.temperature_set = regression_cfg.get('temperature_set', None)
        if self.temperature_set is not None:
            self.use_temperature = True
            self.temperature_type = self.temperature_set.get('type', 'const')
            if self.temperature_type == 'const':
                self.temperature = self.temperature_set.get('temperature', 1.0)
            elif self.temperature_type == 'learned':
                self.temperature = nn.Parameter(torch.ones((1, int(self.sparse_token), 1, 1)))  # 3D Volume B D H W

    def forward(self, cost, disparity_samples, **kwargs):

        assert len(cost.shape) == 4

        if hasattr(self, 'use_temperature') and self.use_temperature:
            cost = cost/self.temperature

        _, ind = cost.sort(1, True)
        pool_ind = ind[:, :self.top_k]
        top_cost = torch.gather(cost, 1, pool_ind)
        prob = F.softmax(top_cost, 1)
        disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
        pred = torch.sum(disparity_samples * prob, dim=1, keepdim=False)

        return pred, prob


class TopKRegression(nn.Module):
    def __init__(self, cfg, disp=None):
        super(TopKRegression, self).__init__()
        # basic configuration
        model_set = cfg['net']
        regression_cfg = model_set['regression']

        if disp is None:
            max_disparity = cfg.get('max_disparity', 192)
            self.D = int(max_disparity // 4)
        else:
            assert isNum(disp)
            self.D = disp

        self.top_k = regression_cfg.get('top_k', 2)

        # temperature set
        self.temperature_set = regression_cfg.get('temperature_set', None)
        if self.temperature_set is not None:
            self.use_temperature = True
            self.temperature_type = self.temperature_set.get('type', 'const')
            if self.temperature_type == 'const':
                self.temperature = self.temperature_set.get('temperature', 1.0)
            elif self.temperature_type == 'learned':
                self.temperature = nn.Parameter(torch.ones((1, int(self.D), 1, 1)))  # 3D Volume B D H W
            Log.info("consider temperature in the regression, the temperature type is %s" % self.temperature_type)

        Log.info("Using the TopK Regression with TopK: %d" % self.top_k)

    def forward(self, cost, **kwargs):
        # b, _, h, w = spg.shape

        assert len(cost.shape) == 4

        if hasattr(self, 'use_temperature') and self.use_temperature:
            cost = cost/self.temperature

        corr, disp = self.topkpool(cost, self.top_k)
        corr = F.softmax(corr, 1)
        disp_4 = torch.sum(corr * disp, 1, keepdim=True)

        return disp_4

        """disp_1 = upfeat(disp_4, spg, 4, 4)
        disp_1 = disp_1.squeeze(1) * 4  # + 1.5

        if self.training:
            disp_4 = disp_4.squeeze(1) * 4  # + 1.5
            return [disp_1, disp_4]
        else:
            return [disp_1]"""

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(1, True)
            pool_ind_ = ind[:, :k]
            b, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 3, h, w))
            pool_ind[:, 1:2] = pool_ind_
            pool_ind[:, 0:1] = torch.max(
                pool_ind_ - 1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, 2:] = torch.min(
                pool_ind_ + 1, self.D * pool_ind_.new_ones(pool_ind_.shape))
            corr = torch.gather(cost, 1, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(1, True)
            pool_ind = ind[:, :k]
            corr = torch.gather(cost, 1, pool_ind)
            disp = pool_ind

        return corr, disp
    
