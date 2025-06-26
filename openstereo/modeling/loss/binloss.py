import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# import path
from openstereo.dataloader.stereodataset import dataloader_path  # 导入全局变量

# utils
from openstereo.evaluation.experiments import AverageMeter
from utils.logger import logging as Log
from utils.distributed import all_reduce_bysize
from utils.check import tensor2float, isNum, is_list, mkdir
# imbalance tools
from scipy.ndimage import convolve1d
from Imbalance.utils.smoothing_basic import smoothing_setting, get_lds_kernel_window


class BINLoss(nn.Module):
    def __init__(self, cfg, rank=0, device=None):
        super(BINLoss, self).__init__()
        # basic config
        loss_cfg = cfg['loss']
        self.max_disp = cfg['net']['max_disparity']  # 可以设置为None
        self.rank = rank
        self.device = device
        self.loss_weight = loss_cfg['weight']
        self.highest_only = loss_cfg.get("highest_only", False)  # used in AANet
        # binloss （目前）有两种形式 一种是 最基本的MSE(Gaussion) 另一种 我们使用SmoothL1
        self.main_disp_loss = loss_cfg.get("main_disp_loss", 'Smoothl1')
        assert self.main_disp_loss in ['Smoothl1', 'Gaussion']  #后面还会不会增加robust loss呢 TODO: delete

        # 当disp的分辨率和gt分辨率不一样的时候，是对disp进行插值还是对gt进行
        self.gt_downsample = loss_cfg.get("gt_downsample", False)
        self.inter_mode = loss_cfg.get("interpolation_mode", "nearest")  # bilinear for upsampling nearest for downsampling
        Log.info('Setting loss_weight {}, highest_only {}, down sampling gt [{}], interpolation_mode {}'
                 .format(self.loss_weight, self.highest_only, self.gt_downsample, self.inter_mode))
        
        # extra mask setting, 对部分区间的disp进行额外的mask
        self.mask_set = loss_cfg.get("mask", None)
        if self.mask_set is not None:
            assert isinstance(self.mask_set, dict)
            self.mask_floor = self.mask_set.get('mask_floor', None)
            self.mask_ceil = self.mask_set.get('mask_ceil', None)
            assert isinstance(self.mask_floor, list)
            assert isinstance(self.mask_ceil, list)
            assert len(self.mask_floor) == len(self.mask_ceil)
            self.set_mask = False
            if self.mask_floor is not None and self.mask_ceil is not None:
                self.set_mask = True
                if rank == 0:
                    self.mask_rate = AverageMeter()

        # bin loss setting
        binloss_cfg = loss_cfg.get("bin_cfg", None)
        if not binloss_cfg:
            raise ValueError("Please set bin_cfg in loss config! Needed!")
        # label smoothing techniques
        self.using_lds = binloss_cfg.get('lds', True)
        if self.using_lds:
            self.lds_cfg = smoothing_setting(binloss_cfg.get('lds_cfg', None))
        else:
            self.lds_cfg = None
        # bin process
        self.bin_type = binloss_cfg.get("type", None)  # typically sqrt or id(恒等)
        self.binloss_index = binloss_cfg.get('binloss_index', [-1])  # use bin loss for where
        self.binloss_weight = binloss_cfg.get('balbin_loss_weight', [1])  # corresponding bin-loss weight
        if isNum(self.binloss_index):
            self.binloss_index = [self.binloss_index]
        if isNum(self.binloss_weight):
            self.binloss_weight = [self.binloss_weight]
        assert len(self.binloss_index) == len(self.binloss_weight)
        assert is_list(self.binloss_index) and is_list(self.binloss_weight)
        assert self.bin_type is not None and self.bin_type in ['sqrt', 'id']
        
        # highest only: 
        if self.highest_only:
            self.loss_weight = [1.]
            self.binloss_weight = [self.binloss_weight[-1]]  # 仅仅
            self.binloss_index = [self.binloss_index[-1]] # 仅取最后一位index
        self.pred_num = len(self.loss_weight)  # len(preds) == len(self.loss_weight) should!
        self.extra_lossnum = len(self.binloss_index)

        for i in range(self.extra_lossnum):   # 限制index的范围
            if self.binloss_index[i] < 0:
                assert self.binloss_index[i] >= -self.pred_num
                self.binloss_index[i] = self.binloss_index[i] + self.pred_num  # 变成最后一个index
            elif self.binloss_index[i] >= 0:
                assert self.binloss_index[i] < self.pred_num
        # self.main_disp_loss使用Gaussian loss的时候 一般倾向于self.binloss_weight 与 loss weight相等
        if self.main_disp_loss == 'Gaussion':
            Log.warn("Using Gaussion loss here, typically we tend to use equal weight in euqal index."
                     "Now, disp loss weight is set to {}, bin loss and cor index are {} and {} respectly"
                     .format(self.loss_weight, self.binloss_weight, self.binloss_index))

        # gaussion setting
        init_noise_sigma = binloss_cfg.get('init_noise_sigma', 1.0)
        self.noise_sigma = nn.ParameterList()
        for i in range(len(self.binloss_index)):
            self.noise_sigma.append(torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device)))
        # self.noise_sigma['sigma_' + str(i)] = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        Log.info('Extra loss for pred[{}] added, detail weight setting is that {}, '
                 'and sigma is initialized as {}'.format(self.binloss_index, self.binloss_weight, init_noise_sigma))

        # bin statistic and corrosponding process
        bin_statistic = binloss_cfg.get("bin_statistic", None)
        assert bin_statistic is not None and isinstance(bin_statistic, str)
        Log.info(f'Using bin number type: [{self.bin_type}]')
        bin_num = None
        self.bin_size = binloss_cfg.get("bin_size", 1)  # 1 by default
        self.binshift = binloss_cfg.get("binshift", 0)  # 0 by default 0.5 --> center
        bin_num = np.load(bin_statistic)
        # related to process for the tail data
        self.bin_refine_type = binloss_cfg.get("refine_type", None)  # 处理尾部的数据
        if self.bin_refine_type is not None:
            self.bin_refine_cfg = binloss_cfg.get("refine_cfg", None)
        self.bin_weight = self.get_bin_weight(bin_num)  # prob related
        shift = self.binshift * self.bin_size  # 0.5 * bin_size  --> center
        if shift == 0:
            shift = None
        self.origin = self.shift(shift=shift)  # 记得加上偏心
        self._save_bin_info(bin_statistic)
        
        self.disp_losses = AverageMeter()
        self.total_losses = AverageMeter()
        if self.binloss_weight is not None:
            self.balbin_losses = AverageMeter()

    def mask_generator(self, disp, mask_record=False):
        if self.max_disp is None:
            # 当我们不需要指定mask的时候，我们使用None来表示
            basic_mask = disp > 0 
            return basic_mask
        basic_mask = (disp > 0) & (disp < self.max_disp)
        original_mask_num = torch.sum(basic_mask)
        # print("basic mask", torch.sum(basic_mask))
        if hasattr(self, 'set_mask') and self.set_mask:
            # 这里的逻辑好像是仅仅保存在floor以及ceil之间的有用信息，因为后面都是用的加法
            mask_list = []
            for i in range(len(self.mask_floor)):
                floor = self.mask_floor[i]
                ceil = self.mask_ceil[i]
                mask = (disp > floor) & (disp < ceil)
                mask_list.append(mask)
            mask_now = torch.zeros_like(mask_list[0]).bool()
            for i in range(len(mask_list)):
                 mask_now = mask_now + mask_list[i]
            mask_num = torch.sum(mask_now)  #这句话是为什么 我都不记得了
            if mask_record and self.rank == 0:
                # 更新mask rate的信息：
                # self.mask_rate.update(tensor2float(mask_num/original_mask_num))
                pass
        else:
            mask_now = basic_mask
        return mask_now
    
    def prepare_inputs(self, data, device=None, **kwargs):
        disp = data['disp']
        assert disp is not None, "The disp ground truth is necessary for the smooth loss."

        processed_inputs = {
            'disp': disp,
        }

        if device is not None:
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device) if torch.is_tensor(v) else v
            
        return processed_inputs

    def d_range(self, pred, scale, **kwargs):
        # 4D range
        d_range = torch.arange(0, self.max_disp//scale, dtype=pred.dtype, device=pred.device)
        d_range = d_range.view(1, -1, 1, 1)
        d_range = d_range.repeat(pred.size(0), 1, pred.size(-2), pred.size(-1))
        return d_range
    
    def forward(self, preds_dict, loss_inputs, **kwargs):

        for name in preds_dict.keys():
            if ("preds" in name) and ("pyramid" in name):
                preds = preds_dict[name]
        if not isinstance(preds, list):
            preds = [preds]
        if self.highest_only:
            preds = [preds[-1]]  # only the last highest resolution output
        preds_nums = len(preds)
        if preds_nums != self.pred_num:
            raise ValueError("The number of prediction is not equal to the number of loss weight!")
        
        total_loss, disp_loss, bin_loss = 0, 0, 0
        pyramid_loss, bin_pyramid = [], []

        index_count = 0
        balance_term = self.binloss_index[index_count]
        shift = self.origin.clone()  #center
        b_weight = self.bin_weight.log().clone()
        disp_gt = loss_inputs['disp']  # 必须要有这个

        for k in range(self.pred_num):
            pred_disp = preds[k]
            weight = self.loss_weight[k]  # disp loss weight
            gt_now = disp_gt.clone()  # [B, H, W]

            if pred_disp.size(-1) != gt_now.size(-1):
                if not self.gt_downsample:
                    pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                    pred_disp = F.interpolate(pred_disp, size=(gt_now.size(-2), gt_now.size(-1)),
                                              mode=self.inter_mode, align_corners=False) 
                    # 在网络中已经x4了 * (target.size(-1) / pred_disp.size(-1))
                    pred_disp = pred_disp.squeeze(1)  # [B, H, W]
                elif self.gt_downsample:
                    gt_now = gt_now.unsqueeze(1)  # [B, 1, H, W]
                    gt_now = F.interpolate(gt_now, size=(pred_disp.size(-2), pred_disp.size(-1)),
                                              mode='nearest')
                    gt_now = gt_now.squeeze(1)  # [B, H, W]
                    
            mask_now = self.mask_generator(gt_now)
            gt_input = gt_now[mask_now]
            pred_input = pred_disp[mask_now]

            if k != balance_term or self.main_disp_loss == 'Smoothl1':
                curr_loss = F.smooth_l1_loss(gt_input, pred_input, reduction='mean')  # L1
            elif k == balance_term and index_count <= self.extra_lossnum and self.main_disp_loss == 'Gaussion':
                curr_loss = 0.5 * F.mse_loss(pred_input, gt_input, reduction='mean') / self.noise_sigma[index_count]  # Gaussion/MSE

            disp_loss += weight * curr_loss
            reduced_disp_loss = all_reduce_bysize(curr_loss)
            pyramid_loss.append(reduced_disp_loss)

            if k == balance_term and index_count <= self.extra_lossnum:
                bin_term = - 0.5 * (pred_disp.unsqueeze(dim=1) - shift).pow(2) / self.noise_sigma[index_count] + b_weight
                bin_term = torch.logsumexp(bin_term, dim=1, keepdim=False)
                cur_bin_loss = (bin_term * mask_now).sum() / (mask_now.sum() + 1e-6)
                bin_loss += self.binloss_weight[index_count] * cur_bin_loss
                index_count = index_count + 1
                if index_count < self.extra_lossnum:  # 小于等于 self.extra_lossnum - 1
                    balance_term = self.binloss_index[index_count]
                reduced_bin_loss = all_reduce_bysize(cur_bin_loss)
                bin_pyramid.append(reduced_bin_loss)

        total_loss = disp_loss + bin_loss
        d_loss = all_reduce_bysize(disp_loss)
        b_loss = all_reduce_bysize(bin_loss)
        t_loss = all_reduce_bysize(total_loss)

        if self.rank == 0:
            # update需要all reduce更新 但是需要反传的total loss是不可以的
            self.disp_losses.update(tensor2float(d_loss))
            self.total_losses.update(tensor2float(t_loss))
            if self.binloss_weight is not None:
                self.balbin_losses.update(tensor2float(b_loss))

        # loss的数值都是没有经过all reduce的
        # pyramid的数值都是经过all reduce的
        return {
            "total_loss": total_loss,
            "multi_preds_loss": disp_loss,
            "multi_bin_loss": bin_loss,
            "multi_preds_pyramid": pyramid_loss, 
            "multi_bin_pyramid": bin_pyramid,
        }
    
    def loss_stat(self):
        # 这两个函数都是在rank0中执行的
        disp_stat = "Disp Loss: {disp_loss.val:.4f} ({disp_loss.avg:.4f})\t".format(disp_loss=self.disp_losses)
        total_stat = "Total Loss: {total_loss.val:.4f} ({total_loss.avg:.4f})\t".format(total_loss=self.total_losses)
        if self.binloss_weight is not None:
            bin_stat = "Bin Loss: {bin_loss.val:.4f} ({bin_loss.avg:.4f})\t".format(bin_loss=self.balbin_losses)
        else:
            bin_stat = ''
        loss_stat = disp_stat + total_stat + bin_stat
        return loss_stat
    
    def scalar_outputs_update(self, loss_dict):
        scalar_outputs = {"disp_loss": self.disp_losses.avg, "total_loss": self.total_losses.avg}
        self.disp_losses.reset()
        self.total_losses.reset()

        """if hasattr(self, 'set_mask') and self.set_mask:
            scalar_outputs['included_mask'] = self.mask_rate.avg
            self.mask_rate.reset()"""
        
        if self.binloss_weight is not None:
            scalar_outputs['bin_loss'] = self.balbin_losses.avg
            self.balbin_losses.reset()

        if "multi_preds_pyramid" in loss_dict.keys():
            pyramid_loss = loss_dict["multi_preds_pyramid"]
            for s in range(len(pyramid_loss)):
                key_name = 'disp_loss_pyrmd_' + str(len(pyramid_loss) - s - 1)
                scalar_outputs[key_name] = pyramid_loss[s]

        if "multi_bin_pyramid" in loss_dict.keys():
            bin_pyramid = loss_dict["multi_bin_pyramid"]
            for s in range(len(bin_pyramid)):
                # index = self.binloss_index[s]  这个命名顺序 我还真没注意
                key_name = 'bin_loss_pyrmd_' + str(len(bin_pyramid) - s - 1)
                scalar_outputs[key_name] = bin_pyramid[s]

        return scalar_outputs
    
    def shift(self, shift=None):  # can shift to the center but we dont use it
        if shift is None:
            # no shift
            d_range = torch.arange(0, self.max_disp, device=self.device)  # step = self.bin_size
            d_range = d_range.view(1, -1, 1, 1)
        else:
            assert isNum(shift)
            d_range = torch.arange(0, self.max_disp, device=self.device) + shift  # typically bin_size//2
            d_range = d_range.view(1, -1, 1, 1)
        return d_range
    
    def get_bin_weight(self, bin_num):
        bin_num = bin_num + 1
        value = bin_num.copy()
        if self.using_lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_cfg['kernel'], self.lds_cfg['kernel_size'],
                                                      self.lds_cfg['sigma'])
            if self.bin_type == 'sqrt':
                value = np.sqrt(value)
            smoothed_value = convolve1d(np.asarray(value), weights=lds_kernel_window, mode='reflect')
            bucket_weights = np.asarray(smoothed_value)
        else:
            if self.bin_type == 'sqrt':
                value = np.sqrt(value)
            bucket_weights = np.asarray(value)
        bucket_weights = bucket_weights / bucket_weights.sum()

        bucket_weights = self.weight_refine(bin_num, bucket_weights)
        bucket_weights = torch.tensor(bucket_weights, device=self.device)
        bucket_weights = bucket_weights.view(1, -1, 1, 1)
        return bucket_weights

    def bin_refine_cfg(self):
        if self.bin_refine_cfg is None:
            self.bin_refine_type = None  # reset to False
        
        if self.bin_refine_type == 'tail_refine':
            self.zero_end = self.bin_refine_cfg.get("zero_end", None)
            self.refine_begin = self.bin_refine_cfg.get("refine_begin", None)
            self.head_refine = self.bin_refine_cfg.get("head_refine", False)
            self.head_range = self.bin_refine_cfg.get("head_range", 0)
            if self.tail_refine:
                if self.zero_end is None or self.refine_begin is None:
                    raise ValueError("must set the number!")
                
    def weight_refine(self, bin_num, bucket_weights):
        if self.bin_refine_type == 'tail_refine':
            zero_positions = np.where(bin_num < self.refine_begin)[0]
            zero_positions = zero_positions[self.zero_end:]
            min_zero_position = zero_positions.min()
            bucket_weights[zero_positions] = bucket_weights[zero_positions[0]]

        return bucket_weights
    
    def _save_bin_info(self, bin_statistic):
        save_bin_weight = self.bin_weight.clone()
        save_bin_weight = save_bin_weight.view(-1)
        array = save_bin_weight.cpu().numpy()
        path = mkdir(os.path.join(dataloader_path, 'bin_statistic'))  # path to save the weight
        bin_statistic_name = bin_statistic.split("/")[-1]
        weight_name = bin_statistic_name
        if self.using_lds:
            weight_name = 'lds' + 'kernel_' + self.lds_cfg['kernel'] + '_ks_' + \
            str(self.lds_cfg['kernel_size']) + '_sigma_' + str(self.lds_cfg['sigma']) + bin_statistic_name
        if self.bin_refine_type is not None:
            weight_name = self.bin_refine_type + weight_name
        # np.save(os.path.join(path, weight_name), array)

