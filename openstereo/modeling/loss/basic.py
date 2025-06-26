import torch
import torch.nn as nn
from torch.nn import functional as F

# utils
from openstereo.evaluation.experiments import AverageMeter
from utils.logger import logging as Log
from utils.distributed import all_reduce_bysize
from utils.check import tensor2float

class DispSmooth(nn.Module):
    def __init__(self, cfg, rank=0, device=None):
        super(DispSmooth, self).__init__()
        # basic config
        loss_cfg = cfg['loss']
        self.max_disp = cfg['net']['max_disparity']  # 可以设置为None
        self.rank = rank
        self.device = device
        self.loss_weight = loss_cfg['weight']
        self.highest_only = loss_cfg.get("highest_only", False)  # used in AANet
        # self.main_disp_loss  only SmoothL1 here

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

        # highest only: 
        if self.highest_only:
            self.loss_weight = [1.]
        self.pred_num = len(self.loss_weight)

        # re-weiging
        self.reweight_cfg = loss_cfg.get('reweight', None)
        if self.reweight_cfg is not None:
            self.reweight_index = self.reweight_cfg.get('index', None)
            if self.reweight_index is not None:
                self.reweight_index = range(len(self.loss_weight))

        if rank == 0:
            self.disp_losses = AverageMeter()
            self.total_losses = AverageMeter()


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

        if self.reweight_cfg is not None:
            reweight = data['reweight']
            assert reweight is not None, "The reweight ground truth is necessary for reweighting process."
            processed_inputs.update(
                {
                    'reweight': reweight,
                }
            )
        if device is not None:
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device) if torch.is_tensor(v) else v
            
        return processed_inputs

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
            raise ValueError('Len of the loss weight does match the preds')
        disp_loss = 0
        pyramid_loss = []

        disp_gt = loss_inputs['disp']  # 必须要有这个
        if self.reweight_cfg is not None:
            pixel_weight = loss_inputs['reweight']

        for k in range(preds_nums):
            pred_disp = preds[k]
            loss_weight = self.loss_weight[k]
            gt_now = disp_gt.clone()  # [B, H, W]
            if self.reweight_cfg is not None:
                reweight_now = pixel_weight.clone()

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
                    if self.reweight_cfg is not None and k in self.reweight_index and k != (len(preds)-1):
                        reweight_now = reweight_now.unsqueeze(1)
                        reweight_now = F.interpolate(reweight_now, size=(pred_disp.size(-2), pred_disp.size(-1)),
                                              mode='nearest')
                        reweight_now = reweight_now.squeeze(1) # [B H W]
            mask_now = self.mask_generator(gt_now)
            gt_input = gt_now[mask_now]
            pred_input = pred_disp[mask_now]
            if self.reweight_cfg is None or k not in self.reweight_index:
                curr_loss = F.smooth_l1_loss(gt_input, pred_input, reduction='mean')
            elif self.reweight_cfg is not None and k in self.reweight_index:
                curr_loss = F.smooth_l1_loss(gt_input, pred_input, reduction='none')
                curr_loss = curr_loss * reweight_now[mask_now]
                curr_loss = torch.mean(curr_loss)

            disp_loss += loss_weight * curr_loss
            c_loss = all_reduce_bysize(curr_loss)
            pyramid_loss.append(c_loss)

        total_loss = disp_loss
        d_loss = all_reduce_bysize(disp_loss)
        t_loss = all_reduce_bysize(total_loss)

        if self.rank == 0:
            # update需要all reduce更新 但是需要反传的total loss是不可以的
            self.disp_losses.update(tensor2float(d_loss))
            self.total_losses.update(tensor2float(t_loss))

        # loss的数值都是没有经过all reduce的
        # pyramid的数值都是经过all reduce的
        return {
            "total_loss": total_loss,
            "multi_preds_loss": disp_loss,
            "multi_preds_pyramid": pyramid_loss,
        }

    def loss_stat(self):
        # 这两个函数都是在rank0中执行的
        disp_stat = "Disp Loss: {disp_loss.val:.4f} ({disp_loss.avg:.4f})\t".format(disp_loss=self.disp_losses)
        total_stat = "Total Loss: {total_loss.val:.4f} ({total_loss.avg:.4f})\t".format(total_loss=self.total_losses)

        loss_stat = disp_stat + total_stat
        return loss_stat

    def scalar_outputs_update(self, loss_dict):
        scalar_outputs = {"disp_loss": self.disp_losses.avg, "total_loss": self.total_losses.avg}

        """if hasattr(self, 'set_mask') and self.set_mask:
            scalar_outputs['included_mask'] = self.mask_rate.avg
            self.mask_rate.reset()"""
        
        if "multi_preds_pyramid" in loss_dict.keys():
            pyramid_loss = loss_dict["multi_preds_pyramid"]
            for s in range(len(pyramid_loss)):
                key_name = 'disp_loss_pyrmd_' + str(len(pyramid_loss) - s - 1)
                scalar_outputs[key_name] = pyramid_loss[s]

        self.disp_losses.reset()
        self.total_losses.reset()

        return scalar_outputs

        