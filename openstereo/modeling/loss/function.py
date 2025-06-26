# import numpy as np
# import scipy.ndimage as nd
import torch
# import torch.nn as nn
# from torch.nn import functional as F

def robust_loss(x, a, c):
    # A general and adaptive robust loss function
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


def sparse_volume_recover(sparse_volume, sparse_index, max_range, count=0.1):
    assert len(sparse_volume.size()) == 4  # B D H W
    assert sparse_volume.size() == sparse_index.size()
    b, d, h, w = sparse_volume.shape
    vdevice = sparse_volume.device
    full_volume = torch.ones((b, max_range, h, w))*count
    spase_new = sparse_volume.clone()
    ind = sparse_index.clone()
    ind = ind.long()
    ind = ind.to(vdevice)
    full_volume = full_volume.to(vdevice)
    full = full_volume.scatter_(1, ind, spase_new)
    return full


def sparse_gt_recover(gt, mask_now, max_range, inter_two=False):
    # only one
    assert len(gt.size()) == 3
    b, h, w = gt.shape
    vdevice = gt.device
    gt_floor = torch.floor(gt)  # [B, h, w]
    gt_floor = gt_floor * mask_now
    floor_index1 = gt_floor.long()
    floor_index1 = floor_index1.unsqueeze(dim=1)
    full_volume = torch.zeros((b, max_range, h, w))
    full_volume = full_volume.to(vdevice)
    if not inter_two:
        full = full_volume.scatter_(1, floor_index1, 1)
    elif inter_two:
        ceil_index = floor_index1 + 1 
        one_mask = ceil_index >= max_range-1  
        ceil_index[one_mask] = 0  #
        delta = gt - gt_floor
        delta_verse = 1 - delta
        delta = delta.unsqueeze(dim=1)
        delta_verse = delta_verse.unsqueeze(dim=1)
        delta_set = torch.cat((delta, delta_verse), dim=1)
        index_set = torch.cat((floor_index1, ceil_index), dim=1)
        full = full_volume.scatter_(1, index_set, delta_set)
    return full

