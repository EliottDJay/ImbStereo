import math
import logging
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

# utils
from utils.logger import Logger as Log


def smoothing_setting(smoothing_cfg):
    """
    including: 'kernel', 'kernel_size', 'sigma'
    """
    if smoothing_cfg is None:
        smoothing_cfg = dict()
    if 'kernel' not in smoothing_cfg.keys():
        smoothing_cfg['kernel'] = 'gaussian'
    elif smoothing_cfg['kernel'] not in ['gaussian', 'triang', 'laplace']:
        kernel = smoothing_cfg['kernel']
        Log.info(f'We suppoose smoothing kernel to be in [gaussian, triang or laplace] but we get {kernel}'
                    f'we will change it to gaussian kernel')
        smoothing_cfg['kernel'] = 'gaussian'
    if 'kernel_size' not in smoothing_cfg.keys():
        smoothing_cfg['kernel_size'] = 5
    smoothing_cfg['kernel_size'] = int(smoothing_cfg['kernel_size'])
    if smoothing_cfg['kernel_size'] % 2 != 1:
        ks = smoothing_cfg['kernel_size']
        Log.info(f'LDS kernel size should be odd number, but now is {ks}, and we change it to 5')
        smoothing_cfg['kernel_size'] = 5
    if 'sigma' not in smoothing_cfg.keys():
        smoothing_cfg['sigma'] = 2
    Log.info(
        "Using LDS : {}, {}/{}".format(smoothing_cfg['kernel'], smoothing_cfg['kernel_size'], smoothing_cfg['sigma']))

    return smoothing_cfg


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.2, clip_max=5.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        # print(torch.sum(valid_pos))
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.2, clip_max=5.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        # print(torch.sum(valid_pos))
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2