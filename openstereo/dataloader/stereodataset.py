from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
# import sys
from functools import partial
# obtain the abspath to save weight file
dataloader_file = os.path.abspath(__file__)  # /./././.py
dataloader_split = dataloader_file.split('/')
dataloader_path = os.path.join(*dataloader_split[:-1])  # .py

del dataloader_split

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random

# utils
from utils.logger import Logger as Log
from utils.check import mkdir

# data_io and augmentation
from PIL import Image
from openstereo.dataloader.basic_function.data_io import read_text_lines, pil_loader, cv2_loader, pfm_disp_loader, \
    _read_kitti_disp, _read_disp_png
# from .trans_and_aug import build_transformer

# imbalance tools
from scipy.ndimage import convolve1d
from Imbalance.utils.smoothing_basic import smoothing_setting, get_lds_kernel_window


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StereoDataset(Dataset):
    def __init__(self, cfg, dataset_name, mode, transform=None, seed=None, max_disp=192, loss_cfg=None):
        super(StereoDataset, self).__init__()

        # some config
        self.mode = mode  # scope  ['val', 'noval', 'test']
        self.dataset_name = dataset_name.lower()
        self.D = max_disp  # may be used in imbalanced learning, disp can be None but unfit here
        if loss_cfg is None:
           loss_cfg = dict()
        self.loss_type = loss_cfg.get("type", "DispSmooth") 
        self.transform = transform
        self.scale_factor = cfg.get("scale_factor", 256.)  # SCARED 128. KITTI 256. FPM no need
        if 'SCARED'.lower() in self.dataset_name:
            self.scale_factor = 128.
        Log.info("scale_factor is {}".format(self.scale_factor))
        self.image_loader, self.disp_loader = None, None
        self.build_info_read(cfg)  
        self.epoch_pass()

        self.save_filename = cfg.get("save_filename", True)  
        self.data_root = cfg['dataroot'] 
        data_list = cfg['data_list']  
        self.use_noc = cfg.get("use_noc", False) 
        self.load_right_disp = cfg.get("right_disp", False)  
        self.load_occ_mask = cfg.get("occ_mask", False)  
        self.load_pseudo_gt = cfg.get("pseudo_gt", False)  
        self.load_slant = cfg.get("slant", False)  
        self.with_gradient = cfg.get("with_gradient", False) 
        self.samples = []

        if data_list.endswith(".txt"):
            self.read_from_txt(data_list)
        else:
            pass  

        # imbalance part (weight, lds)
        self.lds = loss_cfg.get('lds', False)
        # label smoothing techniques
        if self.lds:
            lds_config = loss_cfg.get('lds_set', None)
            self.lds_config = smoothing_setting(lds_config)
        # re-weighting techniques
        self.pixel_reweight = loss_cfg.get('reweight', None) 
        self.weights = None
        if self.pixel_reweight is not None:
            assert self.pixel_reweight in {'inverse', 'sqrt_inv'}
            self.weight_refined = loss_cfg.get('weight_refined', False)  
            
            Log.info(f'Using re-weighting: [{self.pixel_reweight}], and Using weight-refined is setting [{self.weight_refined}]')

            self.bin_size = 1
            self.max_w_index = int(self.D / self.bin_size - 1)  # finite counts
            bin_num = None  
            bin_statistic = loss_cfg.get("bin_weight", None)
            assert isinstance(bin_statistic, str)
            bin_num = np.load(bin_statistic)
            self.weights = self._get_imbalance_weight(bin_num)  # bin weight list
            path = mkdir(os.path.join(dataloader_path, 'reweight'))  # path to save the weight
            weight_name = self.pixel_reweight
            if self.lds:
                weight_name = weight_name + 'lds' + 'kernel_' + self.lds_set['kernel'] + '_ks_' + str(self.lds_set['kernel_size']) + '_sigma_' + str(self.lds_set['sigma'])
            if self.weight_refined:
        
                weight_name = "weight_refined_" + weight_name
            bin_name = bin_statistic.split('/')
            weight_name = weight_name + bin_name[-1]
            np.save(os.path.join(path, weight_name), self.weights)

    def __len__(self):
        return len(self.samples)

    def build_info_read(self, cfg):
        image_reader_type = cfg.get("image_reader", 'PIL')
        disp_reader_type = cfg.get("disp_reader", 'PIL')
        if image_reader_type == 'PIL':
            self.image_loader = pil_loader
        elif image_reader_type == 'CV2':
            self.image_loader = cv2_loader
        else:
            raise NotImplementedError('Image reader type not supported: {}'.format(image_reader_type))

        if disp_reader_type == 'PIL' and 'kitti' in self.dataset_name:
            self.disp_loader = _read_kitti_disp
        elif disp_reader_type == 'PIL' and self.scale_factor is not None:
            self.disp_loader = partial(_read_disp_png, scale_factor=self.scale_factor)
        elif disp_reader_type == 'PFM':
            # dont choose use subset, for only -1* for the left disp but 1* for the right in 'FlyingThings3DSubset'
            self.disp_loader = pfm_disp_loader
        else:
            raise NotImplementedError('Disp reader type not supported: {}'.format(disp_reader_type))
        
    def epoch_pass(self, set_stat=False):
        self._epoch_pass = set_stat

    def __getitem__(self, index):
        sample = {}
        # Log.info("index info {}".format(index))
        sample["index"] = index
        sample_path = self.samples[index]
        if self.save_filename:  # use to save info
            sample['left_name'] = sample_path['left_name']
        sample['left'] = self.image_loader(sample_path['left'])  # [H, W, 3]
        sample['right'] = self.image_loader(sample_path['right'])

        if self._epoch_pass:
            pass_sample = {'left_shape': list(sample['left'].shape), 'right_shape': list(sample['right'].shape)}
            pass_sample = self.transform.epoch_pass(pass_sample)
            # Log.info("into")
            return pass_sample

        sample['top_pad'], sample['right_pad'] = 0, 0
        sample['shifted'] = torch.tensor([[-1, -1], [-1, -1]])  
        if sample_path['disp'] is not None:
            disp_map = self.disp_loader(sample_path['disp'])
            disp_map = np.nan_to_num(disp_map, nan=0.0)  # replace nan with 0
            disp_map[disp_map == np.inf] = 0
            sample['disp'] = disp_map
        if sample_path['pseudo_disp'] is not None and sample_path['disp'] is not None:
            sample['pseudo_disp'] = self.disp_loader(sample_path['pseudo_disp'])  # [H, W]
        if self.with_gradient:
            sample['gradient'] = self._get_gradient_map(sample['left'])
        if sample_path['disp'] is not None and self.load_slant and sample_path['dxdy'] is not None:
            sample["dxdy"] = self.disp_loader(sample_path["dxdy"])  # [2, H, W]
        if self.load_right_disp:
            # sample['disp_right']
            pass
        if self.load_occ_mask and sample_path['disp'] is not None:
            sample["occ_mask"] = np.array(Image.open(sample_path['occ_path'])).astype(np.bool)
            sample['occ_mask_right'] = np.array(Image.open(sample_path['occ_right_path'])).astype(np.bool)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.weights is not None and sample_path['disp'] is not None and 'val' in self.mode:
            sample['weight'] = self._get_weight(sample['disp'])

        return sample
    
    @staticmethod
    def collect_fn(batch):
        return batch[0]

    def read_from_txt(self, data_list):
        lines = read_text_lines(data_list)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]  # when testing in KITTI, usually there is no disp

            sample = dict()
            sample['left_name'] = left_img

            sample['left'] = os.path.join(self.data_root, left_img)
            sample['right'] = os.path.join(self.data_root, right_img)
            sample['disp'] = os.path.join(self.data_root, gt_disp) if gt_disp is not None else None

            if gt_disp is not None and "kitti" in self.dataset_name.lower() and self.use_noc:
                sample['disp'] = sample['disp'].replace('disp_occ', 'disp_noc')

            if self.load_slant and gt_disp is not None:
                sample['dxdy'] = None
                if "kitti" in self.dataset_name.lower():
                    splits = str(gt_disp).split("/")
                    name = os.path.splitext(splits[-1])[0]  
                    name = name + ".npy"
                    mode = splits[0]
                    tag = splits[1]
                    if "0" in tag and "disp_occ" in tag:
                        set = "_2015"
                    elif "0" not in tag and "disp_occ" in tag:
                        set = "_2012"
                    sample['dxdy'] = os.path.join(self.data_root, mode, ("slant_window" + set), name)

            self.samples.append(sample)

    def _get_imbalance_weight(self, bin_num):
        bin_num = bin_num + 1e-06
        value = bin_num.copy()
        if self.lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_config['kernel'], self.lds_config['kernel_size'], self.lds_config['sigma'])
            if self.pixel_reweight == 'sqrt_inv':
                value = np.sqrt(value)
            smoothed_value = convolve1d(np.asarray(value), weights=lds_kernel_window, mode='reflect')
            scaling = np.sum(bin_num) / np.sum(np.array(bin_num) / np.array(smoothed_value))
            bin_weights = [np.float32(scaling / smoothed_value[bucket]) for bucket in range(len(bin_num))]
        else:
            if self.pixel_reweight == 'sqrt_inv':
                value = np.sqrt(value)
            scaling = np.sum(bin_num) / np.sum(np.array(bin_num) / np.array(value))
            bin_weights = [np.float32(scaling / value[bucket]) for bucket in range(len(bin_num))]
        Log.info("getting the re-weighting weight")

        return bin_weights

    def get_bin_idx(self, x):
        index = int(x * np.float32(self.bin_size))
        if index < 0:
            index = 0
        ind = min(index, self.max_w_index)
        return ind

    def _get_weight(self, disp):
        sp = disp.shape
        if self.weights is not None:
            disp = disp.reshape(-1).cpu().numpy()
            assert disp.dtype == np.float32
            weights = np.array(list(map(lambda v: self.weights[self.get_bin_idx(v)], disp)))
            weights = torch.tensor(weights, dtype=torch.float32).view(*sp)
        else:
            weights = torch.tensor([np.float32(1.)], dtype=torch.float32).repeat(*sp)
        return weights

    def _get_gradient_map(self, img_np):
        dx_imgL = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        dy_imgL = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        dxy_imgL = np.sqrt(np.sum(np.square(dx_imgL), axis=-1) + np.sum(np.square(dy_imgL), axis=-1))
        dxy_imgL = dxy_imgL / (np.max(dxy_imgL) + 1e-5)
        return dxy_imgL
