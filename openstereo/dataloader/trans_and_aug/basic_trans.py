# import random
# import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from utils.logger import logging as Log


class ToTensor(object):
    def __init__(self):
        Log.info("Using ToTensor")

    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                # Convert the numpy array to a PyTorch Tensor and set its datatype to float32
                sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample
    
    def epoch_pass(self, sample):
        return sample



class TransposeImage(object):
    def __init__(self):
        Log.info("Using TransposeImage")

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample
    
    def epoch_pass(self, sample):
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        Log.info("Using Normalization")

    def __call__(self, sample):
        sample['left'] = self.normalize(sample['left'], self.mean, self.std)
        sample['right'] = self.normalize(sample['right'], self.mean, self.std)
        return sample
    """def __call__(self, sample):
        norm_keys = ['left', 'right']
        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)
        return sample"""

    @staticmethod
    def normalize(img, mean, std):
        return TF.normalize(img / 255.0, mean=mean, std=std)
    
    def epoch_pass(self, sample):
        return sample


class GetValidDisp(object):
    def __init__(self, max_disp):
        self.max_disp = max_disp

    def __call__(self, sample):
        disp = sample['disp']
        disp[disp > self.max_disp] = 0
        disp[disp < 0] = 0
        sample.update({
            'disp': disp,
        })
        if 'disp_right' in sample.keys():
            disp_right = sample['disp_right']
            disp_right[disp_right > self.max_disp] = 0
            disp_right[disp_right < 0] = 0
            sample.update({
                'disp_right': disp_right
            })

        return sample
    
    def epoch_pass(self, sample):
        return sample


class GetValidDispNOcc(object):
    def __call__(self, sample):
        w = sample['disp'].shape[-1]
        occ_mask = self.compute_left_occ_region(w, sample['disp'])
        sample['occ_mask'][occ_mask] = True  # update
        sample['occ_mask'] = np.ascontiguousarray(sample['occ_mask'])
        try:
            occ_mask = self.compute_right_occ_region(w, sample['disp_right'])
            sample['occ_mask_right'][occ_mask] = 1
            sample['occ_mask_right'] = np.ascontiguousarray(sample['occ_mask_right'])
        except KeyError:
            sample['occ_mask_right'] = np.zeros_like(occ_mask).astype(np.bool)

        # remove invalid disp from occ mask
        occ_mask = sample['occ_mask']
        sample['disp'][occ_mask] = 0
        sample['disp'] = np.ascontiguousarray(sample['disp'], dtype=np.float32)
        return sample

    @staticmethod
    def compute_left_occ_region(w, disp):
        """
        Compute occluded region on the left image border

        :param w: image width
        :param disp: left disparity
        :return: occ mask
        """

        coord = np.linspace(0, w - 1, w)[None,]  # 1xW
        shifted_coord = coord - disp
        occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ
        return occ_mask

    @staticmethod
    def compute_right_occ_region(w, disp):
        """
        Compute occluded region on the right image border

        :param w: image width
        :param disp: right disparity
        :return: occ mask
        """
        coord = np.linspace(0, w - 1, w)[None,]  # 1xW
        shifted_coord = coord + disp
        occ_mask = shifted_coord > w  # occlusion mask, 1 indicates occ

        return occ_mask
    
    def epoch_pass(self, sample):
        return sample


# used in some Non-geometric augmentation
class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))
        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)
        return sample

