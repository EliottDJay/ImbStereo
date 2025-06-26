import random
import numpy as np
import cv2
import torch
from .basic_function import Compose
from PIL import Image
from torchvision.transforms import ColorJitter, functional

from utils.logger import Logger as Log


class TestCrop(object):
    # same in FastACV sceneflow_dataset # https://github.com/gangweiX/Fast-ACVNet
    def __init__(self, cfg, stage=None):
        if stage == 'train':
            Log.error("Maybe you should reset to val or test stage")
        img_height, img_width = cfg["height"], cfg["width"]
        self.size = [img_height, img_width]
        Log.info(f"Adding TestCrop, cropping {stage} img to height{str(img_height)} x width{str(img_width)}")
        Log.info("TestCrop starts at the bottom left corner! [h - crop_h:h, w - crop_w: w]")

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)  # ensure crop_h is within the bounds of the image
        crop_w = min(w, crop_w)  # ensure crop_w is within the bounds of the image

        for k in sample.keys():
            # crop the specified arrays to the desired size
            if k in ['left', 'right']:
                sample[k] = sample[k][h - crop_h:h, w - crop_w: w, :]
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'pseudo_disp', 'gradient']:
                sample[k] = sample[k][h - crop_h:h, w - crop_w: w]
            elif k in ['dxdy']:  # slant
                value = np.copy(sample[k])
                shape = value.shape
                if shape[0] == 2:  # [2, H, W]
                    # dxdy = dxdy.transpose(1, 2, 0) # dxdy = dxdy[h - crop_h:h, w - crop_w: w] # dxdy = dxdy.transpose(2, 0, 1)
                    value = value[:, h - crop_h:h, w - crop_w: w]
                elif shape[2] == 2:
                    value = value[h - crop_h:h, w - crop_w: w]
                else:
                    raise ValueError("the ground truth of the slant dont match the project!")
                sample[k] = value

        return sample
    
    def epoch_pass(self, sample):
        h, w, c = sample['left_shape']
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)  # ensure crop_h is within the bounds of the image
        crop_w = min(w, crop_w)  # ensure crop_w is within the bounds of the image
        pass_sample = {'left_shape': [crop_h, crop_w, c], 'right_shape': [crop_h, crop_w, c]}
        return pass_sample


class StereoPad(object):
    # typically used in KITTI dataset
    def __init__(self, cfg, stage=None):
        if stage == 'train':
            Log.error("Maybe you should reset to val or test stage")
        img_height, img_width = cfg["height"], cfg["width"]
        self.size = [img_height, img_width]
        Log.info(f"Adding StereoPad, padding the {stage} image to turn its size" 
                 "to height{str(img_height)} x width{str(img_width)}")
        Log.info("Only padding the top and the right side")

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size

        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:  # [H, W, 3]
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'pseudo_disp', 'gradient']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
            elif k in ['dxdy']:  # slant
                value = np.copy(sample[k])
                shape = value.shape
                if shape[0] == 2:  # [2, H, W]
                    # dxdy = dxdy.transpose(1, 2, 0) # dxdy = dxdy[h - crop_h:h, w - crop_w: w] # dxdy = dxdy.transpose(2, 0, 1)
                    value = np.pad(value, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                       constant_values=0)
                elif shape[2] == 2:
                    value = np.pad(value, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant',
                                       constant_values=0)
                else:
                    raise ValueError("the ground truth of the slant dont match the project!")
                sample[k] = value
        sample['top_pad'] = pad_top
        sample['right_pad'] = pad_right
        return sample
    
    def epoch_pass(self, sample):
        h, w, c = sample['left_shape']
        th, tw = self.size
        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image
        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        pass_sample = {'left_shape': [h + pad_top + pad_bottom, w + pad_left + pad_right, c], 'right_shape': [h + pad_top + pad_bottom, w + pad_left + pad_right, c]}
        return pass_sample


class CenterCrop(object):
    def __init__(self, cfg, stage=None):
        if stage == 'train':
            Log.error("Maybe you should reset to val or test stage")
        img_height, img_width = cfg["height"], cfg["width"]
        self.size = [img_height, img_width]
        Log.info(f"Adding CenterCrop, cropping {stage} img to height{str(img_height)} x width{str(img_width)}")
        Log.info("Cropping the content in the center of the image")

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        tw = min(w, tw)  # ensure tw is within the bounds of the image
        th = min(h, th)  # ensure th is within the bounds of the image

        x1 = (w - tw) // 2  # compute the left edge of the centered rectangle
        y1 = (h - th) // 2  # compute the top edge of the centered rectangle

        for k in sample.keys():
            # crop the specified arrays to the centered rectangle
            if k in ['left', 'right']:  # [H, W, 3]
                sample[k] = sample[k][y1: y1 + th, x1: x1 + tw, :]
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'pseudo_disp', 'gradient']:
                sample[k] = sample[k][y1: y1 + th, x1: x1 + tw]
            elif k in ['dxdy']:  # slant
                value = np.copy(sample[k])
                shape = value.shape
                if shape[0] == 2:  # [2, H, W]
                    # dxdy = dxdy.transpose(1, 2, 0) # dxdy = dxdy[h - crop_h:h, w - crop_w: w] # dxdy = dxdy.transpose(2, 0, 1)
                    value = value[:, y1: y1 + th, x1: x1 + tw]
                elif shape[2] == 2:
                    value = value[y1: y1 + th, x1: x1 + tw]
                else:
                    raise ValueError("the ground truth of the slant dont match the project!")
                sample[k] = value
        return sample
    
    def epoch_pass(self, sample):
        h, w, c = sample['left_shape']
        th, tw = self.size
        tw = min(w, tw)  # ensure tw is within the bounds of the image
        th = min(h, th)  # ensure th is within the bounds of the image

        pass_sample = {'left_shape': [th, tw, c], 'right_shape': [th, tw, c]}
        return pass_sample


class DivisiblePad(object):
    def __init__(self, cfg, stage=None):
        if stage == 'train':
            Log.error("Maybe you should reset to val or test stage")
        self.by = cfg['by']
        Log.info(f"Adding DivisiblePad, padding the {stage} image to be evenly divided by{str(cfg['by'])}")

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_top = h + self.by - h % self.by - h
        else:
            pad_top = 0
        if w % self.by != 0:
            pad_right = w + self.by - w % self.by - w
        else:
            pad_right = 0
        pad_left = 0
        pad_bottom = 0

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
            elif k in ['dxdy']:  # slant
                value = np.copy(sample[k])
                shape = value.shape
                if shape[0] == 2:  # [2, H, W]
                    # dxdy = dxdy.transpose(1, 2, 0) # dxdy = dxdy[h - crop_h:h, w - crop_w: w] # dxdy = dxdy.transpose(2, 0, 1)
                    value = np.pad(value, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
                elif shape[2] == 2:
                    value = np.pad(value, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant',
                                   constant_values=0)
                else:
                    raise ValueError("the ground truth of the slant dont match the project!")
                sample[k] = value
        sample['top_pad'] = pad_top
        sample['right_pad'] = pad_right
        return sample
    
    def epoch_pass(self, sample):
        h, w, c = sample['left_shape']
        if h % self.by != 0:
            pad_top = h + self.by - h % self.by - h
        else:
            pad_top = 0
        if w % self.by != 0:
            pad_right = w + self.by - w % self.by - w
        else:
            pad_right = 0
        pad_left = 0
        pad_bottom = 0
        pass_sample = {'left_shape': [h + pad_top + pad_bottom, w + pad_right + pad_left, c], 'right_shape': [h + pad_top + pad_bottom, w + pad_right + pad_left, c]}
        return pass_sample

class RandomCrop(object):
    def __init__(self, cfg, stage=None):
        if stage != 'train':
            Log.error("Maybe you should use in train stage")
        img_height, img_width = cfg["height"], cfg["width"]
        self.size = [img_height, img_width]  # [h, w]
        Log.info("Adding RandomCrop, randomly cropping the area of size" 
                 " height {} x width {} in {} image".format(img_height, img_width, stage))

    def __call__(self, sample):
        crop_height, crop_width = self.size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        assert crop_width < width and crop_height < height
        if crop_width >= width or crop_height >= height:  #invalid
            return sample
        else:
            x1, y1, x2, y2 = self.get_random_crop_coords(height, width, crop_height, crop_width)
            # Log.info('sample id is {}. x1-x2: {}:{}, y1-y2: {}:{}'.format(sample["index"], x1, x2, y1, y2))
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = sample[k][y1: y2, x1: x2, :]
                elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'pseudo_disp', 'gradient']:
                    sample[k] = sample[k][y1: y2, x1: x2]
                elif k in ['dxdy']:  # slant
                    value = np.copy(sample[k])
                    shape = value.shape
                    if shape[0] == 2:  # [2, H, W]
                        value = value[:, y1: y2, x1: x2]
                    elif shape[2] == 2:
                        value = value[y1: y2, x1: x2, :]
                    else:
                        raise ValueError("the ground truth of the slant dont match the project!")
                    sample[k] = value
            return sample

    @staticmethod
    def get_random_crop_coords(height, width, crop_height, crop_width):
        """
        get coordinates for cropping, start from 0

        :param height: image height, int
        :param width: image width, int
        :param crop_height: crop height, int
        :param crop_width: crop width, int
        :return: xy coordinates
        """
        x1 = random.randint(0, width - crop_width)
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    @staticmethod
    def crop(img, x1, y1, x2, y2):
        """
        crop image given coordinates

        :param img: input image, [H,W,3]
        :param x1: coordinate, int
        :param y1: coordinate, int
        :param x2: coordinate, int
        :param y2: coordinate, int
        :return: cropped image
        """
        img = img[y1:y2, x1:x2]
        return img
    
    def epoch_pass(self, sample):
        crop_height, crop_width = self.size
        height, width, channel = sample['left_shape'] # (H, W, 3)
        assert crop_width < width and crop_height < height
        pass_sample = {'left_shape': [height, width, channel], 'right_shape': [height, width, channel]}
        return pass_sample
    

class RandomCropv2(object):
    def __init__(self, cfg, stage=None, validate=False):
        img_height, img_width = cfg["height"], cfg["width"]
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        height_reset, width_reset = False, False
        if self.img_height is None:
            self.img_height = (ori_height // 32 ) * 32
            height_reset = True
        if self.img_width is None:
            self.img_width = (ori_width // 32 ) * 32
            width_reset = True
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0
            sample['top_pad'] = top_pad
            sample['right_pad'] = right_pad

            sample['left'] = np.lib.pad(sample['left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            #  ((top_pad//2, top_pad-top_pad//2), (0, right_pad), (0, 0)),
            sample['right'] = np.lib.pad(sample['right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)

            if 'disp' in sample.keys():
                sample['disp'] = np.lib.pad(sample['disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.lib.pad(sample['pseudo_disp'],
                                                   ((top_pad, 0), (0, right_pad)),
                                                   mode='constant',
                                                   constant_values=0)
            if 'dxdy' in sample.keys():
                sample['dxdy'] = np.lib.pad(sample['dxdy'],
                                            ((0, 0), (top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

            if 'original_left' in sample.keys():
                sample['original_left'] = np.lib.pad(sample['original_left'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
                sample['original_right'] = np.lib.pad(sample['original_right'],
                                             ((top_pad, 0), (0, right_pad), (0, 0)),
                                             mode='constant',
                                             constant_values=0)

            if "extra_aug_times" in sample.keys():
                for i in range(sample['extra_aug_times']):
                    sample['extra_aug_left' + str(i)] = np.lib.pad(sample['extra_aug_left' + str(i)],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
                    sample['extra_aug_right' + str(i)] = np.lib.pad(sample['extra_aug_right' + str(i)],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                start_weight = 0
                self.offset_x = random.randint(start_weight, ori_width - self.img_width)  # need +1？

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = random.randint(start_height, ori_height - self.img_height)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width)  # // 2
                self.offset_y = (ori_height - self.img_height)  # // 2
            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            if 'original_left' in sample.keys():
                sample['original_left'] = self.crop_img(sample['original_left'])
                sample['original_right'] = self.crop_img(sample['original_right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = self.crop_img(sample['pseudo_disp'])
            if 'dxdy' in sample.keys():
                dxdy = np.copy(sample['dxdy'])
                shape = dxdy.shape
                if shape[0] == 2:
                    dxdy = dxdy.transpose(1, 2, 0)
                    dxdy = self.crop_img(dxdy)
                    dxdy = dxdy.transpose(2, 0, 1)
                elif shape[2] == 2:
                    pass
                else:
                    Log.error("the ground truth of the slant dont match the project!")
                    exit(1)
                sample['dxdy'] = dxdy

            if "extra_aug_times" in sample.keys():
                for i in range(sample['extra_aug_times']):
                    sample['extra_aug_left' + str(i)] = self.crop_img(sample['extra_aug_left' + str(i)])
                    sample['extra_aug_right' + str(i)] = self.crop_img(sample['extra_aug_right' + str(i)])

        if height_reset:
            self.img_height = None
        if width_reset:
            self.img_width = None

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomFlip(object):
    def __init__(self, do_flip_type='h', h_flip_prob=0.5, v_flip_prob=0.1):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.do_flip_type = do_flip_type

    def __call__(self, sample):

        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        if np.random.rand() < self.h_flip_prob and self.do_flip_type == 'hf':  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            disp = disp[:, ::-1] * -1.0

        if np.random.rand() < self.h_flip_prob and self.do_flip_type == 'h':  # h-flip for stereo
            tmp = img1[:, ::-1]
            img1 = img2[:, ::-1]
            img2 = tmp

        if np.random.rand() < self.v_flip_prob and self.do_flip_type == 'v':  # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            disp = disp[::-1, :]

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample
    
    def epoch_pass(self, sample):
        _ = np.random.rand()
        _ = np.random.rand()
        _ = np.random.rand()
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        return self.horizontal_flip(sample) if self.p < random.random() else sample

    @staticmethod
    def horizontal_flip(sample):
        img_left = sample['left']  # (3, H, W)
        img_right = sample['right']  # (3, H, W)
        disp_left = sample['disp']  # (H, W)
        disp_right = sample['disp_right']  # (H, W)

        left_flipped = img_left[:, ::-1]
        right_flipped = img_right[:, ::-1]
        img_left = right_flipped
        img_right = left_flipped
        disp = disp_right[:, ::-1]
        disp_right = disp_left[:, ::-1]

        sample['left'] = img_left
        sample['right'] = img_right
        sample['disp'] = disp
        sample['disp_right'] = disp_right

        if 'occ_mask' in sample.keys():
            occ_left = sample['occ_mask']  # (H, W)
            occ_right = sample['occ_mask_right']  # (H, W)
            occ = occ_right[:, ::-1]
            occ_right = occ_left[:, ::-1]
            sample['occ_mask'] = occ
            sample['occ_mask_right'] = occ_right
        return sample
    
    def epoch_pass(self, sample):
        _ = random.random()
        return sample

