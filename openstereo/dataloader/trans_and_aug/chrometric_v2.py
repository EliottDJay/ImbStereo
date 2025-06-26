import copy
import random
import numpy as np
import torchvision.transforms.functional as F
from .basic_trans import ToPILImage, ToNumpyArray

# utils
from utils.logger import Logger as Log
from utils.check import isNum

## One of nongeo augmentation, this fill is to continue with the Lab2's augmentation

class SymorAsymAdjustments(object):
    def __init__(self, p_op=1, p_asym=0.5, intervel=[0.8, 1.2], asym_inter=None):
        """
        :param p_op: conduct the operation
        :param p_asym: asymmetric adjustment
        """
        assert 0 <= p_op <= 1, "The probability of operation should be in [0, 1]"
        assert 0 <= p_asym <= 1, "The probability of asymmetric adjustment should be in [0, 1]"
        self.p_op = p_op
        self.p_asym = p_asym
        self.intervel = intervel
        if asym_inter is not None:
            self.asym_intervel = asym_inter
        else:
            self.asym_intervel = self.intervel

    def __call__(self, sample):
        raise NotImplementedError

    def _conduct(self):
        if self.p_op == 1:
            return True
        elif self.p_op < 1:
            return np.random.random() < self.p_op

    def _symorasym(self):
        if self.p_asym == 1:
            factor = np.random.uniform(self.asym_intervel[0], self.asym_intervel[1], 2)
        elif self.p_asym < 1 and np.random.random() < self.p_asym:
            factor = np.random.uniform(self.asym_intervel[0], self.asym_intervel[1], 2)
        else:
            sym_factor = np.random.uniform(self.intervel[0], self.intervel[1])
            factor = [sym_factor, sym_factor]
        return factor


# basic
# Random coloring
class RandomContrast(SymorAsymAdjustments):
    """
    Random contrast
    np.random.uniform(0.8, 1.2) AANet
    """
    def __call__(self, sample):
        if self._conduct():
            contrast_factor = self._symorasym()
            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor[0])
            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor[1])
        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            _ = self._symorasym()
        return sample


class RandomBrightness(SymorAsymAdjustments):
    # np.random.uniform(0.5, 2.0) 
    def __call__(self, sample):
        if self._conduct():
            brightness = self._symorasym()
            sample['left'] = F.adjust_brightness(sample['left'], brightness[0])
            sample['right'] = F.adjust_brightness(sample['right'], brightness[1])
        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            _ = self._symorasym()
        return sample


class RandomHue(SymorAsymAdjustments):
    # np.random.uniform(-0.1, 0.1) 
    def __call__(self, sample):
        if self._conduct():
            hue = self._symorasym()
            sample['left'] = F.adjust_hue(sample['left'], hue[0])
            sample['right'] = F.adjust_hue(sample['right'], hue[1])
        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            _ = self._symorasym()
        return sample


class RandomSaturation(SymorAsymAdjustments):
    # np.random.uniform(0.8, 1.2)
    def __call__(self, sample):
        if self._conduct():
            saturation = self._symorasym()
            sample['left'] = F.adjust_saturation(sample['left'], saturation[0])
            sample['right'] = F.adjust_saturation(sample['right'], saturation[1])
        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            _ = self._symorasym()
        return sample


class RandomGamma(SymorAsymAdjustments):
    # np.random.uniform(0.7, 1.5) 
    def __call__(self, sample):
        if self._conduct():
            gamma_factor = self._symorasym()
            sample['left'] = F.adjust_gamma(sample['left'], gamma_factor[0])
            sample['right'] = F.adjust_gamma(sample['right'], gamma_factor[1])
        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            _ = self._symorasym()
        return sample


class ChromaticAugmentationV2(object):
    def __init__(self, chromatic_cfg):
        Log.info("ChromaticAugmentationV2 is the same as ChromaticAugmentation, but with little defferent details. This" 
                 "function is to continue with our code Lab2")
        self.p_op = chromatic_cfg.get('prob', 1.0)
        assert 0 <= self.p_op <= 1, "The probability of operation should be in [0, 1]"

        self.chromatic_strategy = chromatic_cfg.get('chromatic_strategy', 'fixed')  
        self.sample_aug_init = chromatic_cfg.get('sample_aug_init', 1)  
        self.transform = []
        self._aug_strategy_init(chromatic_cfg) 
        self.shuffle = False
        self.considered_aug_init = range(len(self.transform))
        self._paras_init()
        self._log_info()
        self.sample_aug = copy.deepcopy(self.sample_aug_init)  

    def _aug_strategy_init(self, cfg):
        aug_order = cfg.get('strategy_order', None)
        original_order = copy.deepcopy(aug_order)
        detail_info = ""
        stored_order = []
        if aug_order is None:
            aug_order = list(cfg.keys())
        for k in aug_order:
            if k.lower() == 'brightness':
                brightness = cfg['brightness']
                self.transform.append(RandomBrightness(p_op=brightness['p_op'], p_asym=brightness['p_asym'],
                                      intervel=brightness['intervel'], asym_inter=brightness.get('asym', None)))
                stored_order.append('brightness')
                brightness_info = f"brightness with prob {brightness['p_op']}, asym prob {brightness['p_asym']}, intervel {brightness['intervel']}"
                detail_info += brightness_info + '\n'
            elif k.lower() == 'contrast':
                contrast = cfg['contrast']
                self.transform.append(RandomContrast(p_op=contrast['p_op'], p_asym=contrast['p_asym'],
                                      intervel=contrast['intervel'], asym_inter=contrast.get('asym', None)))
                stored_order.append('contrast')
                contrast_info = f"contrast with prob {contrast['p_op']}, asym prob {contrast['p_asym']}, intervel {contrast['intervel']}"
                detail_info += contrast_info + '\n'
            elif k.lower() == 'gamma':
                gamma = cfg['gamma']
                self.transform.append(RandomGamma(p_op=gamma['p_op'], p_asym=gamma['p_asym'],
                                                  intervel=gamma['intervel'], asym_inter=gamma.get('asym', None)))
                stored_order.append('gamma')
                gamma_info = f"gamma with prob {gamma['p_op']}, asym prob {gamma['p_asym']}, intervel {gamma['intervel']}"
                detail_info += gamma_info + '\n'
            elif k.lower() == 'hue':
                hue = cfg['hue']
                self.transform.append(RandomHue(p_op=hue['p_op'], p_asym=hue['p_asym'],
                                                intervel=hue['intervel'], asym_inter=hue.get('asym', None)))
                stored_order.append('hue')
                hue_info = f"hue with prob {hue['p_op']}, asym prob {hue['p_asym']}, intervel {hue['intervel']}"
                detail_info += hue_info + '\n'
            elif k.lower() == 'saturation':
                saturation = cfg['saturation']
                self.transform.append(RandomSaturation(p_op=saturation['p_op'], p_asym=saturation['p_asym'],
                                      intervel=saturation['intervel'], asym_inter=saturation.get('asym', None)))
                stored_order.append('saturation')
                saturation_info = f"saturation with prob {saturation['p_op']}, asym prob {saturation['p_asym']}, intervel {saturation['intervel']}"
                detail_info += saturation_info + '\n'
        appended_order = self._order2str(stored_order)
        pointed_order = self._order2str(original_order)
        Log.info(f"(Non-Geometric) Chromatic Augmentation Operations are all added with the order [{appended_order}];\t"
                 f"the original pointed order in Chromatic Augmentation Configuration file is [{pointed_order}]")
        Log.info(f"Details:\n{detail_info}")
        
    def _paras_init(self):
        num = len(self.transform)
        if self.chromatic_strategy in ['random']:
            self.shuffle = True
            self.considered_aug_init = np.arange(num)
        elif self.chromatic_strategy in ['fixed']:
            self.shuffle = False
            self.sample_aug_init = num
            self.considered_aug_init = np.arange(num)
        else:
            self.shuffle = False

    def _order2str(self, order_list: list=None):
        order_str = ''
        if order_list is None:
            return order_str
        elif isinstance(order_list, list):
            order_link = order_list[:-1]
            for op in order_link:
                order_str = order_str + op + '>>>'
            order_str += order_list[-1]
        else:
            TypeError
        return order_str

    def _conduct(self):
        return np.random.random() < self.p_op

    def _aug_sample(self):
        aug_set = copy.deepcopy(self.considered_aug_init)
        if self.shuffle:
            random.shuffle(aug_set)
        sample_set = aug_set[:self.sample_aug]
        return sample_set

    def __call__(self, sample):
        if self._conduct():
            sample = ToPILImage()(sample)
            sample_set = self._aug_sample()
            for op in sample_set:
                sample = self.transform[op](sample)
            sample = ToNumpyArray()(sample)

        return sample
    
    def epoch_pass(self, sample):
        if self._conduct():
            sample_set = self._aug_sample()
            for op in sample_set:
                sample = self.transform[op].epoch_pass(sample)
        return sample

    def _log_info(self):
        if self.chromatic_strategy == 'fixed':
            Log.info("The augmentation to data will be fixed with the order setup in configuration!")
        elif self.chromatic_strategy == 'random':
            Log.info("Augmentation operations will shuffle and sample at every iteration.")
            Log.info("Augmentation Strength will keep num {} during training".format(self.sample_aug_init))  #TODO delete
        else:
            NotImplementedError('Chromatic strategy {} is not valid'.format(self.chromatic_strategy))