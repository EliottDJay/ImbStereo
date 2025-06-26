import numpy as np
import torch

from utils.logger import Logger as Log



class RightSelfAugmentation(object):
    """
    Hitnet and Hierarchical deep stereo matching on highresolution images:
    We then replace random areas of the right image with random crops taken from another portion of the right image: this
    helps the network to deal with occluded areas and encourages a better “inpainting”. The crop size to be replaced is
    randomly sampled between [50,50] and [180, 250].
    """
    def __init__(self, cfg, stage=None):
        prob = cfg.get('prob', 0.5)
        self.h_size = cfg.get('h', [50, 180])  
        self.w_size = cfg.get('w', [50, 250]) 
        self.p = prob
        Log.info("Using RightShift Transfomer to {} dataset. The crop size to be replaced randomly sampled "
                 "between {} and {} at probability {}.".format(stage, self.h_size, self.w_size, self.p))

    def __call__(self, sample):

        if np.random.random() < self.p:
            # Log.info("conduct the shift right operation")
            size_h = int(np.random.uniform(self.h_size[0], self.h_size[1]))
            size_w = int(np.random.uniform(self.w_size[0], self.w_size[1]))

            x1 = int(np.random.uniform(0, sample['right'].shape[1] - size_w))
            x2 = int(np.random.uniform(0, sample['right'].shape[1] - size_w))
            y1 = int(np.random.uniform(0, sample['right'].shape[0] - size_h))
            y2 = int(np.random.uniform(0, sample['right'].shape[0] - size_h))
            right = np.copy(sample['right'])  # #deep copy
            sample['shifted'] = torch.tensor([[y1, size_h], [x1, size_w]])
            sample['right'][y1:y1+size_h, x1:x1+size_w] = right[y2:y2+size_h, x2:x2+size_w]

        return sample
    
    def epoch_pass(self, sample):
        if np.random.random() < self.p:
            size_h = int(np.random.uniform(self.h_size[0], self.h_size[1]))
            size_w = int(np.random.uniform(self.w_size[0], self.w_size[1]))
            _ = int(np.random.uniform(0, sample['right_shape'][1] - size_w))
            _ = int(np.random.uniform(0, sample['right_shape'][1] - size_w))
            _ = int(np.random.uniform(0, sample['right_shape'][0] - size_h))
            _ = int(np.random.uniform(0, sample['right_shape'][0] - size_h))
        return sample



