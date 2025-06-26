from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from utils.logger import Logger as Log

class ClipGrad(object):
    def __init__(self, cfg):
        super(ClipGrad, self).__init__()
        grad_clip_set = cfg.get('grad_clip', {})
        """if grad_clip_set is None:
            grad_clip_set = dict()"""
        self.clip_type = grad_clip_set.get('type', None)
        self.clip_value = grad_clip_set.get('clip_value', 0.1)  # clip_value
        self.max_norm = grad_clip_set.get('max_norm', 35)  # max_norm
        self.norm_type = grad_clip_set.get('norm_type', 2)

        if self.clip_type == 'value':
            Log.info(f"Using clip grad value tools in training, clip value is set to {self.clip_value}")
        if self.clip_type == 'norm':
            Log.info(f"Using clip grad {str(self.norm_type)}-norm tools in training, max norm is set to self.max_norm")

    def __call__(self, model):
        if self.clip_type is None:
            pass
        elif self.clip_type == 'value':
            clip_grad_value_(model.parameters(), self.clip_value)
        elif self.clip_type == 'norm':
            clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)
        else:
            raise ValueError(f"Unknown clip type {self.clip_type}.")