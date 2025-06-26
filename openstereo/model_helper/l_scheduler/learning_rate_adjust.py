import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class LearningRateAdjust(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, base_lr, lrepochs, warmup_steps=None, last_epoch=-1):
        self.base_lr = base_lr
        splits = lrepochs.split(':')
        assert len(splits) == 2  
        self.downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
        self.downscale_rate = [float(eid_str) for eid_str in splits[1].split(',')]
        self.wamup_steps = warmup_steps
        super(LearningRateAdjust, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if self.wamup_steps:
            # 这里还没写
            if step < self.wamup_steps:
                return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        lr = self.base_lr
        for eid, downscale_rate in zip(self.downscale_epochs, self.downscale_rate):
            if step >= eid:
                lr /= downscale_rate
            else:
                break
        # print("setting learning rate to {}".format(lr))
        return lr


def learningrate_adjust_inresume(optimizer, scheduler):
    lr_list = scheduler.get_last_lr()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_list[0]