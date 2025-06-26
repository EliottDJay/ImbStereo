from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import torch
import math
from utils.logger import Logger as Log
# from contextlib import contextmanager
import torchcontrib
from torch.optim import Optimizer, lr_scheduler
from .learning_rate_adjust import LearningRateAdjust, WarmupCosineSchedule


def get_scheduler(cfg_trainer, optimizer, len_data, last):
    # basic scheduler
    scheduler = None
    cfg_lr = cfg_trainer["lr_scheduler"]
    policy = cfg_lr["mode"]
    on_epoch = cfg_lr.get('on_epoch', True)
    if on_epoch:
        metric = 'epoch'
    else:
        metric = 'step'
    lr_kwargs = cfg_lr["kwargs"]
    max_iters = int(cfg_trainer["epochs"] * len_data)

    if policy == 'step':
        if metric == 'epoch':
            scheduler = lr_scheduler.StepLR(optimizer, **lr_kwargs, last_epoch=last)
        else:
            NotImplementedError('lr_scheduler.StepLR cant update following iter updating but the epoch!')
    elif policy == 'multistep':
        if metric == 'epoch':
            scheduler = lr_scheduler.MultiStepLR(optimizer, **lr_kwargs, last_epoch=last)
            gama = lr_kwargs.get('gamma', 0.1)
            Log.info("beginning at {} epoch, setting MultiStepLR learning schedule renew"
                     " the learning rate at epoch {} with gamma {}".format(
                last, lr_kwargs['milestones'], gama
            ))
        else:
            raise NotImplementedError('lr_scheduler.MultiStepLR cant update following iter updating but the epoch!')
    elif policy =="learningrateadjust":
        scheduler = LearningRateAdjust(optimizer, **lr_kwargs, last_epoch=last)
    elif policy =="onecycle":
        if metric == 'iter':
            # to compatible with RAFT
            max_lr = lr_kwargs['max_lr']
            del lr_kwargs['max_lr']
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, **lr_kwargs, last_epoch=last)
            Log.info("beginning at {} iteration, setting OneCycleLR learning schedule renew")
        else:
            raise NotImplementedError("lr_scheduler.OneCycleLR cant update following epoch updating but the iter!")
    elif policy == 'lambda_poly':
        if metric == 'iter':
            power = lr_kwargs.get('power', 0.9)
            Log.info('Use lambda_poly policy with power {}'.format(0.9))
            lambda_poly = lambda iters: pow((1.0 - iters / max_iters), power)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly, last_epoch=last)
        else:
            raise NotImplementedError('lambda_poly cant update following epoch updating but the iter!')
    elif policy == 'lambda_cosine':
        if metric == 'iter':
            lambda_cosine = lambda iters: (math.cos(math.pi * iters / max_iters)
                                           + 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine, last_epoch=last)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **lr_kwargs)
    elif policy == 'swa_lambda_poly':
        optimizer = torchcontrib.optim.SWA(optimizer)
        normal_max_iters = int(max_iters * 0.75)
        swa_step_max_iters = (max_iters - normal_max_iters) // 5 + 1  # we use 5 ensembles here

        def swa_lambda_poly(iters):
            if iters < normal_max_iters:
                return pow(1.0 - iters / normal_max_iters, 0.9)
            else:  # set lr to half of initial lr and start swa
                return 0.5 * pow(1.0 - ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters, 0.9)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_poly, last_epoch=last)
    elif policy == 'swa_lambda_cosine':
        optimizer = torchcontrib.optim.SWA(optimizer)
        normal_max_iters = int(max_iters * 0.75)
        swa_step_max_iters = (max_iters - normal_max_iters) // 5 + 1  # we use 5 ensembles here

        def swa_lambda_cosine(iters):
            if iters < normal_max_iters:
                return (math.cos(math.pi * iters / normal_max_iters) + 1.0) / 2
            else:  # set lr to half of initial lr and start swa
                return 0.5 * (math.cos(
                    math.pi * ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters) + 1.0) / 2

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_cosine, last_epoch=last)

    elif policy == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000,
                                         t_total=max_iters, last_epoch=last)

    else:
        NotImplementedError('Policy:{} is not valid.'.format(policy))

    return scheduler