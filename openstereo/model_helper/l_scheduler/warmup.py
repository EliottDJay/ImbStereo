import math
# from contextlib import contextmanager

from utils.logger import Logger as Log
from torch.optim import Optimizer

from utils.basic import NoOp


def get_warmup_scheduler(cfg_trainer, optimizer, last=-1):
    """
    :param cfg_trainer:
    :param optimizer:
    :param last:
    :return:
    """
    warmup_scheduler = NoOp()
    if 'warmup' not in cfg_trainer.keys():

        cfg_trainer['warmup'] = {'warmup_steps': 0}  
        return warmup_scheduler
    warmup_set = cfg_trainer['warmup']
    type = warmup_set.get('type', 'linear')
    warmup_steps = warmup_set.get('warmup_steps', 100)
    if type == 'linear':
        # TODO： LinearWarmup  的 warmup_params 是 warmup_steps
        warmup_scheduler = LinearWarmup(optimizer, warmup_steps, last_step=last)
    else:
        Log.error('Policy:{} is not valid.'.format(type))
        exit(1)
    Log.info("Using warmup with {} before {} step".format(type, warmup_set))

    return warmup_scheduler


######################################################################
# ***************************** WarmUp  ******************************
# See OpenStereo
# ####################################################################

class BaseWarmup(object):
    """Base class for all warmup schedules

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_params, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.warmup_step = warmup_params
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampen the learning rates.

        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] *= omega

    def reloading(self):
        for group, lr in zip(self.optimizer.param_groups, self.lrs):
            group['lr'] = lr

    def dampening(self):
        """for group, lr in zip(self.optimizer.param_groups, self.lrs):
            group['lr'] = lr
        yield"""
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def warmup_factor(self, step, **params):
        raise NotImplementedError


def get_warmup_params(warmup_period, group_count):
    if type(warmup_period) == list:
        if len(warmup_period) != group_count:
            raise ValueError(
                'size of warmup_period does not equal {}.'.format(group_count))
        for x in warmup_period:
            if type(x) != int:
                raise ValueError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif type(warmup_period) == int:
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(warmup_period).__name__))
    return warmup_params


class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(LinearWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step+1) / warmup_period)


class ExponentialWarmup(BaseWarmup):
    """Exponential warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Effective warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(ExponentialWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return 1.0 - math.exp(-(step+1) / warmup_period)


def rho_inf_fn(beta2):
    return 2.0 / (1 - beta2) - 1


def rho_fn(t, beta2, rho_inf):
    b2t = beta2 ** t
    rho_t = rho_inf - 2 * t * b2t / (1 - b2t)
    return rho_t


def get_offset(beta2, rho_inf):
    if not beta2 > 0.6:
        raise ValueError('beta2 ({}) must be greater than 0.6'.format(beta2))
    offset = 1
    while True:
        if rho_fn(offset, beta2, rho_inf) > 4:
            return offset
        offset += 1


class RAdamWarmup(BaseWarmup):
    """RAdam warmup schedule.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        warmup_params = [
            dict(
                beta2=x['betas'][1],
                rho_inf=rho_inf_fn(x['betas'][1]),
            )
            for x in optimizer.param_groups
        ]
        for x in warmup_params:
            x['offset'] = get_offset(**x)
        super(RAdamWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, beta2, rho_inf, offset):
        rho = rho_fn(step+offset, beta2, rho_inf)
        numerator = (rho - 4) * (rho - 2) * rho_inf
        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
        return math.sqrt(numerator/denominator)


class UntunedLinearWarmup(LinearWarmup):
    """Untuned linear warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(2.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedLinearWarmup, self).__init__(optimizer, warmup_period, last_step)


class UntunedExponentialWarmup(ExponentialWarmup):
    """Untuned exponetial warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(1.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedExponentialWarmup, self).__init__(optimizer, warmup_period, last_step)