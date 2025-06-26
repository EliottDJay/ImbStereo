from __future__ import print_function, division
import sys
import os
import argparse
import copy
import yaml
import json
import inspect

import numpy as np

#import torchvision.utils as vutils
#import torch.nn.functional as F
from collections import OrderedDict, namedtuple
import torch
import torch.nn.parallel
import torch.utils.data
import torch.autograd as autograd
import torch.nn as nn
from .basic import Odict

from .basic import make_iterative_func, make_nograd_func
from .logger import Logger as Log


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


# count the number of the net
def params_count(net):
    num_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num = sum(p.numel() for p in net.parameters())
    return num_trainable, num


def save_args(args, path, filename='args.json'):
    # args_dict = vars(args)
    mkdir(path)
    save_path = os.path.join(path, filename)

    with open(save_path, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=False)

def save_command(save_path, filename='command_train.txt'):
    mkdir(save_path)
    command = sys.argv
    save_file = os.path.join(save_path, filename)
    with open(save_file, 'w') as f:
        f.write(' '.join(command))

@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)

@make_iterative_func
def check_alltenser(vars):
    assert isinstance(vars, torch.Tensor)

@make_iterative_func
def allclonedetach(vars):
    assert isinstance(vars, torch.Tensor)


#######################################################################################################################
# type transform from one to another
#
#
#######################################################################################################################


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def int2bool(v):
    """ Usage:
    parser.add_argument('--x', type=int2bool, nargs='?', const=True,
                        dest='x', help='Whether to use pretrained models.')
    """
    if int(v) == 1:
        return True
    elif int(v) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def input2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    elif isNum(v):
        if int(v) == 1:
            return True
        elif int(v) == 0:
            return False
        else:
            return False
    else:
        Log.error("Can not return a bool")
        exit(1)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


#######################################################################################################################
# check args
#
#
#######################################################################################################################

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    # with open("./configs/default.yaml", 'r') as stream:
    #     dst_cfgs = yaml.safe_load(stream)
    # MergeCfgsDict(src_cfgs, dst_cfgs)
    return src_cfgs


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        Log.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v

#######################################################################################################################
# check type
#
#
#######################################################################################################################


def isNum(n):
    try:
        # n=eval(n)
        if type(n)==int or type(n)==float or type(n)==complex:
            return True
    except NameError:
        return False


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)

