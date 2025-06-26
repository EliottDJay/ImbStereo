from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import SGD, Adam, AdamW, RMSprop
from .lamb import Lamb

from utils.basic import NoOp
from utils.logger import Logger as Log


# AANet
def specific_params_group(kv, specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']):
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False


def basic_params_group(kv, specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']):
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True


def get_optimizer(model, cfg_trainer):

    """
    """
    # adopt swa
    cfg_optim = cfg_trainer["optimizer"]
    optim_type = cfg_optim["type"]
    optim_kwargs = cfg_optim["kwargs"]

    # para dict
    optim_para = cfg_optim.get('optim_para', "basic")
    # basic deformable

    params_list = []
    
    if not optim_kwargs.get('lr', False):
        Log.error("Find no lr args in the Configuration")
        exit(1)

    # params_list.append(dict(params=model.parameters(), lr=cfg_optim["kwargs"]["lr"]))
    if optim_para == "basic":
        params_list.append(dict(params=model.parameters(), lr=cfg_optim["kwargs"]["lr"]))
    elif optim_para == "deformable":
        # AANet
        specific_params = list(filter(specific_params_group, model.named_parameters()))
        base_params = list(filter(basic_params_group, model.named_parameters()))
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]
        specific_lr = cfg_optim["kwargs"]["lr"] * 0.1
        params_list = [
        {'params': base_params, 'lr': cfg_optim["kwargs"]["lr"]},
        {'params': specific_params, 'lr': specific_lr},
    ]
    else:
        raise NotImplementedError

    if optim_type == "SGD":
        optimizer = SGD(params_list, **optim_kwargs)
    elif optim_type == "Adam":
        optimizer = Adam(params_list, **optim_kwargs)
    elif optim_type == "AdamW":
        optimizer = AdamW(params_list, **optim_kwargs)
    elif optim_type == "RMSprop":
        optimizer = RMSprop(params_list, **optim_kwargs)
    elif optim_type == "Lamb":
        optimizer = Lamb(params_list, **optim_kwargs)
    else:
        optimizer = NoOp()

    assert optimizer is not None, "optimizer type is not supported by Lab"
    optim_kwargs_str = ", ".join([f"'{key}': {value}" for key, value in optim_kwargs.items()])
    log_info = f"For parameter using the [{optim_para}] type and using [{optim_type}] optimizer, detail configurations are as below and the others are follow the default:\t [{optim_kwargs_str}]"
    Log.info(log_info)

    return optimizer

