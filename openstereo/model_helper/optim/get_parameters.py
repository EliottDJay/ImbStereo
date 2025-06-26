import torch.nn as nn


# ContraSeg
def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        else:
            if hasattr(m, 'weight'):
                group_no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def get_parameters(self):
    bb_lr = []
    nbb_lr = []
    fcn_lr = []
    params_dict = dict(self.seg_net.named_parameters())
    for key, value in params_dict.items():
        if 'backbone' in key:
            bb_lr.append(value)
        elif 'aux_layer' in key or 'upsample_proj' in key:
            fcn_lr.append(value)
        else:
            nbb_lr.append(value)

    params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
    return params
