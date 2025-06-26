import os
#import random
#from collections import OrderedDict
from glob import glob
from utils.logger import Logger as Log

from utils.distributed import get_rank
import torch
# import torch.distributed as dist
from collections import OrderedDict


def convert_state_dict(ori_state_dict, is_dist=True):
    new_state_dict = OrderedDict()
    if is_dist:
        if not next(iter(ori_state_dict)).startswith('module'):
            for k, v in ori_state_dict.items():
                new_state_dict[f'module.{k}'] = v
        else:
            new_state_dict = ori_state_dict
    else:
        if not next(iter(ori_state_dict)).startswith('module'):
            new_state_dict = ori_state_dict
        else:
            for k, v in ori_state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v

    return new_state_dict


def load_state(path, model, optimizer=None, key="state_dict", best_dict=None):
    rank = get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            Log.info("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

            # V1
            """metric_now = checkpoint["pred"]
            best_metric = checkpoint["best_pred"]
            save_metric = checkpoint["save_metric"]
            last_epoch = checkpoint["epoch"]
            best_epoch = checkpoint["best_epoch"]
            train_metric = checkpoint["train_metric"]

            if best_dict is not None:
                best_dict.best_epoch = best_epoch
                best_dict.best_pred = best_metric
                best_dict.best_metric = save_metric
                Log.info('setting the MeterDictBest: best epoch is {}, '
                         'best pred is {} {}'.format(best_dict.best_epoch, best_dict.best_pred, best_dict.best_metric))
                if 'swa_best' in checkpoint.keys():
                    assert 'swa_best_pred' in best_dict.extra_dict.keys()
                    best_dict.extra_dict['swa_best_pred'] = checkpoint['swa_best']
                if 'swa_best_epoch' in checkpoint.keys():
                    assert 'swa_best_epoch' in best_dict.extra_dict.keys()
                    best_dict.extra_dict['swa_best_epoch'] = checkpoint['swa_best_epoch']"""
            # V2
            last_epoch = checkpoint["epoch"]
            train_metric = checkpoint["train_metric"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if rank == 0:
                Log.info(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_epoch
                    )
                )
        return last_epoch, train_metric
    else:
        if rank == 0:
            Log.info("=> no checkpoint found at '{}'".format(path))


def save_checkpoint(save_path, optimizer, model, epoch, pred_dict_now, best_dict, save_metric=None,
                    swa_dict=None, train_metric='epoch', filename=None, save_optimizer=True, net_name=None):
    if (net_name is None) and (filename is None):
        if hasattr(model, 'get_name'):
            net_name = model.get_name()
        else:
            net_name = 'Stereo'

    if filename is None:
        net_filename = net_name + '_epoch_{:0>3d}_'.format(epoch)
        net_filename = net_filename + '.pth'
    elif filename is not None:
        net_filename = filename

    if not net_filename.endswith('.pth'):
        net_filename = net_filename + '.pth'

    net_save_path = os.path.join(save_path, net_filename)

    """if best_dict is None:
        # MeterDictBest 的版本
        save_metric = 'epoch'
        best_pred_dict = None
        best_pred = None
        best_epoch = None 
    else:
        save_metric = best_dict.best_metric
        best_pred_dict = best_dict.data
        # data的 字典更新的比较可靠
        if best_pred_dict is not None:
            best_pred = best_pred_dict[save_metric][0]
            best_epoch = best_pred_dict[save_metric + '_epoch'][0]
        else:
            best_pred = None
            best_epoch = None

    if pred_dict_now is not None:
        pred = pred_dict_now[save_metric][0]
    else:
        pred = None"""


    if best_dict is not None:
        best_pred = best_dict.best_result
    else:
        best_pred = None

    """state = {
            'epoch': epoch,
            # 'num_iter': num_iter,
            'pred': pred,
            'best_pred': best_pred,
            'best_epoch': best_epoch,
            'pred_dict': pred_dict_now,
            'best_dict': best_pred_dict,
            'save_metric': save_metric,
            'train_metric': train_metric,
            'state_dict': model.state_dict(),
        }"""

    state = {
        'epoch': epoch,
        # 'num_iter': num_iter,
        'best_dict': best_pred,
        'pred_dict': pred_dict_now,
        'save_metric': save_metric,
        'train_metric': train_metric,
        'state_dict': model.state_dict(),
    }

    if save_optimizer:
        state['optimizer_state'] = optimizer.state_dict()

    """if swa_dict is not None:
        state['swa_model'] = swa_dict['swa_model'].state_dict()
        state['swa_best'] = best_pred_dict['swa_'+save_metric][0]
        state['swa_best_epoch'] = best_pred_dict['swa_'+save_metric+'_epoch'][0]"""

    torch.save(state, net_save_path)


def resume_latest_ckpt(checkpoint_dir, net, optimizer, best=False, bestdict=None):
    if best:
        ckpts = sorted(glob(checkpoint_dir + '/' + '*best.pth'))
    else:
        # in descending order: best first and then latest
        ckpts = sorted(glob(checkpoint_dir + '/' + '*.pth'))

    if len(ckpts) == 0:
        raise RuntimeError('=> No checkpoint found while resuming training')

    latest_ckpt = ckpts[-1]

    if hasattr(net, 'get_name'):
        net_name = net.get_name()
    else:
        net_name = 'Stereo'

    print('=> Resume latest %s checkpoint: %s' % (net_name, os.path.basename(latest_ckpt)))

    last_epoch, train_metric = load_state(latest_ckpt, net, optimizer, best_dict=bestdict)

    return last_epoch, train_metric