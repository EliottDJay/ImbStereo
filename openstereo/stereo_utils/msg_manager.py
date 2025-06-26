import logging
import os.path as osp
import time
from time import strftime, localtime

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# utils
from .visualization import disp_error_image_func
from utils.check import is_list, is_tensor, ts2np, mkdir
from utils.basic import Odict, NoOp
from utils.logger import Logger as Log
from utils.check import tensor2numpy, tensor2float


class MessageManager:
    def __init__(self, summary_path):
        self.info_dict = Odict()
        self.writer_hparams = ['image', 'scalar']
        self.init_manager(summary_path)
        self.no_img_summary = True

    def init_manager(self, summary_path, iteration=0):
        self.iteration = iteration
        self.writer = SummaryWriter(summary_path, purge_step=self.iteration)

    def train_step(self, summary):
        self.write_to_tensorboard(summary, self.iteration)
        self.iteration += 1

    def save_scalars(self, mode_tag, scalar_dict, global_step):
        scalar_dict = tensor2float(scalar_dict)
        for tag, values in scalar_dict.items():
            if not isinstance(values, list) and not isinstance(values, tuple):
                values = [values]
            for idx, value in enumerate(values):
                scalar_name = '{}/{}'.format(mode_tag, tag)
                # if len(values) > 1:
                scalar_name = scalar_name + "_" + str(idx)
                self.writer.add_scalar(scalar_name, value, global_step)

    def save_images(self, mode_tag, images_dict, global_step):
        if self.no_img_summary:
            return
        
        # update error map
        images_dict["errormap"] = [disp_error_image_func.apply(images_dict['disp_est'], images_dict['disp_gt'])]
        del images_dict['disp_est']

        images_dict = tensor2numpy(images_dict)
        for tag, values in images_dict.items():
            if not isinstance(values, list) and not isinstance(values, tuple):
                values = [values]
            for idx, value in enumerate(values):
                if len(value.shape) == 3:
                    value = value[:, np.newaxis, :, :]
                value = value[:1]
                value = torch.from_numpy(value)

                image_name = '{}/{}'.format(mode_tag, tag)
                if len(values) > 1:
                    image_name = image_name + "_" + str(idx)
                self.writer.add_image(image_name,
                                 vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                                 global_step)

    def write_to_tensorboard(self, summary, iteration=None):
        iteration = self.iteration if iteration is None else iteration
        for k, v in summary.items():
            module_name = k.split('/')[0]
            board_name = k.replace(module_name + "/", '')
            writer_module = getattr(self.writer, 'add_' + module_name)
            v = v.detach() if is_tensor(v) else v
            v = vutils.make_grid(v, normalize=True, scale_each=True) if 'image' in module_name else v
            writer_module(board_name, v, iteration)

    # def write_to_tensorboard(self, summary):
    #
    #     for k, v in summary.items():
    #         module_name = k.split('/')[0]
    #         if module_name not in self.writer_hparams:
    #             self.log_warning(
    #                 'Not Expected --Summary-- type [{}] appear!!!{}'.format(k, self.writer_hparams))
    #             continue
    #         board_name = k.replace(module_name + "/", '')
    #         writer_module = getattr(self.writer, 'add_' + module_name)
    #         v = v.detach() if is_tensor(v) else v
    #         v = vutils.make_grid(
    #             v, normalize=True, scale_each=True) if 'image' in module_name else v
    #         if module_name == 'scalar':
    #             try:
    #                 v = v.mean()
    #             except:
    #                 v = v
    #         writer_module(board_name, v, self.iteration)


# msg_mgr = MessageManager()
# noop = NoOp()

def update_image_log(preds):
    img_out = {}
    for s in range(len(preds)):
        save_name = 'pred_disp' + str(len(preds) - s - 1)
        img_out[save_name] = preds[s]

    return img_out

def init_logger(args):
    Log.init(logfile_level=args['logfile_level'],
             stdout_level=args['stdout_level'],
             log_file=args["log_file"],
             log_format=args['log_format'],
             rewrite=args['rewrite'])


def logging_initialized(args):
    if not torch.distributed.is_initialized():
        init_logger(args)
    elif torch.distributed.get_rank() > 0:
        pass
    else:
        init_logger(args)


def get_msg_mgr(cfg):
    summary_path = cfg.get('summary_path', None)

    assert summary_path is not None, "It is supposed to initialize the path setup and make the summary path!!"

    if not torch.distributed.is_initialized():
        return MessageManager(summary_path)
    elif torch.distributed.get_rank() > 0:
        return NoOp()
    else:
        return MessageManager(summary_path)
