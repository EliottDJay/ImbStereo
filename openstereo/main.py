import argparse
import os
import sys
import yaml

current_path = os.path.abspath(__file__)
file_split = current_path.split('/')  

path_new = os.path.join(* file_split[:-2])
abspath = "/" + path_new
sys.path.append(abspath)

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
#CUDA_LAUNCH_BLOCKING=1

import random
import numpy as np
import torch

# torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openstereo.modeling import models


# utils
from openstereo.stereo_utils.msg_manager import logging_initialized
from utils.distributed import DDPPassthrough
from utils.check import params_count, str2bool, config_loader, int2bool, save_command, save_args, params_count
from utils.logger import Logger as Log
from utils.initialization import init_seeds, path_checking, PathManager


def arg_parse():
    parser = argparse.ArgumentParser(description='Main program for OpenStereo.')
    parser.add_argument('--config', type=str, default='',
                        help="path of config file")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--distributed', type=int2bool, default=False, help="disable distributed training")
    parser.add_argument('--master_addr', type=str, default='localhost', help="master address")
    parser.add_argument('--master_port', type=int, default=12355, help="master port")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for non-distributed mode.")
    parser.add_argument('--restore_hint', type=str, default=0, help="restore hint for loading checkpoint.")
    # ***********  Params for logging. and screen  **********
    parser.add_argument('--logfile_level', default='info', type=str, help='To set the log level to files.')
    parser.add_argument('--stdout_level', default='info', type=str, help='To set the level to print to screen.')
    # parser.add_argument('--log_file', default="log/stereo.log", type=str, dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=False, help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        help='Whether to write logging into files.')
    parser.add_argument('--log_format', type=str, nargs='?', default="%(asctime)s %(levelname)-7s %(message)s"
                        , help='Whether to write logging into files.')

    opt = parser.parse_args()

    # cfgs = config_loader(opt.config)
    cfgs = yaml.load(open(opt.config, "r"), Loader=yaml.FullLoader)
    opt = vars(opt)  # convert to a dictionary

    opt_new = opt.copy()
    opt_new.update(cfgs)

    return opt_new


def ddp_init(rank, world_size, master_addr, master_port):
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        assert isinstance(master_port, int)
        os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def initialization(args):
    # Log configuration
    pth_mgr = PathManager(args, abspath)
    pth_mgr.add2args(args)
    # path_checking(args, abspath)
    file_sub_name = 'test' in pth_mgr.test_file_sign
    
    if not args['trainer']['resume'] and not args['dataset']['scope'] == 'test' and not file_sub_name: 
        files = os.listdir(pth_mgr.exp_dir)
        for file in files:
            if file.endswith('.pth'):
                raise NotImplementedError("Experiment exit."
                "The purpose of this error message is to prevent overwriting the original experiment!")

    logging_initialized(args)
    # seed setting and deterministic
    seed = args.get("seed", 0)
    seed = seed if not args.get("distributed", False) else seed + dist.get_rank()
    deterministic = args.get('deterministic', True)
    init_seeds(seed, cuda_deterministic=deterministic)
    return pth_mgr


def worker(rank, world_size, args):
    is_dist = args.get("distributed", False)
    if is_dist:
        ddp_init(rank, world_size, args['master_addr'], args['master_port'])
        torch.cuda.set_device(rank)
    # print(args['device'])
    device = torch.device(f'cuda:{rank}') if is_dist else torch.device(args['device'])
    pth_mgr = initialization(args)
    # msg_mgr = get_msg_mgr()
    model_cfg = args['net']
    data_cfg = args['dataset']
    trainer_cfg = args['trainer']
    scope = data_cfg['scope']
    Model = getattr(models, model_cfg['method'])
    model = Model(args)

    if rank == 0:
        # model_name = model.get_name()
        num_params_train, num = params_count(model)
        Log.info('=> Number of trainable parameters: %d' % num_params_train)
        Log.info('=> Number of parameters: %d' % num)

    Trainer = model.Trainer

    if is_dist and trainer_cfg.get('sync_bn', False):
        Log.info('convert batch norm to sync batch norm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)

    if is_dist:

        find_unused_parameters = model_cfg.get('find_unused_parameters', False)
        model = DDPPassthrough(model, device_ids=[rank], output_device=rank,
                               find_unused_parameters=find_unused_parameters)  # DDPmodel
    else:
        pass

    Log.info("{} model. Model Initialization Finished!".format(scope))

    model_trainer = Trainer(model=model, cfg=args, is_dist=is_dist, rank=rank, device=device, path_mgr=pth_mgr)

    if 'val' in scope and rank == 0:
        save_args(args, args["exp_dir"])
        save_command(args["exp_dir"], "command.txt")
        open(os.path.join(args["exp_dir"], '%d_parameters_trainable' % num_params_train), 'a').close()
        open(os.path.join(args["exp_dir"], '%d_parameters' % num), 'a').close()

    # run model  # scope: train noval test
    if 'val' in scope:
        model_trainer.train_model()
    elif scope == 'test':  # TODO 这里只打算同test了
        model_trainer.model.eval()
        model_trainer.test_all(args)
    elif args['evaluate_only']:
        pass
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = arg_parse()
    is_dist = args.get("distributed", False)
    if is_dist:
        print("Distributed mode.")
        world_size = torch.cuda.device_count()
        mp.spawn(worker, args=(world_size, args), nprocs=world_size)
    else:
        print("Non-distributed mode.")
        worker(0, None, args)
