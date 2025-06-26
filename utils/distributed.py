import torch
import torch.nn as nn
import subprocess
import sys
import os
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP


def only_rank0(func):
    def wrapper(*f_args, **f_kwargs):
        rank = get_rank()
        if rank == 0:
            func(*f_args, **f_kwargs)
    return wrapper


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_distributed():
    return torch.distributed.is_initialized()


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def all_reduce_bysize(tensor):
    tensor = tensor.clone().detach()
    if is_distributed():
        dist.barrier()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        wordsize = get_world_size()
        tensor = tensor/wordsize
    return tensor


# OpenStereo
#
# by
def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device, **kwargs)
    return module


# U2PL: Semi-Supervised Semantic Segmentation Using Unreliable Pseudo Labels
# https://github.com/Haochen-Wang409/U2PL
# by Haochen Wang
def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size