import torch
import numpy as np
import random
import copy

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# utils
from utils.logger import Logger as Log
from utils.distributed import is_distributed

# dataset
from .stereodataset import StereoDataset

# data augmentation
from .trans_and_aug import build_transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(cfg, seed=None):
    cfg_dataset = cfg["dataset"]
    loss_cfg = cfg["loss"]
    mode = cfg["dataset"]["scope"]  # scope: train noval test/ stage：train val test
    disp = cfg["net"].get('max_disparity', 192)

    if mode == 'test':
        test_loader = build_dataloader(cfg_dataset, mode, seed=seed, disp=disp)
        Log.info("Get loader Done for Test...")
        return test_loader
    train_loader = build_dataloader(cfg_dataset, "train", seed=seed, loss_cfg=loss_cfg, disp=disp)
    if mode == 'noval':
        Log.info("Get loader Done and there is no validation when training...")
        return train_loader, None
    if mode == 'val':
        val_loader = build_dataloader(cfg_dataset, "val", seed=seed, disp=disp)
        Log.info("Get Both train and validation loader Done...")
        return train_loader, val_loader
    
    raise NotImplementedError('Scope:{} is not valid.'.format(mode))


def build_dataloader(dataset_cfg, stage, seed=None, loss_cfg=None, disp=192):

    mode_cfg = copy.deepcopy(dataset_cfg[stage])
    cfg = copy.deepcopy(dataset_cfg)
    mode = dataset_cfg['scope']

    img_mean = cfg['mean']
    img_std = cfg['std']

    dataset_name = cfg.get('type', 'SceneFlow')
    workers = cfg.get("workers", 0)
    batch_size = mode_cfg.get("batch_size", 1)
    shuffle = mode_cfg.get("shuffle", False)
    memory_pin = mode_cfg.get("pin_memory", True)
    drop = mode_cfg.get("drop_last", False)
    Log.info("Collecting {} dataset for {} with {} batchsize!".format(dataset_name, stage, batch_size))

    transform = build_transform(mode_cfg, img_mean, img_std, stage=stage)

    stereodataset = StereoDataset(mode_cfg, dataset_name, mode, transform=transform, seed=seed, max_disp=disp, loss_cfg=loss_cfg)

    if is_distributed():
        sampler = DistributedSampler(stereodataset, shuffle=shuffle)
    else:
        sampler = None

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    loader = DataLoader(
            dataset=stereodataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=workers,
            # collate_fn=stereodataset.collect_fn,
            pin_memory=memory_pin,
            drop_last=drop,
            worker_init_fn=seed_worker,
            generator=g
        )
    
    return loader
