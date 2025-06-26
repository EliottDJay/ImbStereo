"""
# OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline
The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `openstereo/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_val(model)
"""
from abc import abstractmethod, ABCMeta

import torch
from torch import nn

# utils
from openstereo.stereo_utils.msg_manager import get_msg_mgr
from utils.logger import Logger as Log

from openstereo.modeling.models.base_trainer import BaseTrainer


class BaseModel(nn.Module):
    """
    The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    
    """ 
    def __init__(self, cfg, **kwargs):
        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr(cfg)  # build a summary writer
        # self.cfg = cfg
        # self.model_cfg = cfg['net']
        # self.model_name = self.model_cfg['method']
        # self.max_disp = self.model_cfg.get('max_disparity', 192)
        # self.DispProcessor = None
        # self.CostProcessor = None
        # self.Backbone = None
        # self.loss_fn = None
        # self.build_network(cfg)
        # self.build_loss_fn(cfg)
        self.Trainer = BaseTrainer
        
    def build_network(self, cfg):
        """Build your network here."""
        raise NotImplementedError

    def build_loss_fn(self):
        """Build your optimizer here."""
        raise NotImplementedError

    def prepare_inputs(self, inputs, device=None, **kwargs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError
   
    def forward(self, inputs):
        """Forward the network."""
        raise NotImplementedError

    def forward_step(self, inputs):
        """Forward the network for one step."""
        raise NotImplementedError

    def init_parameters(self):
        pass
        """for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)"""
        
    def get_name(self):
        return 'net'
