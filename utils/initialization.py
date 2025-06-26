import os
from datetime import datetime

import random
import numpy as np
import torch

from .check import mkdir

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False  
        torch.backends.cudnn.benchmark = False  
    else:  
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def path_checking(args, abspath):


    if args.get("config", False):
        checkpoint_root = os.path.join(abspath, 'checkpoint')
        yaml_path = args["config"]
        path_split = yaml_path.split('/')
       
        args["exp_dir"] = os.path.join(checkpoint_root, *path_split[-4:])   
        sub_name = path_split[-3]
    else:
        checkpoint_root = os.path.join(abspath, 'checkpoint')
        args["exp_dir"] = os.path.join(checkpoint_root, 'expitest')
        sub_name = "test"
    exp_path = args['exp_dir']  
    args["model_path"] = os.path.join(exp_path, "model")  
    args["summary_path"] = os.path.join(exp_path, "summary")
    log_file = os.path.join(exp_path, 'log')  
    mkdir(log_file)
    mkdir(args["model_path"])
    mkdir(args["summary_path"])
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args["log_file"] = os.path.join(exp_path, 'log', sub_name + current_time + ".log")


class PathManager:
    def __init__(self, args, abspath):
        self.exp_dir = None
        self.model_path = None
        self.summary_path = None
        self.log_file = None
        self.val_file = None
        self.test_file_sign = None 

        self.output_dir = None
        self.data_analysis_dir = None

        self._path_init(args, abspath)  

    def _path_init(self, args, abspath):
        if args.get("config", False):
            checkpoint_root = os.path.join(abspath, 'checkpoint')
            yaml_path = args["config"]

            path_split = yaml_path.split('/')  

            self.exp_dir = os.path.join(checkpoint_root, *path_split[-5:-1])  

            sub_name = path_split[-3]
            self.test_file_sign = sub_name
        else:
            checkpoint_root = os.path.join(abspath, 'checkpoint')
            self.exp_dir = os.path.join(checkpoint_root, 'expitest')
            sub_name = "test"
            self.test_file_sign = "test"

        self.model_path = os.path.join(self.exp_dir, "model")
        self.summary_path = os.path.join(self.exp_dir, "summary")
        self.output_dir = os.path.join(self.exp_dir, "output")
        self.data_analysis_dir = os.path.join(self.exp_dir, "data_analysis")
        log_file = os.path.join(self.exp_dir, 'log') 
        mkdir(log_file)
        mkdir(self.model_path)
        mkdir(self.summary_path)
        mkdir(self.output_dir)
        mkdir(self.data_analysis_dir)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.exp_dir, 'log', sub_name + current_time + ".log")

        self.val_file = os.path.join(self.exp_dir, 'val_results.txt')

    def add2args(self, args):
        args['exp_dir'] = self.exp_dir
        args["model_path"] = self.model_path
        args["summary_path"] = self.summary_path
        args["log_file"] = self.log_file

    def model_save_type(self, model_name, metric_dict):
        # latest model
        self.latest_model_path = os.path.join(self.exp_dir, model_name + '_latest.pth')
        # best list
        self.best_model_path_dict = {}
        for i in metric_dict:
            self.best_model_path_dict[i] = os.path.join(self.exp_dir, model_name + '_' + i + '_best.pth')

        # model_path
        self.model_path = os.path.join(self.model_path, model_name + '_epoch_')

    def ckpt_epoch_path(self, epoch):
        return self.model_path + '{:0>3d}.pth'.format(epoch)