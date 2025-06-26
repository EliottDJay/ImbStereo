import os
from glob import glob
import gc

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np


import time
from tqdm import tqdm
from PIL import Image
import cv2

# utils
from utils.check import get_attr_from
from openstereo.stereo_utils.msg_manager import update_image_log
from utils.logger import Logger as Log
from utils.basic import NoOp
from utils.check import tensor2float, tensor2numpy, mkdir, isNum
from openstereo.model_helper.model_helper import convert_state_dict

# data and evaluate
from openstereo.dataloader.get_dataloader import get_dataloader
from openstereo.evaluation.evaluator import OpenStereoEvaluator
from openstereo.evaluation.experiments import MeterDictBestV2, AverageMeterDictV2, get_test_stat

# model helper
from openstereo.modeling import loss as losses
from openstereo.model_helper.optim.basic import get_optimizer
from openstereo.model_helper.l_scheduler import get_scheduler
from openstereo.model_helper.l_scheduler.warmup import get_warmup_scheduler
from openstereo.model_helper.grad_clip import ClipGrad
from openstereo.model_helper.model_helper import convert_state_dict
from openstereo.modeling.sub_models.sub_models.fix_bn import fix_bn

# test
from openstereo.evaluation import DataAnalysis


class BaseTrainer:
    def __init__(self, model: nn.Module = None, cfg: dict = None, is_dist: bool = True,
            rank: int = None, device: torch.device = torch.device('cpu'), path_mgr=None, **kwargs ):
        
        self.msg_mgr = model.msg_mgr # model.msg_mgr  model.module.msg_mgr  # write summary
        self.path_mgr = path_mgr  
        self.model = model
        self.trainer_cfg = cfg['trainer']
        self.data_cfg = cfg['dataset']
        self.load_state_dict_strict = self.trainer_cfg.get('load_state_dict_strict', True)
        self.optimizer = NoOp()  # get_optimizer(self.model, trainer_cfg['optimizer'])
        self.evaluator = NoOp()
        self.warmup_scheduler = NoOp()  # get_warmup_scheduler(trainer_cfg)
        self.clip_gard = NoOp()  # ClipGrad(self.trainer_cfg)
        self.epoch_scheduler = NoOp()
        self.batch_scheduler = NoOp()

        self.mode = self.data_cfg['scope']  # ['val', 'noval', 'test'] 
        self.is_dist = is_dist
        self.rank = rank if is_dist else 0
        self.seed = cfg['seed'] + self.rank
        self.device = torch.device('cuda', rank) if is_dist else device
        self.current_epoch = 0
        self.current_iter = 0
        self.amp = self.trainer_cfg.get('amp', False)
        self.part_amp = self.trainer_cfg.get('part_amp', False)  # compatible with RAFT
        assert (not self.amp) and (not self.part_amp), "cant set both to True"
        if self.amp or self.part_amp:
            Log.info("Using amp partly ot totally")
            self.scaler = torch.cuda.amp.GradScaler()
            if self.part_amp:
                Log.info("make sure that part of the model set to mix_prec")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.bestdict = None
        self.loss_fn = None
        self.build_model()
        self.basic_init(cfg)

    def basic_init(self, cfg):
        Log.info("Trainer Initing ... ")

        scope = self.data_cfg['scope']
        evaluator_cfg = self.trainer_cfg['evaluator']
        metrics = evaluator_cfg.get('metrics', ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3'])
        self.evaluator = OpenStereoEvaluator(metrics, use_np=False)
        self.path_mgr.model_save_type(self.model.get_name(), metrics)  # self.model.get_name() self.model.module.get_name() 

        if 'val' in scope:
            Loss = get_attr_from([losses], cfg['loss']['loss_type'])
            self.loss_fn = Loss(cfg, device=self.device)
            self.train_loader, self.val_loader= get_dataloader(cfg, seed=self.seed)
            if self.is_dist and self.rank == 0 or not self.is_dist:
                self.bestdict = MeterDictBestV2(metric_dict=self.evaluator.metrics)
            # train data loader
            self.optimizer = get_optimizer(self.model, self.trainer_cfg)
            scheduler = get_scheduler(self.trainer_cfg, self.optimizer, len(self.train_loader),
                                      last=self.current_epoch - 1)
            self.clip_gard = ClipGrad(self.trainer_cfg)
            self.warmup_scheduler = get_warmup_scheduler(self.trainer_cfg, self.optimizer)

            Log.info("current epoch is {}".format(self.current_epoch))
            scheduler_cfg = self.trainer_cfg['lr_scheduler']
            if scheduler_cfg.get('on_epoch', True):
                self.on_epoch = True
                self.epoch_scheduler = scheduler
                self.eval_iter = None
            else:
                self.on_epoch = False
                self.eval_iter = self.trainer_cfg.get('eval_iter', 32)  #compatible with like RAFT
                self.batch_scheduler = scheduler
            self.path_list = self._path_obtain(cfg)
            if self.path_list is not None:
                path = self.path_list[0]
                Log.info("Loading ckpt from path: {}".format(path))
                resume = self.load_ckpt(path)  # load the model and if resume, continue to trainer_resume

        if 'test' in scope:
            self.test_loader = get_dataloader(cfg)
            self.path_list = self._path_obtain(cfg)
            # _ = self.load_ckpt(path)  # load the model and if resume, continue to trainer_resume

    def build_model(self, *args, **kwargs):
        # apply fix batch norm
        if self.trainer_cfg.get('fix_bn', False):
            Log.info('fix batch norm')
            self.model = fix_bn(self.model)
        # init parameters
        if self.trainer_cfg.get('init_parameters', False):
            Log.info('init parameters')
            self.model.init_parameters()
        # for some models, we need to set static graph eg: STTR
        if self.is_dist and self.model.model_cfg.get('_set_static_graph', False):  # .module
            self.model._set_static_graph()

    def train_epoch(self):
        # check the warmup
        if self.current_iter > self.trainer_cfg['warmup']['warmup_steps']:
            self.warmup_scheduler = NoOp()

        self.current_epoch += 1
        log_iter = self.trainer_cfg.get('log_iter', 100)
        summary_iter = self.trainer_cfg.get('summary_iter', 100)
        total_training = self.trainer_cfg['epochs']  # max iteration when not use on_epoch
        total_iteration = len(self.train_loader)

        self.model.train()
        Log.info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.train_loader)},"
        )
 
        # for distributed sampler to shuffle data
        # the first sampler is batch sampler and the second is distributed sampler
        if self.is_dist:
            self.train_loader.sampler.set_epoch(self.current_epoch)
        for i, data in enumerate(self.train_loader):
            step = i + 1
            # for max iter training
            if self.current_iter > self.trainer_cfg.get('max_iter', 1e10):
                Log.info('Max iter reached.')
                break
            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                    # ISSUE:
                    #   1. use forward_step will cause torch failed to find unused parameters
                    #   this will cause the model can not sync properly in distributed training
                    batch_inputs = self.model.prepare_inputs(data, device=self.device)
                    outputs = self.model.forward(batch_inputs)
                    loss_inputs = self.loss_fn.prepare_inputs(data, device=self.device)
                    loss_dict = self.loss_fn(outputs, loss_inputs)
                loss = loss_dict['total_loss']
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)  # optional
                self.clip_gard(self.model)
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
            elif self.part_amp:
                batch_inputs = self.model.prepare_inputs(data, device=self.device)
                outputs = self.model.forward(batch_inputs)
                loss_inputs = self.loss_fn.prepare_inputs(data, device=self.device)
                loss_dict = self.loss_fn(outputs, loss_inputs)
                loss = loss_dict['total_loss']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)  # optional
                self.clip_gard(self.model)
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
            else:
                # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                # ISSUE:
                #   1. use forward_step will cause torch failed to find unused parameters
                #   this will cause the model can not sync properly in distributed training
                batch_inputs = self.model.prepare_inputs(data, device=self.device)
                outputs = self.model.forward(batch_inputs)
                loss_inputs = self.loss_fn.prepare_inputs(data, device=self.device)
                loss_dict = self.loss_fn(outputs, loss_inputs)
                loss = loss_dict['total_loss']
                loss.backward()
                self.clip_gard(self.model)
                self.optimizer.step()
            self.current_iter += 1

            # warmup
            self.warmup_scheduler.reloading()
            self.batch_scheduler.step()
            self.warmup_scheduler.dampening()

            # report the Log information: loss learning rate now
            lr = self.optimizer.param_groups[0]['lr']
            if self.rank == 0 and self.current_iter % log_iter == 0 and i != 0:
                loss_stat = self.loss_fn.loss_stat()
                if self.on_epoch:
                    training_stat = "Epoch: [{}/{}]  Iter: [{}/{}]\t".format(self.current_epoch, total_training, step, total_iteration)
                elif not self.on_epoch:
                    training_stat = "Epoch: [{}]  Iter: [{}/{}]\t".format(self.current_epoch, self.current_iter, total_training)
                Log.info("{} Loss stat: ({}) learning rate is {} now".format(training_stat,loss_stat, lr))

            # save the training info into summary file:
            if self.current_iter % summary_iter == 0 and i != 0:
                for name in outputs.keys():
                    if ("preds" in name) and ("pyramid" in name):
                        preds = outputs[name]
                disp_gt = loss_inputs['disp']
                mask = self.loss_fn.mask_generator(disp_gt)
                val_data = {
                'disp_est': preds[-1],
                'disp_gt': disp_gt,
                'mask': mask,  }
                val_res = self.evaluator(val_data)
                avg_test_scalars = AverageMeterDictV2(self.device)
                avg_test_scalars.update(val_res, batch_inputs['batch_size'])
                if self.is_dist:
                    dist.barrier() 
                    avg_test_scalars.reduce_all_metrics()
                avg_scalars = avg_test_scalars.mean()
                if self.rank == 0:
                    scalar_outputs = self.loss_fn.scalar_outputs_update(loss_dict)
                    scalar_outputs.update(avg_scalars)
                    scalar_outputs = tensor2float(scalar_outputs)
                    self.msg_mgr.save_scalars('train', scalar_outputs, self.current_epoch)
                    if not self.msg_mgr.no_img_summary:
                        image_outputs = update_image_log(preds)
                        image_outputs.update({"disp_gt": disp_gt, "left_img": batch_inputs, 
                                              "right_img": batch_inputs, 'disp_est': preds[-1]})
                        self.msg_mgr.save_images('train', image_outputs, self.current_epoch)
                        del image_outputs

            # evaluation with iter pass not the epoch: # if on_epoch
            if self.eval_iter is not None and self.current_iter % self.eval_iter == 0 and i != 0:

                if self.val_loader is not None and self.mode != 'noval':
                    self.val_epoch()

                if self.current_iter % self.trainer_cfg['save_ckpt_freq'] == 0:
                    path_name = self.path_mgr.ckpt_epoch_path(self.current_iter)
                    self.save_ckpt(path_name)  
                    Log.info('Model is saved to path {}.'.format(path_name))
                best_flag = self.bestdict.best_reset
                for metric_name, flag in best_flag.items():
                    if flag:
                        path_name = self.path_mgr.best_model_path_dict[metric_name]
                        self.save_ckpt(path_name)
                        Log.info(
                            'Reach the best result on metric {}, the related best Model is saved to path {}.'.format(
                                metric_name, path_name))
                # save latest
                path_name = self.path_mgr.latest_model_path  # latest model
                self.save_ckpt(path_name)

            # check whether to reach max iteration
            if not self.on_epoch and self.current_iter >= total_training:
                break

        self.warmup_scheduler.reloading()
        self.epoch_scheduler.step()
        self.warmup_scheduler.dampening()

        gc.collect()  # Garbage collection process
        # clear cache
        if next(self.model.parameters()).is_cuda and self.mode != 'noval':
            torch.cuda.empty_cache()

        return None

    def train_model(self):
        Log.info('Training started.')
        total_epoch = self.trainer_cfg.get('epochs', 10) 
        training_continue = True
        while training_continue: 
            self.train_epoch()
            Log.info("One epoch finished...")
            if self.on_epoch:
                if self.current_epoch % self.trainer_cfg['save_ckpt_freq'] == 0:
                    path_name = self.path_mgr.ckpt_epoch_path(self.current_epoch)
                    self.save_ckpt(path_name)
                    Log.info('Model is saved to path {}.'.format(path_name))
                if self.val_loader is not None:
                    self.val_epoch()  # 最后save latest
                best_flag = self.bestdict.best_reset
                for metric_name, flag in best_flag.items():
                    if flag:
                        path_name = self.path_mgr.best_model_path_dict[metric_name]
                        self.save_ckpt(path_name)
                        Log.info('Reach the best result on metric {}, the related best Model is saved to path {}.'.format(metric_name, path_name))
                # save latest
                path_name = self.path_mgr.latest_model_path  # latest model
                self.save_ckpt(path_name)

            if self.current_iter >= self.trainer_cfg.get('max_iter', 1e10):
                Log.info('Max iter reached. Training finished.')
                return
            if self.on_epoch:
                if self.current_epoch >= total_epoch:
                    training_continue = False
            elif not self.on_epoch:
                if self.current_iter >= total_epoch:
                    training_continue = False

        Log.info('Training finished.')
        self.bestdict.expi_over(self.path_mgr.val_file)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()

        # init metrics
        log_iter = self.trainer_cfg.get('test_log', 5)
        avg_test_scalars = AverageMeterDictV2(self.device) 
        img_save_num = 0
        msg_img_save_time = self.trainer_cfg.get('msg_img_save_time', 3)

        Log.info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.val_loader)}," )

        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.val_loader), desc=f'Eval epoch {self.current_epoch}')
        else:
            pbar = NoOp()

        for i, data in enumerate(self.val_loader):
            batch_inputs = self.model.prepare_inputs(data, device=self.device)  
            with autocast(enabled=self.amp):  
                outputs = self.model.forward(batch_inputs)
            loss_inputs = self.loss_fn.prepare_inputs(data, device=self.device)
            disp_gt = loss_inputs['disp']
            for name in outputs.keys():
                if ("preds" in name) and ("pyramid" in name):
                    preds = outputs[name]
            mask = self.loss_fn.mask_generator(disp_gt)
            val_data = {
                'disp_est': preds[-1],
                'disp_gt': disp_gt,
                'mask': mask,}
            
            val_res = self.evaluator(val_data)
            avg_test_scalars.update(val_res, batch_inputs['batch_size'])
            
            if i % log_iter == 0 and i != 0:  
                pbar.update(log_iter)
                pbar.set_postfix({
                    'epe': val_res['EPE'].item(), 'd1': val_res['D1'].item(), 'thres1': val_res['Thres1'].item(),
                    'thres2': val_res['Thres2'].item(), 'thres3': val_res['Thres3'].item()
                })

            if i % msg_img_save_time == 0 and i != 0:
                
                visual_summary = {"disp_gt": disp_gt, "left_img": batch_inputs, "right_img": batch_inputs, 'disp_est': preds[-1]}
                self.msg_mgr.save_images('val_'+str(img_save_num), visual_summary, self.current_epoch)
                img_save_num += 1
        # update rest pbar
        rest_iters = len(self.val_loader) - pbar.n if self.is_dist and self.rank == 0 or not self.is_dist else 0
        pbar.update(rest_iters)
        pbar.close()

        if self.is_dist:
            dist.barrier() 
         
            avg_test_scalars.reduce_all_metrics()

        avg_test_scalars = avg_test_scalars.mean()
        self.msg_mgr.save_scalars('val', avg_test_scalars, self.current_epoch)
        test_stat = get_test_stat(avg_test_scalars)

        if (self.is_dist and self.rank == 0) or not self.is_dist:
            self.bestdict.update(avg_test_scalars, self.current_epoch)
            # write to txt file
            begin_index = 'epoch: %03d\t' % self.current_epoch
            self.bestdict.write2file(self.path_mgr.val_file, avg_test_scalars, begin_index)

        Log.info("In epoch {}: Mean validation scalars: {}".format(self.current_epoch, test_stat))
        # clear cache
        if next(self.model.parameters()).is_cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def test_all(self, cfg):
        self.model.eval()
        path_list = self.path_list
        Log.info("Start testing all ckpt ...")
        for path in path_list:
            Log.info("Loading ckpt from path: {}".format(path))
            _ = self.load_ckpt(path)
            ckpt_name_path = path.split('/')[-1]
            ckpt_name = ckpt_name_path.split('.')[0]

            data_analysis = DataAnalysis(cfg)

            self.test(cfg, ckpt_name, data_analysis=data_analysis)
            if data_analysis.use_representation_analysis:
                self.test(cfg, ckpt_name, data_analysis=data_analysis, repround=False)

            del data_analysis

    @torch.no_grad()
    def test(self, cfg, ckpt_name=None, data_analysis=None, repround=False):
        self.model.eval()
        # init metrics
        dataname = cfg['dataset']['type']
        dataname = dataname.lower()
        test_cfg = cfg.get('test', None)
        assert test_cfg is not None
        if not repround:
            avg_test_scalars = AverageMeterDictV2(self.device)
        output_dir = self.path_mgr.output_dir
        data_anlysis_dir = self.path_mgr.data_analysis_dir
        if ckpt_name is not None:
            output_dir = os.path.join(output_dir, ckpt_name)  
            data_anlysis_dir = os.path.join(data_anlysis_dir, ckpt_name)
        max_disparity = cfg['net'].get('max_disparity', 192)  

        # about path to  save the output img
        if test_cfg.get('save_disp_png', False):
            disp_path = os.path.join(output_dir, dataname, 'disp')
            mkdir(disp_path)
        if test_cfg.get('save_disp_colormap', False):
            cmap_path = os.path.join(output_dir, dataname, 'colormap_jet')
            mkdir(cmap_path)
        if test_cfg.get('errormap', False):
            error_path = os.path.join(output_dir, dataname, 'error')
            mkdir(error_path)
        # aout warmup
        warmup = test_cfg.get('warmup', False)
        if warmup:
            warmup_times = test_cfg.get('warmuptimes', 10)
        
        pbar = tqdm(total=len(self.test_loader), desc='Start testing...')
        pbar_update = len(self.test_loader) // 5
        
        for i, sample in enumerate(self.test_loader):

            ipts = self.model.prepare_inputs(sample, device=self.device)  # .module
            if i == 0 and warmup:
                for _ in range(warmup_times):
                    outputs = self.model.forward(ipts)

            with autocast(enabled=self.amp):
                outputs = self.model.forward(ipts)


            # preds gt pad name
            for name in outputs.keys():
                if ("preds" in name) and ("pyramid" in name):
                    preds = outputs[name]
            pred = preds[-1]  # pred

            left_names = ipts['left_name']  # save name

            top_pad = ipts['top_pad']
            right_pad = ipts['right_pad']
            if 'disp' in sample.keys():
                use_gt = True
                gt = sample['disp']
                gt = gt.to(self.device) 

            else:
                use_gt = False
                gt = np.zeros(pred.shape[0])


            if use_gt:
                if max_disparity is not None:
                    assert isNum(max_disparity)
                    mask = (gt > 0) & (gt < max_disparity)
                elif max_disparity is None:
                    mask = (gt > 0)

                if not repround:
                    val_data = {
                            'disp_est': pred,
                            'disp_gt': gt,
                            'mask': mask,}
                    val_res = self.evaluator(val_data)
                    batch_size = pred.shape[0]
                    avg_test_scalars.update(val_res, batch_size)

                if "sparse_representation" in outputs.keys():
                    representation = outputs["sparse_representation"][-1]
                    index = outputs["sparse_index"][-1]
                elif "representation" in outputs.keys():
                    representation = outputs["representation"][-1]
                    index = None
                else:
                    representation = None
                    index = None
                # data analysis
                if data_analysis is not None:
                    data_analysis(pred, gt, mask, representation=representation, index=index)

            if not repround:

                for disp_one, fn, tp, rp in zip(pred, left_names, top_pad, right_pad):
                    # save result
                    tp = tp.item()
                    rp = rp.item()
                    if tp > 0 and rp > 0:
                        disp_one = disp_one[tp:, :-rp]

                    sub_file = fn.split('/')[:-1]
                    sub_name = fn.split('/')[-1]
                    img = disp_one.squeeze(0).cpu().numpy()
                    img = (img * 256).astype('uint16')

                    if test_cfg.get('save_disp_colormap', False):
                        cmap_save_path = os.path.join(cmap_path, *sub_file)
                        mkdir(cmap_save_path)
                        cmp_path = cmap_save_path + sub_name
                        cv2.imwrite(cmp_path, cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.01), cv2.COLORMAP_JET))
                    if test_cfg.get('save_disp_png', False):
                        img = Image.fromarray(img)
                        disp_save_path = os.path.join(disp_path, *sub_file)
                        mkdir(disp_save_path)
                        im_path = disp_save_path + '/' + sub_name
                        #print(im_path)
                        img.save(im_path)
                    
                if i % pbar_update == 0:
                    pbar.update(pbar_update)
                    #pbar.set_postfix({
                    #    'epe': val_res['EPE'].item(), 'd1': val_res['D1'].item(), 'thres1': val_res['Thres1'].item(),
                    #    'thres2': val_res['Thres2'].item(), 'thres3': val_res['Thres3'].item()
                    #})
            else:
                if i % pbar_update == 0:
                    pbar.update(pbar_update)
        
        rest_iters = len(self.test_loader) - pbar.n
        pbar.update(rest_iters)
        pbar.close()

        #if not repround:
        #    # avg_test_scalars.tofloat() 
        #    avg_test_scalars = avg_test_scalars.mean()
        #    test_stat = get_test_stat(avg_test_scalars)
        #    Log.info("In epoch {}: Mean validation scalars: {}".format(self.current_epoch, test_stat))
        #    begin_index = "test:\t"
        #    if ckpt_name is not None:
       #         begin_index = begin_index + "ckpt: \t " + ckpt_name
        #    MeterDictBestV2.write2file(self.path_mgr.val_file, avg_test_scalars, begin_index)

        data_analysis.save_result(data_anlysis_dir)

        Log.info("evaluation done!")
        gc.collect()


    def save_ckpt(self, name=None, only_model=False):

        # Only save model from master process
        if not self.is_dist or self.rank == 0:
            assert name is not None, "Please specify the name of the checkpoint."
            save_name = name
            state_dict = {
                'model': self.model.state_dict(),
                'loss': self.loss_fn.state_dict(),
                'epoch': self.current_epoch,
                'iter': self.current_iter,
                'best_result': self.bestdict.best_result,
            }
            # for amp
            if self.amp and not only_model:
                state_dict['scaler'] = self.scaler.state_dict()
            # for loss !
            if not isinstance(self.optimizer, NoOp) and not only_model:
                state_dict['optimizer'] = self.optimizer.state_dict()
            if not isinstance(self.batch_scheduler, NoOp) and not only_model:
                Log.info('Batch scheduler saved.')
                state_dict['batch_scheduler'] = self.batch_scheduler.state_dict()
            if not isinstance(self.epoch_scheduler, NoOp) and not only_model:
                Log.info('Epoch scheduler saved.')
                state_dict['epoch_scheduler'] = self.epoch_scheduler.state_dict()
            torch.save(
                state_dict,
                save_name
            )
            # Log.info(f'Model saved to {save_name}')
        if self.is_dist:
            # for distributed training, wait for all processes to finish saving
            dist.barrier()

    def _path_obtain(self, cfg):
        """
        about path:
        load_ckpt load the model from the path, and the path is obtained in 3 ways:
        1. not the test mode and not resume, path = cfg['pretrained_net']
        2. resume is True, path = the lastest ckpt in the exp_dir
        3. test mode, path = cfg['test_net']
        4. path is None, return False
        """
        path_key = 'pretrained_net'
        if 'test' in self.mode:
            path_key = 'test_net'
        path = cfg.get(path_key, None)

        if self.trainer_cfg.get('resume', True):
            ckpts = sorted(glob(cfg["exp_dir"] + '/' + '*.pth'))
            if len(ckpts) == 0:
                raise RuntimeError('=> No checkpoint found while resuming training')
            path = ckpts[-1]

        if path is None:
            return None
        
        if not path.endswith('.pth'):
            assert os.path.isdir(path)
            path_list = glob(os.path.join(path, '*.pth'))
        elif path.endswith('.pth'):
            if 'replace' in path:
                path_list = []
                for i in self.evaluator.metrics:
                    # ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']
                    best_path = path.replace('replace', i)
                    path_list.append(best_path)
            else:
                path_list = [path]

        assert len(path_list) > 0, 'No checkpoint found.'

        return path_list

    def load_ckpt(self, path):

        if not os.path.exists(path):
            Log.warn(f"Checkpoint {path} not found.")
            return False
        
        map_location = {'cuda:0': f'cuda:{self.rank}'} if self.is_dist else self.device
        checkpoint = torch.load(path, map_location=map_location)
        # convert state dict for or not for distributed training
        model_state_dict = convert_state_dict(checkpoint['model'], is_dist=self.is_dist) 
        # fix size mismatch error
        ignore_keys = []
        for k, v in model_state_dict.items():
            if k in self.model.state_dict().keys():
                v_dst = self.model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    Log.info(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            ))
                    

        self.model.load_state_dict(model_state_dict, strict=self.load_state_dict_strict)
        Log.info(f'Model loaded from {path}')

        # for amp
        if self.amp:
            if 'scaler' not in checkpoint:
                Log.warn('Loaded model is not amp compatible.')
            else:
                self.scaler.load_state_dict(checkpoint['scaler'])

        # skip loading optimizer and scheduler if resume is False
        #if not self.trainer_cfg.get('resume', True) or not self.load_state_dict_strict:
        if not self.trainer_cfg.get('resume', True):
            return False
        
        # load loss
        if not self.trainer_cfg.get('loss_reset', False):
            loss_dict = checkpoint['loss']
            dict_len = len(loss_dict)
            if dict_len > 0: 
                loss_state_dict = convert_state_dict(checkpoint['loss'], is_dist=self.is_dist)  
                self.loss_fn.load_state_dict(loss_state_dict, strict=self.load_state_dict_strict)

        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
        self.msg_mgr.iteration = self.current_iter
        Log.info("Resume, epoch now is {}, and iteration now is {}".format(self.current_epoch, self.current_iter))


        self.bestdict.best_result = checkpoint['best_result']
        lr_before = self.optimizer.param_groups[0]['lr']
        try:
            # load optimizer
            if not self.trainer_cfg.get('optimizer_reset', False):
                Log.info('Optimizer loaded .')
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        except KeyError:
            Log.warn('Optimizer not loaded.')
        lr_after = self.optimizer.param_groups[0]['lr']
        Log.info(f'Optimizer lr changed from {lr_before} to {lr_after}.')

        lr_before = self.optimizer.param_groups[0]['lr']
        last_epoch_before = self.epoch_scheduler.last_epoch
        try:
            # load optimizer
            #if not self.trainer_cfg.get('optimizer_reset', False):
                #Log.info('Optimizer loaded .')
                #self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load scheduler
            if not self.trainer_cfg.get('scheduler_reset', False):
                Log.info('Scheduler loaded .')
                if not isinstance(self.batch_scheduler, NoOp):
                    self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
                if not isinstance(self.epoch_scheduler, NoOp):
                    self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])

            # load warmup scheduler
            if not self.trainer_cfg.get('warmup_reset', False):
                Log.info('Warmup loaded .')
                self.warmup_scheduler.last_step = self.current_iter - 1 

        except KeyError:
            Log.warn('scheduler not loaded.')

        lr_after = self.optimizer.param_groups[0]['lr']
        last_epoch_after = self.epoch_scheduler.last_epoch
        Log.info(f'Optimizer lr changed from {lr_before} to {lr_after}.')
        Log.info(f'Epoch scheduler last epoch changed from {last_epoch_before} to {last_epoch_after}.')

        if not isinstance(self.warmup_scheduler, NoOp):
            self.warmup_scheduler.last_step = self.current_iter

        # dataset reload
        self.train_loader.dataset.epoch_pass(set_stat=True)
        # Log.info("lenth of train_loader: {} ".format(len(self.train_loader)))
        for i in range(0, self.current_epoch):
            epoch_now = i + 1
            if self.is_dist:
                self.train_loader.sampler.sampler.set_epoch(epoch_now)
            for j, _ in enumerate(self.train_loader):
                # Log.info("iter: {}".format(j))
                pass
            Log.info('pass {} epoch'.format(epoch_now))
        self.train_loader.dataset.epoch_pass(set_stat=False)

        return True # resume


