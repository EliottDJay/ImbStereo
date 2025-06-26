import os
import copy
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils.logger import Logger as Log
from utils.check import mkdir, isNum, is_list

class ConRepTracker(object):
    def __init__(self, cfg, metrics=None, use_gt=True):
        self.max_disparity = cfg['net'].get('max_disparity', None) 

        test_cfg = cfg['test']
        analysis_cfg = test_cfg['data_analysis']
        conrep_tracker_cfg = analysis_cfg['vrepresentation']
        
        use_flag = conrep_tracker_cfg.get('use', False)
        assert use_flag, "conformal Representation is not used"

        self.feature_dim = conrep_tracker_cfg.get('representation_dim', 192)
        self.sparese_representation = conrep_tracker_cfg.get('sparse', False)
        self.bin_scale = conrep_tracker_cfg.get('bin_scale', 1.0)

        self.save_per_img = conrep_tracker_cfg.get('save_per_img', False)


        # information for save:
        dataname = cfg['dataset'].get('type', 'SceneFlow')
        self.dataname = dataname.lower()


        # statistic for per image
        if self.save_per_img:
            self.mean_per_img = torch.zeros(self.max_disparity, self.feature_dim).cuda()
            self.var_per_img = torch.zeros(self.max_disparity, self.feature_dim).cuda()
            self.cor_per_img = torch.zeros(self.max_disparity, self.feature_dim).cuda()
        self.cor_for_all = torch.zeros(self.max_disparity).cuda()
        self.num_samples_img = torch.zeros(self.max_disparity).cuda()

        # statistic for all
        self.num_samples_tracked = torch.zeros(self.max_disparity).cuda()
        self.mean_for_all = torch.zeros(self.max_disparity, self.feature_dim).cuda()
        self.var_for_all = torch.zeros(self.max_disparity, self.feature_dim).cuda()
        self.cor_all = torch.zeros(self.max_disparity).cuda()

        self.round = 1  # [1, 2]


    def __call__(self, representation, disp, mask, sparse_ind=None, **kwargs):
        
        if self.sparese_representation:
            assert sparse_ind is not None, "sparse index is not provided"
            representation = self.sparse_volume_recovery(representation, sparse_ind)

        unrenewed_bin = [-1]
        if disp.size(-1) != representation.size(-1):
            disp = disp.unsqueeze(1)
            disp = F.interpolate(disp, size=(representation.size(-2), representation.size(-1)), mode='nearest') # TODO 'nearest'?
            disp = disp.squeeze(1)

        # print(disp.shape, representation.shape)
        # torch.Size([1, 128, 240]) torch.Size([1, 48, 128, 240])

        representation = representation.permute(0, 2, 3, 1).contiguous().view(-1, self.feature_dim)
        #mask = (disp > 0) & (disp < self.max_disparity)
        disp = disp.squeeze(1).view(-1)  # [HxW 1]
        mask = (disp > 0) & (disp < self.max_disparity)
        bins = self.get_bin_idx(disp, mask)
        #print(bins.shape, representation.shape, mask.shape)
        for b in torch.unique(bins):
            if b not in unrenewed_bin:  # 在demo中测试 是行得通的
                curr_feats = representation[bins.eq(b)]  # [nums, fea_dim]

                if self.round == 1:
                    curr_num_sample = curr_feats.size(0)
                    curr_mean = torch.mean(curr_feats, 0)
                    if self.save_per_img:
                        bias = curr_feats - curr_mean
                        bias = torch.pow(bias, 2)  
                        if curr_num_sample > 1:
                            var = torch.sum(bias, dim=0)/(curr_num_sample-1)
                        elif curr_num_sample == 1:
                            var = torch.sum(bias, dim=0) / 1
                        else:
                            Log.error("current sample numbers must > 0 !")

                    proj_vec = curr_mean.clone()
                    proj_vec = proj_vec.unsqueeze(0)
                    cor = (curr_feats/(torch.norm(curr_feats, 2, dim=1, keepdim=True)+1e-06)) * (proj_vec /(torch.norm(proj_vec, 2, dim=1, keepdim=True)+1e-06))
                    cor = torch.sum(cor, dim=1)
                    cor_mean = cor.mean(dim=0) 

                    if self.save_per_img:
                        self.num_samples_img[b.item()] = curr_num_sample
                        self.mean_per_img[b.item()] = curr_mean
                        self.var_per_img[b.item()] = var
                        self.cor_per_img[b.item()] = cor_mean

                    self.num_samples_tracked[b.item()] += curr_num_sample
                    factor = curr_num_sample / self.num_samples_tracked[b.item()]
                    past_cor = self.cor_for_all[b.item()]

                    cor_now = cor_mean * factor
                    past_cor_factor = past_cor * (1-factor)
                    cor_all_plus = cor_now + past_cor_factor
                    past_mean = self.mean_for_all[b.item()]
                    mean_now = curr_mean * factor
                    past_mean_factor = past_mean * (1-factor)
                    mean_all_plus = mean_now + past_mean_factor

                    self.mean_for_all[b.item()] = mean_all_plus 
                    self.cor_for_all[b.item()] = cor_all_plus  

                elif self.round == 2:
                    curr_mean = self.mean_for_all[b.item()]
                    proj_vec = curr_mean.clone()
                    proj_vec = proj_vec.unsqueeze(0)
                    cor = (curr_feats / (torch.norm(curr_feats, 2, dim=1, keepdim=True) + 1e-06)) * (
                            proj_vec / (torch.norm(proj_vec, 2, dim=1, keepdim=True) + 1e-06))  
                    cor = torch.sum(cor, dim=1)  
                    cor_sum = torch.sum(cor)
                    bias = curr_feats - curr_mean
                    bias = torch.pow(bias, 2) 
                    vars_sum = torch.sum(bias, dim=0) 
                    self.var_for_all[b.item()] += vars_sum
                    self.cor_all[b.item()] += cor_sum

        if self.save_per_img and self.round == 1:
            pass

    def save_result(self, path):
        represenntation_path = os.path.join(path, "representation")
        mkdir(represenntation_path)

        if self.round == 1:
            np.save(os.path.join(represenntation_path, 'mean_for_all.npy'), self.mean_for_all.cpu().numpy())
            np.save(os.path.join(represenntation_path, 'cor_for_all.npy'), self.cor_for_all.cpu().numpy())
            np.save(os.path.join(represenntation_path, 'num_samples_tracked.npy'), self.num_samples_tracked.cpu().numpy())

        elif self.round == 2:
            for i in range(self.max_disparity):
                self.var_for_all[i] = self.var_for_all[i]/(self.num_samples_tracked[i] - 1)
                self.cor_all[i] = self.cor_all[i] / self.num_samples_tracked[i]

            np.save(os.path.join(represenntation_path, 'var_for_all.npy'), self.var_for_all.cpu().numpy())
            np.save(os.path.join(represenntation_path, 'cor_all.npy'), self.cor_all.cpu().numpy())

            Log.info("over!")

        self.round += 1

        if self.round == 3:
            pass

    def _save2png(self, data, path, name=None):
        valid_indices = np.where(data >= 0)[0]
        valid_data = data[valid_indices]
        plt.figure(figsize=(10, 6))
        plt.plot(valid_indices, valid_data, color='blue', linewidth=1, marker='o', markersize=2, alpha=0.8)  
        if name is not None:
            plt.title(name)
        plt.xlabel('disparity bin')
        plt.ylabel('Cor')
        plt.grid()
        plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')  # Add horizontal line at y=0
        plt.savefig(path, format='png') 
        # Close the plot to prevent it from displaying
        plt.close() 


    def sparse_volume_recovery(self, sparse_volume, indx):
        assert len(sparse_volume.size()) == 4  # torch.Size([24, 24, 64, 128]) B=1 D H W
        assert len(indx.size()) == 4  # torch.Size([24, 24, 64, 128])
        b, d, h, w = sparse_volume.shape
        vdevice = sparse_volume.device
        full_volume = torch.ones((b, self.feature_dim, h, w))
        full_volume = full_volume.to(vdevice)
        lower_bound = torch.min(sparse_volume, dim=1, keepdim=True)
        #  TypeError: only integer tensors of a single element can be converted to an index
        bound_volume = lower_bound.values * full_volume
        ind = indx.clone()
        ind = ind.long()
        # Log.info("{}".format(type(ind)))
        ind = ind.to(vdevice)
        spase_new = sparse_volume.clone()
        full = bound_volume.scatter_(1, ind, spase_new)
        del spase_new, ind
        return full
    
    def get_bin_idx(self, label, mask):
        # label_bin = torch.max(torch.min(full_labels * self.label_scale, self.bucket_num - 1), self.bucket_start)
        label_bin = torch.floor(label * self.bin_scale).int()  

        label_bin = torch.masked_fill(label_bin, ~mask, -1)  
        return label_bin
    
    def sparse_volume_fill(self, sparse_volume, indx):
        assert len(sparse_volume.size()) == 4  # torch.Size([24, 24, 64, 128])
        assert len(indx.size()) == 4  # torch.Size([24, 24, 64, 128])
        b, d, h, w = sparse_volume.shape
        vdevice = sparse_volume.device
        sparse_volume = F.softmax(sparse_volume, dim=1)
        full_volume = torch.ones((b, self.feature_dim, h, w))  # torch.Size([1, 48, 64, 128])
        spase_new = sparse_volume.clone()
        ind = indx.clone()
        ind = ind.long()
        ind = ind.to(vdevice)
        full_volume = full_volume.to(vdevice)
        full = full_volume.scatter_(1, ind, spase_new)
        return full
