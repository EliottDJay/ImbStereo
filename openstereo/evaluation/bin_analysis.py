import os
import copy
import torch
import numpy as np
import torch.nn.functional as F

import xlsxwriter as xw

from utils.logger import Logger as Log
from utils.check import mkdir, isNum, is_list


# about matrix

def epe(pred, gt):
    E = torch.abs(pred - gt)
    return E

def d1(pred, gt):
    E = torch.abs(pred - gt)
    err_mask = (E > 3) & (E / torch.abs(gt) > 0.05)
    return err_mask

def thres1(pred, gt):
    threshold = 1
    E = torch.abs(pred - gt)
    err_mask = E > threshold
    return err_mask

def thres2(pred, gt):
    threshold = 2
    E = torch.abs(pred - gt)
    err_mask = E > threshold
    return err_mask

def thres3(pred, gt):
    threshold = 3
    E = torch.abs(pred - gt)
    err_mask = E > threshold
    return err_mask


METRICS = {
    # EPE metric (Average Endpoint Error)
    'EPE': epe,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'D1': d1,
    'Thres1': thres1,  
    'Thres2': thres2,
    'Thres3': thres3,
}


class BinTracker(object):
    def __init__(self, cfg, metrics=None, use_gt=True):
        self.metrics = metrics
        self.max_disparity = cfg['net'].get('max_disparity', None) 

        test_cfg = cfg['test']
        analysis_cfg = test_cfg['data_analysis']
        bin_tracker_cfg = analysis_cfg['bin_tracker']
        use_flag = bin_tracker_cfg.get('use', False)
        assert use_flag, "BinTracker is not used"

        # cfg details
        self.tracker_cfg = bin_tracker_cfg
        self.bin_size = bin_tracker_cfg.get('bin_size', 1) #1
        self.tail_include = bin_tracker_cfg.get('tail_include', False)  
        
        if metrics is None:
            self.metrics = cfg['trainer']['evaluator'].get('metric', ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3'])
        else:
            self.metrics = metrics

        self.datadic = {}


    def __call__(self, pred, gt, mask, **kwargs):
        assert len(gt.size()) == 3 and len(pred.size()) == 3  # [B, H, W] Typically B=1

        if self.max_disparity is not None and not self.tail_include:
            bin_set = np.arange(0, self.max_disparity, self.bin_size)  # (0, 192, 1) --> len() = 192

        res_dict = {}  # result dictionary
        pred = pred[mask]
        gt = gt[mask]
        for m in self.metrics:
            if m not in METRICS:
                raise ValueError("Unknown metric: {}".format(m))
            else:
                metric_func = METRICS[m]
                res_dict[m] = metric_func(pred, gt) 

        for i in bin_set:
            rangefloor = i
            rangeceil = i + self.bin_size
            floor_str = "{floor:.3f}".format(floor=rangefloor)
            ceil_str = "{ceil:.3f}".format(ceil=rangeceil)
            tag_indx = "disprange_" + floor_str + '_to_' + ceil_str

            for m in self.metrics:
                res = res_dict[m]
                total_pixels = (gt > rangefloor) & (gt <= rangeceil)
                valid_res = res[total_pixels]
                tag_now = m + '_' + tag_indx
                
                if tag_now not in self.datadic.keys():
                    self.datadic[tag_now] = torch.sum(valid_res).item()
                    if m == self.metrics[0]:
                        # record pixel num in every bin
                        num_tag = 'pixelnum' + '_' + tag_indx
                        self.datadic[num_tag] = torch.sum(total_pixels).item()
                elif tag_now in self.datadic.keys():
                    self.datadic[tag_now] += torch.sum(valid_res).item()
                    if m == self.metrics[0]:
                        num_tag = 'pixelnum' + '_' + tag_indx
                        self.datadic[num_tag] += torch.sum(total_pixels).item()
    
    def save_result(self, path, extra_name=None):

        if self.max_disparity is not None and not self.tail_include:
            bin_set = np.arange(0, self.max_disparity, self.bin_size)  
            length = len(bin_set)

            save_dict = {}
            statistic = {}
            if extra_name is None:
                bin_tracker_path = os.path.join(path, 'bin_tracker')
            else:
                bin_tracker_path = os.path.join(path, 'bin_tracker', extra_name)
            mkdir(bin_tracker_path)
            for m in self.metrics:
                name = "{}_from_0_to_{}_binsize_{}.npy".format(m, self.max_disparity, self.bin_size)
                path_now = os.path.join(bin_tracker_path, name)
                save_dict[m] = path_now
                statistic[m] = np.zeros([length])
                
            name = "{}_from_0_to_{}_binsize_{}.npy".format('pixelnum', self.max_disparity, self.bin_size)
            path_now = os.path.join(bin_tracker_path, name)
            save_dict['pixelnum'] = path_now
            statistic['pixelnum'] = np.zeros([length])
            xname = "{}_from_0_to_{}_binsize_{}.xlsx".format('data_all', self.max_disparity, self.bin_size)
            xpath_now = os.path.join(bin_tracker_path, xname)
            save_dict['xlsx'] = xpath_now
        elif self.tail_include:
            raise NotImplementedError("Tail include is not implemented yet")

        for i, element in enumerate(bin_set):
            rangefloor = element
            rangeceil = element + self.bin_size
            floor_str = "{floor:.3f}".format(floor=rangefloor)
            ceil_str = "{ceil:.3f}".format(ceil=rangeceil)
            tag_indx = "disprange_" + floor_str + '_to_' + ceil_str
            pixel_tag = 'pixelnum'
            num_tag = pixel_tag + '_' + tag_indx
            num_now = self.datadic[num_tag]
            statistic['pixelnum'][i] = num_now
            for m in self.metrics:
                tag_now = m + '_' + tag_indx
                sum_num = self.datadic[tag_now]
                mean_now = sum_num / (statistic['pixelnum'][i] + 1e-06)
                statistic[m][i] = mean_now

        for m in self.metrics:
            np.save(save_dict[m], statistic[m])
        np.save(save_dict['pixelnum'], statistic['pixelnum'])

        # xlsx cfg
        # xlsx_bin_include = self.tracker_cfg.get('xlsx_bin_include', [25, 50])
        xlsx_bin_include = [25, 50]
        pixel_statistic_file = self.tracker_cfg.get('pixel_statistic_file', None)
        assert pixel_statistic_file is not None, "we should exclude the None bin by the statistic of pixelnum"
        pixel_statistic = np.load(pixel_statistic_file)
        assert len(pixel_statistic)  == self.max_disparity 

        if not is_list(xlsx_bin_include):
            xlsx_bin_include = [xlsx_bin_include]

        workbook = xw.Workbook(save_dict['xlsx'])
        row_index = 1
        worksheet1 = workbook.add_worksheet("sheet1")

        index_l = []
        index_r = []

        for bin_range in xlsx_bin_include:
            buffer_index_l = np.arange(0, self.max_disparity, bin_range)
            buffer_index_r = buffer_index_l + bin_range
            if buffer_index_r[-1] > self.max_disparity:
                buffer_index_r[-1] = self.max_disparity
            index_l = index_l + list(buffer_index_l)
            index_r = index_r + list(buffer_index_r)

        index_r_copy = copy.deepcopy(index_r)
        r_set = list(set(index_r_copy))
        r_set = sorted(r_set)
        index_clude = r_set[1:] 

        insertData = ["name_or_metrix"]
        for l in range(len(index_l)):

            head = "disp{}-{}".format(index_l[l], index_r[l])
            insertData.append(head)
        for i in range(len(index_clude)):
            head = "disp{}-{}".format(0, index_clude[i])
            insertData.append(head)

        row = 'A' + str(row_index)
        worksheet1.write_row(row, insertData)
        row_index = row_index + 1
        for m in self.metrics:
            stat_now = statistic[m]

            insertData = [m]
            for l in range(len(index_l)):
                num_stast = pixel_statistic[index_l[l]:index_r[l]]
                valid_stast = num_stast > 0
                res_range = stat_now[index_l[l]:index_r[l]]
                valid_res = res_range[valid_stast]
                if len(valid_res) == 0:
                    valid_res = np.array([0])
                insertData.append(np.mean(valid_res))

            for l in range(len(index_clude)):
                num_stast = pixel_statistic[:index_clude[l]]
                valid_stast = num_stast > 0
                res_range = stat_now[:index_clude[l]]
                valid_res = res_range[valid_stast]
                if len(valid_res) == 0:
                    valid_res = np.array([0])
                insertData.append(np.mean(valid_res))

            row = 'A' + str(row_index)
            worksheet1.write_row(row, insertData)
            row_index = row_index + 1  

        insertData_1 = ["total_bin"]
        insertData_2 = ["valid_bin"]
        insertData_3 = ["valid_pixel_num"]
        insertData_4 = ["valid_pixel_percent"]

        total_pixel = np.sum(pixel_statistic)
        
        for l in range(len(index_l)):
            num_stast = pixel_statistic[index_l[l]:index_r[l]]
            valid_stast = num_stast > 0
            valid_bin_num = np.sum(valid_stast)
            insertData_2.append(valid_bin_num)

            bin_num = index_r[l] - index_l[l]
            insertData_1.append(bin_num)

            valid_pixel_num = np.sum(num_stast)
            insertData_3.append(valid_pixel_num)

            insertData_4.append(valid_pixel_num / total_pixel)

        for i in range(len(index_clude)):
            num_stast = pixel_statistic[:index_clude[i]]
            valid_stast = num_stast > 0
            valid_bin_num = np.sum(valid_stast)
            insertData_2.append(valid_bin_num)

            bin_num = index_clude[i]
            insertData_1.append(bin_num)

            valid_pixel_num = np.sum(num_stast)
            insertData_3.append(valid_pixel_num)

            insertData_4.append(valid_pixel_num / total_pixel)
        
        row = 'A' + str(row_index)
        worksheet1.write_row(row, insertData_1)
        row_index = row_index + 1  

        row = 'A' + str(row_index)
        worksheet1.write_row(row, insertData_2)
        row_index = row_index + 1  

        row = 'A' + str(row_index)
        worksheet1.write_row(row, insertData_3)
        row_index = row_index + 1  

        row = 'A' + str(row_index)
        worksheet1.write_row(row, insertData_4)
        row_index = row_index + 1  

        workbook.close()







            

