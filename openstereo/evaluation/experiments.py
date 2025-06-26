from __future__ import print_function, division
import torch
import os
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import copy

from utils.basic import make_iterative_func
from utils.check import check_allfloat, check_alltenser, tensor2float
from utils.logger import Logger as Log


def get_test_stat(dict):
    stat = []
    for tag, values in dict.items():
        if isinstance(values, list) and (len(values) > 1):
            for i in range(len(values)):
                tag_now = tag+ '_' +str(i)
                one_stat = "{}:{:.5f}\t".format(tag_now, values[i])
                stat.append(one_stat)
        elif (isinstance(values, list)) and (len(values) == 1):
            one_stat = "{}:{:.5f}\t".format(tag, values[0])
            stat.append(one_stat)
        elif not (isinstance(values, list)):
            one_stat = "{}:{:.5f}\t".format(tag, values)
            stat.append(one_stat)
        else:
            Log.error('cant create the test stat!')
            exit(1)

    test_stats = ''.join(stat)

    return test_stats


class AverageMeter(object):
    """ Computes ans stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict(object):
    # GPU Version
    def __init__(self):
        self.input_type = "sum"
        self.data = None
        self.count = 0
        self.device = None

    def update(self, x, n=1):
        # check_allfloat(x)
        check_alltenser(x)
        self.count += n
        if self.data is None:
            self.data = copy.deepcopy(x) 
        else:
            for k1, v1 in x.items():
                if isinstance(v1, torch.Tensor):
                    self.data[k1] += v1
                    if self.device is None:
                        self.device = v1.device
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                        if self.device is None:
                            self.device = v2.device
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def reduce_all_metrics(self):
        for k1, v1 in self.data.items():
            if isinstance(v1, torch.Tensor):
                dist.barrier()
                dist.all_reduce(v1, op=dist.ReduceOp.SUM)
            elif isinstance(v1, tuple) or isinstance(v1, list):
                for idx, v2 in enumerate(v1):
                    dist.barrier()
                    dist.all_reduce(v2, op=dist.ReduceOp.SUM)
            else:
                assert NotImplementedError("error input type for update AvgMeterDict")
        self._reduce_count()

    def _reduce_count(self):
        count_sum = torch.tensor(self.count, device=self.device)
        dist.barrier()
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        self.count = count_sum.item()  # 

    def tofloat(self):
        for k1, v1 in self.data.items():
            if isinstance(v1, torch.Tensor):
                self.data[k1] = v1.item()
            elif isinstance(v1, tuple) or isinstance(v1, list):
                for idx, v2 in enumerate(v1):
                    self.data[k1][idx] = v2.item()
            else:
                assert NotImplementedError("error input type for update AvgMeterDict")
        check_allfloat(self.data)

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)


class AverageMeterDictV2(object):
    # GPU Version
    def __init__(self, device):
        self.input_type = "sum"
        self.data = None
        self.count = 0
        self.device = device

    def update(self, x, n):
        x = tensor2float(x)
        check_allfloat(x)
        if isinstance(n, torch.Tensor):
            n = float(n.item())
        self.count += n
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def reduce_all_metrics(self):
        data_for_reduce = copy.deepcopy(self.data)
        for k1, v1 in self.data.items():
            if isinstance(v1, float):
                new_tensor = torch.tensor(float, device=self.device)
                dist.barrier()
                dist.all_reduce(new_tensor, op=dist.ReduceOp.SUM)
                data_for_reduce[k1] = float(new_tensor.item())
            elif isinstance(v1, tuple) or isinstance(v1, list):
                for idx, v2 in enumerate(v1):
                    new_tensor = torch.tensor(float, device=self.device)
                    dist.barrier()
                    dist.all_reduce(new_tensor, op=dist.ReduceOp.SUM)
                    data_for_reduce[k1][idx] = float(new_tensor.item())
            else:
                assert NotImplementedError("error input type for update AvgMeterDict")

        self.data = None
        self.data = copy.deepcopy(data_for_reduce)
        self._reduce_count()

    def _reduce_count(self):
        count_sum = torch.tensor(self.count, device=self.device)
        dist.barrier()
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        self.count = float(count_sum.item())  # 

    def tofloat(self):
        pass

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)
    

class MeterDictBestV2(object):

    def __init__(self, metric_dict=None, extra_dict=None):
        # extra dict ['swa']
        # self.data = None  
        # self.float_data = True
        # ["EPE"] ["D1"] ["Thres1"] ["Thres2"] ["Thres3"]
        if metric_dict is None:
            self.best_metric = ["EPE", "D1", "Thres1", "Thres2", "Thres3"]
        else:
            self.best_metric = metric_dict
        # self.best_metric_list_len = len(self.best_metric)
        # 下面的几个重点在初始化更新的时候使用

        self.best_reset = self.get_reset_flag()

        self.best_result = {
            "EPE": 999.0,
            "EPE_epoch": 0,
            "best_EPE": {
                "EPE": 999.0, "D1": 1., "Thres1": 1., "Thres2": 1., "Thres3": 1.,
            },
            "D1": 100.,
            "D1_epoch": 0,
            "best_D1": {
                "EPE": 999.0, "D1": 1., "Thres1": 1., "Thres2": 1., "Thres3": 1.,
            },
            "Thres1": 100.,
            "Thres1_epoch": 0,
            "best_Thres1": {
                "EPE": 999.0, "D1": 1., "Thres1": 1., "Thres2": 1., "Thres3": 1.,
            },
            "Thres2": 100,
            "Thres2_epoch": 0,
            "best_Thres2": {
                "EPE": 999.0, "D1": 1., "Thres1": 1., "Thres2": 1., "Thres3": 1.,
            },
            "Thres3": 100.,
            "Thres3_epoch": 0,
            "best_Thres3": {
                "EPE": 999.0, "D1": 1., "Thres1": 1., "Thres2": 1., "Thres3": 1.,
            },
        }

    def best_metric_init(self, init_dict):
        pass
        return None

    def get_reset_flag(self):
        d = dict()
        for i in self.best_metric:
            d[i] = False
        return d
    
    @staticmethod
    def write2file(val_file, avg_test_scalars, begin_index):
        check_allfloat(avg_test_scalars)
        with open(val_file, 'a') as f:
            f.write(begin_index)
            f.write("\t")
            for k1, v1 in avg_test_scalars.items():
                if isinstance(v1, float):
                    f.write("{}: {:.4f}\t".format(k1, v1))
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    f.write("{}: {:.4f}\t".format(k1, v1[0]))
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")
            f.write(" \n")

    def expi_over(self, val_file):
        with open(val_file, 'a') as f:
            f.write(
                '\nbest EPE epoch: EPE best %s: %.5f\n\n' % (self.best_result["EPE_epoch"], self.best_result["EPE"]))
            f.write('\nbest D1 epoch: D1 best %s: %.5f\n\n' % (self.best_result["D1_epoch"], self.best_result["D1"]))
            f.write('\nbest Thres1 epoch: Thres1 best %s: %.5f\n\n' % (
            self.best_result["Thres1_epoch"], self.best_result["Thres1"]))
            f.write('\nbest Thres2 epoch: Thres2 best %s: %.5f\n\n' % (
            self.best_result["Thres2_epoch"], self.best_result["Thres2"]))
            f.write('\nbest Thres3 epoch: Thres3 best %s: %.5f\n\n' % (
            self.best_result["Thres3_epoch"], self.best_result["Thres3"]))

        Log.info('=> best epoch: %03d \t best EPE: %.5f\n' % (self.best_result["EPE_epoch"], self.best_result["EPE"]))
        Log.info('=> best epoch: %03d \t best D1: %.5f\n' % (self.best_result["D1_epoch"], self.best_result["D1"]))
        Log.info('=> best epoch: %03d \t best Thres1: %.5f\n' % (
        self.best_result["Thres1_epoch"], self.best_result["Thres1"]))
        Log.info('=> best epoch: %03d \t best Thres3: %.5f\n' % (
        self.best_result["Thres2_epoch"], self.best_result["Thres2"]))
        Log.info('=> best epoch: %03d \t best Thres2: %.5f\n' % (
        self.best_result["Thres3_epoch"], self.best_result["Thres3"]))

    def update(self, x, epoch):
        self.best_reset = self.get_reset_flag()  
        check_allfloat(x)

        data = copy.deepcopy(x) 
        best_record = copy.deepcopy(self.best_result) 

        for i in self.best_metric:
            if isinstance(data[i], float):
                data_new = {
                    "EPE": data["EPE"],
                    "D1": data["D1"],
                    "Thres1": data["Thres1"],
                    "Thres2": data["Thres2"],
                    "Thres3": data["Thres3"],
                }
                """ 
               if data[i] > self.best_result[i]:
                    data[i] = self.best_result[i]
                    data[i + '_epoch'] = self.best_result[i + '_epoch']
                    """
                if data[i] < best_record[i]:
                    Log.info('MeterDictBest initialization best {} prediction now is {} '
                             'and prediction now is {}'.format(i, self.best_result[i], data[i]))
                    self.best_result[i] = data[i]
                    self.best_result[i + '_epoch'] = epoch
                    self.best_result['best_' + i] = data_new 
                    self.best_reset[i] = True
                elif data[i] == best_record[i]:
                    if "EPE" in i:
                        self.best_reset[i] = self.epe_equal(data, epoch, float_data=True)
                    elif "Thres1" in i:
                        self.best_reset[i] = self.thres1_equal(data, epoch, float_data=True)
                    elif "Thres3" in i:
                        self.best_reset[i] = self.thres3_equal(data, epoch, float_data=True)
                    elif "D1" in i:
                        self.best_reset[i] = self.d1_equal(data, epoch, float_data=True)
                    elif "Thres2" in i:
                        self.best_reset[i] = self.thres2_equal(data, epoch, float_data=True)

            elif isinstance(data[i], tuple) or isinstance(data[i], list):
                data_new = {
                    "EPE": data["EPE"][0],
                    "D1": data["D1"][0],
                    "Thres1": data["Thres1"][0],
                    "Thres2": data["Thres2"][0],
                    "Thres3": data["Thres3"][0],
                }
                """
                if data[i][0] > self.best_result[i]:
                    data[i][0] = self.best_result[i]
                    data[i + '_epoch'][0] = self.best_result[i + '_epoch']
                """
                if data[i][0] < best_record[i]:
                    
                    Log.info('MeterDictBest initialization best {} prediction now is {} '
                             'and prediction now is {}'.format(i, self.best_result[i], data[i][0]))
                    self.best_result[i] = data[i][0]
                    self.best_result[i + '_epoch'] = epoch
                    self.best_result['best_' + i] = data_new
                    self.best_reset[i] = True
                elif data[i][0] == best_record[i]:
                    
                    if "EPE" in i:
                        self.best_reset[i] = self.epe_equal(data, epoch, float_data=False)
                    elif "Thres1" in i:
                        self.best_reset[i] = self.thres1_equal(data, epoch, float_data=False)
                    elif "Thres3" in i:
                        self.best_reset[i] = self.thres3_equal(data, epoch, float_data=False)
                    elif "D1" in i:
                        self.best_reset[i] = self.d1_equal(data, epoch, float_data=False)
                    elif "Thres2" in i:
                        self.best_reset[i] = self.thres2_equal(data, epoch, float_data=False)

    def epe_equal(self, data, epoch, float_data=True):
        reset = False
        best_last = copy.deepcopy(self.best_result['best_EPE'])
        if float_data:
            data_new = {
                "EPE": data["EPE"],
                "D1": data["D1"],
                "Thres1": data["Thres1"],
                "Thres2": data["Thres2"],
                "Thres3": data["Thres3"],
            }
            if data["Thres1"] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] == best_last["Thres3"] and data["D1"] < \
                    best_last["D1"]:
                reset = True
            else:
                reset = False
            if reset:
                assert data["EPE"] == best_last["EPE"]
                Log.info('MeterDictBest initialization best {} prediction now is EPE '
                         'and prediction now is {}'.format(self.best_result["EPE"], data["EPE"]))
                self.best_result["EPE"] = data["EPE"] 
                self.best_result["EPE_epoch"] = epoch
                self.best_result['best_EPE'] = data_new
        elif not float_data:
            data_new = {
                "EPE": data["EPE"][0],
                "D1": data["D1"][0],
                "Thres1": data["Thres1"][0],
                "Thres2": data["Thres2"][0],
                "Thres3": data["Thres3"][0],
            }
            if data["Thres1"][0] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] == best_last["Thres3"] and data["D1"][
                0] < best_last["D1"]:
                reset = True
            else:
                reset = False
            if reset:
                assert data["EPE"][0] == best_last["EPE"]
                Log.info('MeterDictBest initialization best {} prediction now is EPE '
                         'and prediction now is {}'.format(self.best_result["EPE"], data["EPE"][0]))
                self.best_result["EPE"] = data["EPE"][0]
                self.best_result["EPE_epoch"] = epoch
                self.best_result['best_EPE'] = data_new

        return reset

    def thres1_equal(self, data, epoch, float_data=True):
        reset = False
        best_last = copy.deepcopy(self.best_result['best_Thres1'])
        if float_data:
            data_new = {
                "EPE": data["EPE"],
                "D1": data["D1"],
                "Thres1": data["Thres1"],
                "Thres2": data["Thres2"],
                "Thres3": data["Thres3"],
            }
            if data["Thres3"] < best_last["Thres3"]:
                reset = True
            elif data["Thres3"] == best_last["Thres3"] and data["EPE"] < best_last["EPE"]:
                reset = True
            elif data["Thres3"] == best_last["Thres3"] and data["EPE"] == best_last["EPE"] and data["D1"] < best_last[
                "D1"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres1 '
                         'and prediction now is {}'.format(self.best_result["Thres1"], data["Thres1"]))
                self.best_result["Thres1"] = data["Thres1"]
                self.best_result["Thres1_epoch"] = epoch
                self.best_result['best_Thres1'] = data_new
        elif not float_data:
            data_new = {
                "EPE": data["EPE"][0],
                "D1": data["D1"][0],
                "Thres1": data["Thres1"][0],
                "Thres2": data["Thres2"][0],
                "Thres3": data["Thres3"][0],
            }
            if data["Thres3"][0] < best_last["Thres3"]:
                reset = True
            elif data["Thres3"][0] == best_last["Thres3"] and data["EPE"][0] < best_last["EPE"]:
                reset = True
            elif data["Thres3"][0] == best_last["Thres3"] and data["EPE"][0] == best_last["EPE"] and data["D1"][0] < \
                    best_last["D1"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres1 '
                         'and prediction now is {}'.format(self.best_result["Thres1"], data["Thres1"][0]))
                self.best_result["Thres1"] = data["Thres1"][0]
                self.best_result["Thres1_epoch"] = epoch
                self.best_result['best_Thres1'] = data_new

        return reset

    def thres3_equal(self, data, epoch, float_data=True):
        reset = False
        best_last = copy.deepcopy(self.best_result['best_Thres3'])
        if float_data:
            data_new = {
                "EPE": data["EPE"],
                "D1": data["D1"],
                "Thres1": data["Thres1"],
                "Thres2": data["Thres2"],
                "Thres3": data["Thres3"],
            }
            if data["Thres1"] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["EPE"] < best_last["EPE"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["EPE"] == best_last["EPE"] and data["D1"] < best_last[
                "D1"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres3 '
                         'and prediction now is {}'.format(self.best_result["Thres3"], data["Thres3"]))
                self.best_result["Thres3"] = data["Thres3"]
                self.best_result["Thres3_epoch"] = epoch
                self.best_result['best_Thres3'] = data_new
        elif not float_data:
            data_new = {
                "EPE": data["EPE"][0],
                "D1": data["D1"][0],
                "Thres1": data["Thres1"][0],
                "Thres2": data["Thres2"][0],
                "Thres3": data["Thres3"][0],
            }
            if data["Thres1"][0] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["EPE"][0] < best_last["EPE"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["EPE"][0] == best_last["EPE"] and data["D1"][0] < \
                    best_last["D1"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres3 '
                         'and prediction now is {}'.format(self.best_result["Thres3"], data["Thres3"][0]))
                self.best_result["Thres3"] = data["Thres3"][0]
                self.best_result["Thres3_epoch"] = epoch
                self.best_result['best_Thres3'] = data_new

        return reset

    def d1_equal(self, data, epoch, float_data=True):
        reset = False
        best_last = copy.deepcopy(self.best_result['best_D1'])
        if float_data:
            data_new = {
                "EPE": data["EPE"],
                "D1": data["D1"],
                "Thres1": data["Thres1"],
                "Thres2": data["Thres2"],
                "Thres3": data["Thres3"],
            }
            if data["Thres1"] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] == best_last["Thres3"] and data["EPE"] < \
                    best_last["EPE"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is D1 '
                         'and prediction now is {}'.format(self.best_result["D1"], data["D1"]))
                self.best_result["D1"] = data["D1"]  
                self.best_result["D1_epoch"] = epoch
                self.best_result['best_D1'] = data_new
        elif not float_data:
            data_new = {
                "EPE": data["EPE"][0],
                "D1": data["D1"][0],
                "Thres1": data["Thres1"][0],
                "Thres2": data["Thres2"][0],
                "Thres3": data["Thres3"][0],
            }
            if data["Thres1"][0] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] == best_last["Thres3"] and data["EPE"][
                0] < best_last["EPE"]:
                reset = True
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is D1 '
                         'and prediction now is {}'.format(self.best_result["D1"], data["D1"][0]))
                self.best_result["D1"] = data["D1"][0]
                self.best_result["D1_epoch"] = epoch
                self.best_result['best_D1'] = data_new

        return reset

    def thres2_equal(self, data, epoch, float_data=True):
        reset = False
        best_last = copy.deepcopy(self.best_result['best_Thres2'])
        if float_data:
            data_new = {
                "EPE": data["EPE"],
                "D1": data["D1"],
                "Thres1": data["Thres1"],
                "Thres2": data["Thres2"],
                "Thres3": data["Thres3"],
            }
            if data["Thres1"] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] == best_last["Thres3"] and data["EPE"] < \
                    best_last["EPE"]:
                reset = True
            elif data["Thres1"] == best_last["Thres1"] and data["Thres3"] == best_last["Thres3"] and data["EPE"] == \
                    best_last["EPE"]:
                if data["D1"] < best_last["D1"]:
                    reset = True
                else:
                    reset = False
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres2 '
                         'and prediction now is {}'.format(self.best_result["Thres2"], data["Thres2"]))
                self.best_result["Thres2"] = data["Thres2"]  
                self.best_result["Thres2_epoch"] = epoch
                self.best_result['best_Thres2'] = data_new
        elif not float_data:
            data_new = {
                "EPE": data["EPE"][0],
                "D1": data["D1"][0],
                "Thres1": data["Thres1"][0],
                "Thres2": data["Thres2"][0],
                "Thres3": data["Thres3"][0],
            }
            if data["Thres1"][0] < best_last["Thres1"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] < best_last["Thres3"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] == best_last["Thres3"] and data["EPE"][
                0] < best_last["EPE"]:
                reset = True
            elif data["Thres1"][0] == best_last["Thres1"] and data["Thres3"][0] == best_last["Thres3"] and data["EPE"][
                0] == best_last["EPE"]:
                if data["D1"][0] < best_last["D1"]:
                    reset = True
                else:
                    reset = False
            else:
                reset = False
            if reset:
                Log.info('MeterDictBest initialization best {} prediction now is Thres2 '
                         'and prediction now is {}'.format(self.best_result["Thres2"], data["Thres2"][0]))
                self.best_result["Thres2"] = data["Thres2"][0]  
                self.best_result["Thres2_epoch"] = epoch
                self.best_result['best_Thres2'] = data_new
        return reset
