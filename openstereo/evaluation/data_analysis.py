from typing import Any
from .bin_analysis import BinTracker
from .representation_analysis import ConRepTracker


class DataAnalysis(object):
    def __init__(self, cfg):
        test_cfg = cfg['test']
        analysis_cfg = test_cfg['data_analysis']

        self.round = 1 


        self.use_bin_tracker = False
        self.use_representation_analysis = False

        self.use_bin_tracker = False
        if 'bin_tracker' in analysis_cfg.keys() and analysis_cfg['bin_tracker']['use']:
            self.use_bin_tracker = True
            self.bin_tracker = BinTracker(cfg)
            self.extra_name = analysis_cfg['bin_tracker'].get('extra_name', None)

        if 'vrepresentation' in analysis_cfg.keys() and analysis_cfg['vrepresentation']['use']:
            
            self.use_representation_analysis = True
            self.representation_analysis = ConRepTracker(cfg)

    def __call__(self, pred, gt, mask, representation=None, index=None,*args: Any, **kwds: Any):
        
        if self.use_bin_tracker and self.round == 1:
            self.bin_tracker(pred, gt, mask)
        if self.use_representation_analysis:
            assert representation is not None, "representation is not provided"
            self.representation_analysis(representation, gt, mask, sparse_ind=index)

    def save_result(self, path):

        if self.use_bin_tracker and self.round == 1:
            self.bin_tracker.save_result(path ,extra_name=self.extra_name)


        if self.use_representation_analysis:
            
            self.representation_analysis.save_result(path)


        self.round += 1

