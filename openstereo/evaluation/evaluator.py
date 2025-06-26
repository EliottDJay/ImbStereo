from functools import partial

from openstereo.evaluation.metric import *


"""METRICS_NP = {
    # EPE metric (Average Endpoint Error)
    'epe': epe_metric_np,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'd1_all': d1_metric_np,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    'bad_1': partial(bad_metric_np, threshold=1),
    'bad_2': partial(bad_metric_np, threshold=2),
    'bad_3': partial(bad_metric_np, threshold=3),

}"""

METRICS = {
    # EPE metric (Average Endpoint Error)
    'EPE': epe_metric,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'D1': d1_metric,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    # 'Thres1': partial(bad_metric, threshold=1),  
    # 'Thres2': partial(bad_metric, threshold=2),
    # 'Thres3': partial(bad_metric, threshold=3),
    'Thres1': thres1,  
    'Thres2': thres2,
    'Thres3': thres3,
}


class OpenStereoEvaluator:
    def __init__(self, metrics=None, use_np=False):
        # Set default metrics if none are given
        if metrics is None:
            metrics = ["EPE", "D1", "Thres1", "Thres2", "Thres3"]
        self.metrics = metrics
        self.use_np = False

    def __call__(self, data):
        # Extract input data
        disp_est = data['disp_est']
        disp_gt = data['disp_gt']
        mask = data['mask']
        res = {}

        # Loop through the specified metrics and compute results
        for m in self.metrics:
            # Check if the metric is valid
            if m not in METRICS:
                raise ValueError("Unknown metric: {}".format(m))
            else:
                # Get the appropriate metric function based on use_np
                metric_func = METRICS[m]  # if not self.use_np else METRICS_NP[m]

                # Compute the metric and store the result in the dictionary
                res[m] = metric_func(disp_est, disp_gt, mask)
        return res
