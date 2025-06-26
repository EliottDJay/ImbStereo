import torch

from ..base_model import BaseModel

# build model
from openstereo.modeling.sub_models import backbone as backbones
from openstereo.modeling.sub_models import volume as volumes
from openstereo.modeling.sub_models import aggregation as aggregations
from openstereo.modeling.sub_models import regression as regressions
from openstereo.modeling.sub_models import disp_processor as refinements
from openstereo.modeling import loss as losses

# utils
from utils.check import get_attr_from
from utils.logger import Logger as Log


class CoEx(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(CoEx, self).__init__(cfg, **kwargs)

        self.model_cfg = cfg['net']
        self.name = 'CoEx'
        # self.model_name = self.model_cfg['method']
        self.max_disp = self.model_cfg.get('max_disparity', 192)
        # self.DispProcessor = None
        # self.CostProcessor = None
        # self.Backbone = None

        # model structure
        self.backbone = None
        self.volume_constructor = None
        self.aggregation = None
        self.regression = None
        self.refinement = None

        self.build_network(cfg)

    def build_network(self, cfg):
        # self.model_cfg
        Backbone = get_attr_from([backbones], self.model_cfg['backbone']['backbone_type'])
        self.backbone = Backbone(cfg)

        self.feature_channels = self.backbone.feature_channels
        self.model_cfg['backbone']['feature_channels'] = self.feature_channels
        # self.model_cfg.update({'backbone': {'feature_channels': self.feature_channels}})
        Volume_Structure = get_attr_from([volumes], self.model_cfg['volume']['volume_type'])
        self.volume_constructor = Volume_Structure(self.model_cfg, disp=int(self.max_disp // 4))
        Aggregation = get_attr_from([aggregations], self.model_cfg['aggregation']['type'])
        self.aggregation = Aggregation(cfg, self.model_cfg, disp=int(self.max_disp // 4))
        Regression = get_attr_from([regressions], self.model_cfg['regression']['regression_type'])
        self.regression = Regression(cfg)
        Refinement = get_attr_from([refinements], self.model_cfg['refinement']['refine_type'])
        self.refinement = Refinement(cfg, self.model_cfg)

    def prepare_inputs(self, inputs, device=None, **kwargs):
        """
        Args:
            inputs: the input data.
            device: the device to put the data.
        Returns:
        """
        processed_inputs = {
            'left_img': inputs['left'],
            'right_img': inputs['right'],
        }

        processed_inputs.update(
            {
                'batch_size': torch.tensor(inputs['left'].size()[0]),
            }
        )

        if not self.training:
            for k in ['top_pad', 'right_pad', 'left_name']:
                if k in inputs.keys():
                    processed_inputs[k] = inputs[k]

        if device is not None:
            # move data to device
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device) if torch.is_tensor(v) else v
                
        return processed_inputs
    
    def feature_extractor(self, inputs):
        img_l = inputs['left_img']
        b, c, h, w = img_l.shape
        img_r = inputs['right_img']
        cat = torch.cat((img_l, img_r), dim=0)
        feature_out = self.backbone(cat)
        f_list = feature_out['fea_list']
        stem_2x = feature_out['stem_2x']
        spxfl_x2, _ = stem_2x.split(dim=0, split_size=b)
        fl, fr = [], []
        for v_ in f_list:
            fl_, fr_ = v_.split(dim=0, split_size=b)
            fl.append(fl_)
            fr.append(fr_)
        return {
            'left_feature': fl,
            'right_feature': fr,
            'stem_2x': spxfl_x2,
        }
    
    def to_construct_volume(self, inputs):
        left_feature = inputs['left_feature'][0]
        right_feature = inputs['right_feature'][0]
        volume_out = self.volume_constructor(left_feature, right_feature)
        return {"init_cost": volume_out}
    
    def cost_aggregation(self, inputs):
        left_feature = inputs['left_feature']
        cost = inputs['init_cost']
        final_cost = self.aggregation(left_feature, cost)
        return {"cost": final_cost}
    
    def disp_generate(self, inputs):
        cost = inputs['cost']
        disp_4 = self.regression(cost.squeeze(dim=1))
        fea_L = inputs['left_feature'][0]
        stem_2x = inputs['stem_2x']
        disp_1 = self.refinement(disp_4, fea_L, stem_2x)

        if self.training:
            disp_4 = disp_4.squeeze(1) * 4  # + 1.5
            return {"preds_pyramid": [disp_4, disp_1]}
        else:
            volume_final = torch.squeeze(cost, 1)
            #Log.info("CoEx volume_final shape: %s" % str(volume_final.shape))
            return {"preds_pyramid": [disp_1],
                    "representation": [volume_final.clone()]}

    def forward(self, inputs):
        feature_out = self.feature_extractor(inputs)
        """
        left_feature, right_feature, (if self.backbone.spx_fea) stem_2x
        encoder: [16,24,32,96,160]
        decoder: [_, 48, 64, 192, 160]
        """
        inputs.update(feature_out)
        volume_out = self.to_construct_volume(inputs)
        """
        init_cost
        """
        inputs.update(volume_out)
        final_cost = self.cost_aggregation(inputs)
        """
        cost
        """
        inputs.update(final_cost)
        disp = self.disp_generate(inputs)
        # inputs.update(disp)
        return disp
    
    def forward_step(self, batch_data, device=None):
        batch_inputs = self.prepare_inputs(batch_data, device)
        outputs = self.forward(batch_inputs)
        return outputs
    
    def get_name(self):
        return self.name
    