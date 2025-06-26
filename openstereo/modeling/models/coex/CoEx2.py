import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.stereo_utils.msg_manager import get_msg_mgr
from openstereo.modeling.models.base_trainer import BaseTrainer

# build model
from openstereo.modeling.sub_models.backbone import BasicFeature, BasicFeaUp
from openstereo.modeling.sub_models.sub_models.basic import BasicConv, Conv2x
from openstereo.modeling.sub_models import volume as volumes
from openstereo.modeling.sub_models import aggregation as aggregations
# from openstereo.modeling.sub_models import regression as regressions
from openstereo.modeling.sub_models.regression.regression_fn import regression_topk
from openstereo.modeling.sub_models.disp_processor import upfeat

# utils

from utils.check import get_attr_from
from utils.logger import Logger as Log


class CoEx2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CoEx2, self).__init__()
        # BaseModel
        self.msg_mgr = get_msg_mgr(cfg)  # build a summary writer
        self.Trainer = BaseTrainer

        self.model_cfg = cfg['net']
        self.name = 'CoEx'
        # self.model_name = self.model_cfg['method']
        self.max_disp = self.model_cfg.get('max_disparity', 192)

        # model
        # backbone
        backbone_cfg = self.model_cfg['backbone']  # basic backbone config
        fea_cfg = backbone_cfg['feature']
        refinement_cfg = self.model_cfg.get('refinement', None) # config for refinement
        Log.info("Using the basic encoder-decoder backbone, the encoder(feature extractor) is {}.".format(fea_cfg['type']))
        self.feature = BasicFeature(fea_cfg)
        self.up = BasicFeaUp(fea_cfg)
        spxc = refinement_cfg['spxc']
        Log.info("Using an extra feature extractor")
        # feature_channels
        original_channels = fea_cfg['channels']  # 1/2 1/4 1/8 1/16 1/32
        feature_channels = self.up.channels_recal()  # 1/2 1/4 1/8 1/16 1/32
        feature_channels[1] = feature_channels[1] + spxc[1]
        self.feature_channels = feature_channels
        self.model_cfg['backbone']['feature_channels'] = self.feature_channels
        Log.info("final feature channels: {}".format(self.feature_channels))

        # volume
        Volume_Structure = get_attr_from([volumes], self.model_cfg['volume']['volume_type'])
        self.volume_constructor = Volume_Structure(self.model_cfg, disp=int(self.max_disp // 4))

        # aggregation
        Aggregation = get_attr_from([aggregations], self.model_cfg['aggregation']['type'])
        self.aggregation = Aggregation(cfg, self.model_cfg, disp=int(self.max_disp // 4))

        # regression and refinement
        self.topk = self.model_cfg['regression']['topk']
        # Regression = get_attr_from([regressions], self.model_cfg['regression']['regression_type'])
        # self.regression = Regression(cfg, int(self.max_disp // 4))
        

        # SPX
        self.stem_x2 = nn.Sequential(
            BasicConv(3, spxc[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[0], spxc[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[0]),
            nn.ReLU() 
        )
        self.stem_x4 = nn.Sequential(
            BasicConv(spxc[0], spxc[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[1], spxc[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[1]),
            nn.ReLU() 
        )

        Log.info("Using superpixel in final disp refinement.")
        self.spx_4x = nn.Sequential(
            BasicConv(feature_channels[1], original_channels[1], kernel_size=3, stride=1, padding=1), # bn and leakyrelu
            nn.Conv2d(original_channels[1], original_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(original_channels[1]),
            nn.ReLU()
        )
        self.spx_2x = Conv2x(original_channels[1], spxc[0], True) # Concat  spxc[0] --> 32
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*spxc[0], 9, kernel_size=4, stride=2, padding=1))

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

    def forward(self, inputs):  # inputs  img_l, img_r
        img_l = inputs['left_img']
        img_r = inputs['right_img']
        b, c, h, w = img_l.shape
        cat = torch.cat((img_l, img_r), dim=0)

        # feature extractor
        _, f_list = self.feature(cat)
        f_list = self.up(f_list)

        spxf_x2 = self.stem_x2(cat)
        spxf_x4 = self.stem_x4(spxf_x2)
        spxfl_x2, _ = spxf_x2.split(dim=0, split_size=b)
        spxfl_x4, spxfr_x4 = spxf_x4.split(dim=0, split_size=b)

        fl, fr = [], []
        for v_ in f_list:
            fl_, fr_ = v_.split(dim=0, split_size=b)
            fl.append(fl_)
            fr.append(fr_)

        fl[0] = torch.cat((fl[0], spxfl_x4), 1)
        fr[0] = torch.cat((fr[0], spxfr_x4), 1)

        # cost volume
        volume_out = self.volume_constructor(fl[0], fr[0])

        # aggregation
        volume_final = self.aggregation(fl, volume_out)
        volume_final = torch.squeeze(volume_final, 1)

        # spx
        xspx = self.spx_4x(fl[0])
        xspx = self.spx_2x(xspx, spxfl_x2)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        # regression
        disp_4 = regression_topk(volume_final, self.topk, self.max_disp // 4)  # 4D volume

        disp_1 = upfeat(disp_4.unsqueeze(dim=1), spx_pred, 4, 4)
        disp_1 = disp_1.squeeze(1) * 4  # + 1.5

        if self.training:
            disp_4 = disp_4 * 4  # + 1.5
            return {"preds_pyramid": [disp_4, disp_1]}
        else:
            return {"preds_pyramid": [disp_1],
                    "representation": [volume_final.clone()]}

    
    def forward_step(self, batch_data, device=None):
        batch_inputs = self.prepare_inputs(batch_data, device)
        outputs = self.forward(batch_inputs)
        return outputs
    
    def get_name(self):
        return self.name
    
    def init_parameters(self):
        pass