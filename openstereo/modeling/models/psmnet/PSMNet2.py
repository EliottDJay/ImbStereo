import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.stereo_utils.msg_manager import get_msg_mgr
from openstereo.modeling.models.base_trainer import BaseTrainer

# build model
from openstereo.modeling.sub_models.backbone import PSMFeature
from openstereo.modeling.sub_models.sub_models.basic import BasicConv, Conv2x
from openstereo.modeling.sub_models import volume as volumes
from openstereo.modeling.sub_models.aggregation import CrossHourglass
from openstereo.modeling.sub_models.regression.regression_fn import disparity_regression

# utils
from utils.check import get_attr_from
from utils.logger import Logger as Log


class PSMNet2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PSMNet2, self).__init__()
        # BaseModel
        self.msg_mgr = get_msg_mgr(cfg)  # build a summary writer
        self.Trainer = BaseTrainer

        self.model_cfg = cfg['net']
        self.name = 'PSMNet'
        Log.info("Using PSMNet type model ...  Details: Basic PSMNet.")
        # self.model_name = self.model_cfg['method']
        self.max_disp = self.model_cfg.get('max_disparity', 192)
        # model
        # backbone
        backbone_cfg = self.model_cfg['backbone']  # basic backbone config
        #fea_cfg = backbone_cfg['feature']

        self.feature = PSMFeature(cfg)  # feature extractor  32 channels 1/4 resolution
        self.feature_channels = self.feature.channels_recal()  # 32 1/4
        self.model_cfg['backbone']['feature_channels'] = self.feature_channels
        Log.info("final feature channels: {}".format(self.feature_channels))

        # volume
        Volume_Structure = get_attr_from([volumes], self.model_cfg['volume']['volume_type'])
        self.cat_volume_construction = Volume_Structure(self.model_cfg, disp=int(self.max_disp // 4))
        self.beta_channels = self.cat_volume_construction.get_beta_channels()  # 32 * 2  concat channels
        Log.info("The 4-th beta channels is: {}".format(self.beta_channels))

        # aggregation
        aggregation_cfg = self.model_cfg['aggregation']
        self.batch_norm = aggregation_cfg.get('batch_norm', True)
        self.dres0 = nn.Sequential(
            BasicConv(self.beta_channels, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)),
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1)),
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=3, stride=1, padding=1),
        ) 
        self.dres1 = nn.Sequential(
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=3, stride=1, padding=1),
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=False,
                      kernel_size=3, stride=1, padding=1),
        )

        self.dres2 = CrossHourglass(32, batch_norm=self.batch_norm)
        self.dres3 = CrossHourglass(32, batch_norm=self.batch_norm)
        self.dres4 = CrossHourglass(32, batch_norm=self.batch_norm)

        # regression
        self.classif1 = nn.Sequential(
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            BasicConv(32, 32, deconv=False, is_3d=True, bn=self.batch_norm, relu=True,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

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
        f_list = self.feature(cat)

        fl, fr = f_list.split(dim=0, split_size=b)

        cat_volume = self.cat_volume_construction(fl, fr)  # B 2C D H W

        if not self.training:
            del cat, f_list, fl, fr

        volume0 = self.dres0(cat_volume)

        volume0 = self.dres1(volume0) + volume0

        volume1, pre1, post1 = self.dres2(volume0, None, None)
        volume1 = volume1 + volume0

        volume2, pre2, post2 = self.dres3(volume1, pre1, post1)
        volume2 = volume2 + volume0

        volume3, pre3, post3 = self.dres4(volume2, pre2, post2)
        volume3 = volume3 + volume0

        cost1 = self.classif1(volume1)
        cost2 = self.classif2(volume2) + cost1
        cost3 = self.classif3(volume3) + cost2

        align_corners = True
        if self.training:
            """
            RuntimeError: upsample_trilinear3d_backward_out_cuda does not have a deterministic implementation, 
            but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, 
            or you can use the 'warn_only=True' option, if that's acceptable for your application. 
            You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize 
            adding deterministic support for this operation.
            """
            cost1 = F.interpolate(
                cost1, [self.max_disp, h, w], mode='trilinear', align_corners=align_corners
            )
            cost2 = F.interpolate(
                cost2, [self.max_disp, h, w], mode='trilinear', align_corners=align_corners
            )
            cost1 = torch.squeeze(cost1, 1)  # B D H W
            cost2 = torch.squeeze(cost2, 1)
            cost1 = F.softmax(cost1, dim=1)
            cost2 = F.softmax(cost2, dim=1)
            disp1 = disparity_regression(cost1, self.max_disp)
            disp2 = disparity_regression(cost2, self.max_disp)

        cost3 = F.interpolate(
            cost3, [self.max_disp, h, w], mode='trilinear', align_corners=align_corners
        )
        cost3 = torch.squeeze(cost3, 1)
        # feature = cost3.clone().detach()
        cost3 = F.softmax(cost3, dim=1)
        disp3 = disparity_regression(cost3, self.max_disp)

        if self.training:
            return {"preds_pyramid": [disp1, disp2, disp3]}
        else:
            return {"preds_pyramid": [disp3]}
        
    def forward_step(self, batch_data, device=None):
        batch_inputs = self.prepare_inputs(batch_data, device)
        outputs = self.forward(batch_inputs)
        return outputs
    
    def get_name(self):
        return self.name
    
    def init_parameters(self):
        pass
