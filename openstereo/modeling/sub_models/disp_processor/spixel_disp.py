import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, SubModule, Conv2x

from utils.logger import Logger as Log

def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape

    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(
        feat, (h * up_h, w * up_w), mode='nearest').reshape(
        b, -1, 9, h * up_h, w * up_w)
    feat_sum = (feat * prob.unsqueeze(1)).sum(2)
    return feat_sum


class BasicSpixelDisp(SubModule):
    def __init__(self, cfg, model_cfg):
        super(BasicSpixelDisp, self).__init__()

        self.refinement_cfg = model_cfg['refinement']

        backbone_cfg = model_cfg['backbone']
        # print(backbone_cfg.keys())
        original_channels = backbone_cfg['feature']['channels']
        feature_channels = backbone_cfg['feature_channels']

        self.spxc = self.refinement_cfg['spxc']

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * self.spxc[0], 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2x = Conv2x(original_channels[1], self.spxc[0], True) # Concat  spxc[0] --> 32

        self.spx_4x = nn.Sequential(
            BasicConv(feature_channels[1], original_channels[1], kernel_size=3, stride=1, padding=1), # bn and leakyrelu
            nn.Conv2d(original_channels[1], original_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(original_channels[1]),
            nn.ReLU()
        )

        Log.info("Using superpixel in final disp refinement.")

        # self.weight_init()

    def forward(self, disp_4, fea_L, stem_2x):
        # shape = inputs['left'].shape

        xspx = self.spx_4x(fea_L)
        # Log.info("xspx shape{}, stem_2x shape {}".format(xspx.shape, stem_2x.shape))
        xspx = self.spx_2x(xspx, stem_2x)
        spx_pred = self.spx(xspx)

        spx_pred = F.softmax(spx_pred, dim=1)

        disp_1 = upfeat(disp_4, spx_pred, 4, 4)
        disp_1 = disp_1.squeeze(1) * 4  # + 1.5

        return disp_1


