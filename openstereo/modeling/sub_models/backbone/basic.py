import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from openstereo.modeling.sub_models.sub_models.basic import BasicConv, Conv2x, SubModule

# utils
from utils.logger import Logger as Log

Channels ={
    'mobilenetv3_large_100': [16,24,40,112,160],
    'mobilenetv2_120d': [24,32,40,112,192],
    'mobilenetv2_100': [16,24,32,96,160],
    'mnasnet_100': [16,24,40,96,192],
    'efficientnet_b0': [16,24,40,112,192],
    'efficientnet_b3a': [24,32,48,136,232],
    'mixnet_xl': [40,48,64,192,320],
    'dla34': [32,64,128,256,512]
}

Layers = {
    'mobilenetv3_large_100': [1,2,3,5,6],
    'mobilenetv2_120d': [1,2,3,5,6],
    'mobilenetv2_100': [1,2,3,5,6],
    'mnasnet_100': [1,2,3,5,6],
    'efficientnet_b0': [1,2,3,5,6],
    'efficientnet_b3a': [1,2,3,5,6],
    'mixnet_xl': [1,2,3,5,6],
    'dla34': [1,2,3,5,6]
}


# mobilev2, mobilev3
class BasicFeature(nn.Module):
    def __init__(self, cfg):
        super(BasicFeature, self).__init__()
        # self.cfg = cfg
        self.type = cfg['type']
        self.chans = cfg['channels']
        layers = cfg['layers']

        pretrained = cfg['pretrained']
        model = timm.create_model(self.type, pretrained=pretrained, features_only=True)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x_out = [x4, x8, x16, x32]

        return x2, x_out
    
    def feature_channels(self):
        return self.chans
    

class BasicFeaUp(SubModule):  # SubModule
    """
    using concat here, so channel of img 1,2,3 double
    feature: fea_2 fea_4 fea_8 fea_16 fea_32  channels: [fea_2, fea_4, fea_8, fea_16, fea_32]
    fea_up: fea_32 + fea_16 -> fea_16(cat), fea_16 + fea_8 -> fea_8(cat), fea_8 + fea_4 -> fea_4(cat)
    fea_up_channels: [ , fea_4 x 2, fea_8 x 2, fea_16 x 2, fea_32]
    """
    def __init__(self, cfg):
        super(BasicFeaUp, self).__init__()
        # self.cfg = cfg
        # self.type = cfg['type']
        self.chans = cfg['channels']

        self.deconv32_16 = Conv2x(self.chans[4], self.chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(self.chans[3] * 2, self.chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(self.chans[2] * 2, self.chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(self.chans[1] * 2, self.chans[1] * 2, kernel_size=3, stride=1, padding=1)

        Log.info("decoder is the simple structure stocked by blocks constructed by two convs layers.")
        Log.info("between blocks the before info and coming info will be concatenated.")

        self.weight_init()

    def channels_recal(self):
        channls = copy.deepcopy(self.chans)
        for i in [3, 2, 1]:
            channls[i] = channls[i]*2
        # 1/2 1/4 1/8 1/16 1/32
        return channls

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        if featR is not None:
            y4, y8, y16, y32 = featR

            x16 = self.deconv32_16(x32, x16)
            y16 = self.deconv32_16(y32, y16)

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

            return [x4, x8, x16, x32], [y4, y8, y16, y32]
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(x4)

            # chans[1]*2, chans[1] chans[2] chans[3]
            return [x4, x8, x16, x32]
        

class BasicBackbone(nn.Module):
    def __init__(self, cfg):
        super(BasicBackbone, self).__init__()
        # about all config
        model_cfg = cfg['net']
        backbone_cfg = model_cfg['backbone']  # basic backbone config
        refinement_cfg = model_cfg.get('refinement', None) # config for refinement
        # basic backbone
        fea_cfg = backbone_cfg['feature']
        self.feature = BasicFeature(fea_cfg)
        basic_feature_channels = self.feature.feature_channels()
        self.up = BasicFeaUp(fea_cfg)
        feature_channels = self.up.channels_recal()
        # extra feature
        Log.info("Using the basic encoder-decoder backbone, the encoder(feature extractor) is {}.".format(fea_cfg['type']))

        if refinement_cfg is not None:
            refine_type = refinement_cfg['refine_type']
            method_name = model_cfg['method']  
            if 'spixel' in refine_type.lower():
                self.spx_fea = True
                self.spxc = refinement_cfg['spxc']

        if refinement_cfg is not None and 'FastACV' in method_name and self.spx_fea:
            self.extra_feature = ExtraFea8xtype(self.spxc)
        elif refinement_cfg is not None and self.spx_fea:
            self.extra_feature = ExtraFeaBasic(self.spxc)
    
        if hasattr(self, 'spx_fea') and self.spx_fea:
            feature_channels = self.extra_feature.channels_recal(feature_channels)
        
        # final feature channels
        self.feature_channels = feature_channels

        # output keys
        self.output_keys = ['left_feature', 'right_feature']
        if self.spx_fea:
            self.output_keys.append('stem_2x')

        Log.info("final feature channels: {}".format(self.feature_channels))

    def forward(self, x):

        f_x2, f_list = self.feature(x)
        f_list = self.up(f_list)

        if hasattr(self, 'spx_fea') and self.spx_fea:
            spxf_x2 = self.extra_feature(x)
            stem_2x_l = spxf_x2["stem_2x"]
            f_list_ex = spxf_x2["stem_x"]
            for i in range(len(f_list_ex)):
                f_list[i] = torch.cat([f_list[i], f_list_ex[i]], dim=1)

        output = {
            'fea_list': f_list
        }

        if hasattr(self, 'spx_fea') and self.spx_fea:
                output["stem_2x"] = stem_2x_l

        return output


class ExtraFeaBasic(SubModule):
    def __init__(self, channels):
        super(ExtraFeaBasic, self).__init__()
        self.extra_channels = channels
        self.extra_len = len(channels)

        self.stem_block = nn.ModuleList()
        Log.info("Using an extra feature extractor")

        for i in range(len(channels)):  
            if i == 0:
                in_channel = 3
            else:
                in_channel = channels[i-1]
            stem_x = nn.Sequential(
            BasicConv(in_channel, channels[i], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels[i], channels[i], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[i]),
            nn.ReLU()
            )
            self.stem_block.append(stem_x)

        # self.weight_init()

    def channels_recal(self, channels):
        extra_channels = copy.deepcopy(self.extra_channels)
        # cal from 1/4

        # extra_channels = ([0]) + copy.deepcopy(extra_channels)
        num = len(extra_channels)  # [32, 48]

        for i in range(1, num):
            channels[i] = channels[i] + extra_channels[i]
        return channels

    def forward(self, x):
        input = x.clone()
        extra_fea = []
        for i in range(self.extra_len):
            fea_ = self.stem_block[i](input)
            extra_fea.append(fea_)
            input = fea_

        return {
            "stem_2x": extra_fea[0],
            "stem_x": extra_fea[1:],
        }
    

class ExtraFea8xtype(SubModule):
    def __init__(self, channels):
        super(ExtraFea8xtype, self).__init__()
        self.extra_channels = channels
        spxc = channels
        self.extra_len = len(channels)
        assert len(channels) == 3, "only used for FastACV, blocks number should be 3"
        Log.info("Using an extra feature extractor, which is designed for FastACV(p).")

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

        self.stem_x8 = nn.Sequential(
            BasicConv(spxc[1], spxc[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[1], spxc[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[2]),
            nn.ReLU() 
        )

        # self.weight_init()

    def channels_recal(self, channels):
        extra_channels = copy.deepcopy(self.extra_channels)
        # cal from 1/4

        # extra_channels = ([0]) + copy.deepcopy(extra_channels)
        num = len(extra_channels)  # [32, 48]

        for i in range(1, num):
            channels[i] = channels[i] + extra_channels[i]
        return channels

    def forward(self, x):
        x2 = self.stem_x2(x)
        x4 = self.stem_x4(x2)
        x8 = self.stem_x8(x4)

        return {
            "stem_2x": x2,
            "stem_x": [x4, x8],
        }