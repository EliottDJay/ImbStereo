import torch
import torch.nn as nn
import torch.nn.functional as F

from openstereo.stereo_utils.msg_manager import get_msg_mgr
from openstereo.modeling.models.base_trainer import BaseTrainer

# build model
from openstereo.modeling.sub_models.backbone import BasicFeature, BasicFeaUp
from openstereo.modeling.sub_models.sub_models.basic import BasicConv, Conv2x
from openstereo.modeling.sub_models.sub_models.attention import ChannelAtt
from openstereo.modeling.sub_models import volume as volumes
from openstereo.modeling.sub_models import aggregation as aggregations
# from openstereo.modeling.sub_models import regression as regressions
from openstereo.modeling.sub_models.regression.regression_fn import regression_topk, regression_topk_sparse
from openstereo.modeling.sub_models.disp_processor import upfeat

# utils

from utils.check import get_attr_from
from utils.logger import Logger as Log

class FastACVPlus2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FastACVPlus2, self).__init__()
        # BaseModel
        self.msg_mgr = get_msg_mgr(cfg)  # build a summary writer
        self.Trainer = BaseTrainer

        self.model_cfg = cfg['net']
        self.name = 'FastACV_Plus'
        Log.info("Using FastACVPlus type model ...")
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

        # ACV setting
        acv_cfg = self.model_cfg.get('acv', None)
        assert acv_cfg is not None
        self.coarse_topk = acv_cfg["coarse_topk"]  # top k value in the coarse map (Volume)
        self.att_weights_only = acv_cfg["att_weights_only"] 

        # att volume and volume
        Att_Volume_Structure = get_attr_from([volumes], self.model_cfg['att_volume']['volume_type'])
        self.corr_volume_construction = Att_Volume_Structure(self.model_cfg, self.model_cfg['att_volume'], disp=int(self.max_disp // 4))
        self.att_beta_channels = self.corr_volume_construction.get_beta_channels()  # 32 * 2  concat channels
        # feature_channels compress and then sent to the corr_volume_construction
        self.corr_fea_compress_x4 = nn.Sequential(
            BasicConv(feature_channels[1], feature_channels[1] // 2, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature_channels[1] // 2, feature_channels[1] // 2, kernel_size=1, padding=0, stride=1))
        
        # att volume aggregation
        att_aggregation_cfg = self.model_cfg['att_aggregation']
        init_beta = att_aggregation_cfg.get('init_group', 8)
        self.corr_agg_4x_first = BasicConv(self.att_beta_channels, init_beta, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_4 = ChannelAtt(init_beta, feature_channels[1])
        Att_Aggregation = get_attr_from([aggregations], att_aggregation_cfg['type'])
        self.att_hourglass_4 = Att_Aggregation(cfg, self.model_cfg, agg_cfg=att_aggregation_cfg)

        # volume part 
        if not self.att_weights_only:
            volume_cfg = self.model_cfg['volume']
            cat_group = volume_cfg.get('group', 32) 
            self.feature_compress = nn.Sequential(
                BasicConv(feature_channels[1], 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32//2, 3, 1, 1, bias=False))
            Volume_Structure = get_attr_from([volumes], volume_cfg['volume_type'])
            self.sparse_cat_volume = Volume_Structure(self.model_cfg, disp=int(self.max_disp // 4))
        
        # aggregation part
        if not self.att_weights_only:
            self.cat_agg_first = BasicConv(32, 32//2, is_3d=True, kernel_size=3, stride=1, padding=1)
            self.cat_fea_att_4 = ChannelAtt(32//2, feature_channels[1])
            Aggregation = get_attr_from([aggregations], self.model_cfg['aggregation']['type'])
            self.cat_hourglass_4 = Aggregation(cfg, self.model_cfg)

        # regression and refinement
        self.topk = self.model_cfg['regression']['topk']

        #SPX
        self.stem_x2 = nn.Sequential(
            BasicConv(3, spxc[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[0], spxc[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[0]), nn.ReLU())
        self.stem_x4 = nn.Sequential(
            BasicConv(spxc[0], spxc[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[1], spxc[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[1]), nn.ReLU())
        
        Log.info("Using superpixel in final disp refinement.")
        self.spx_4 = nn.Sequential(
            BasicConv(feature_channels[1], original_channels[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(original_channels[1], original_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(original_channels[1]), nn.ReLU())
        self.spx_2 = Conv2x(original_channels[1], spxc[0], True)
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * spxc[0], 9, kernel_size=4, stride=2, padding=1), )

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

        # att volume
        match_left = self.corr_fea_compress_x4(fl[0])
        match_right = self.corr_fea_compress_x4(fr[0])
        corr_volume = self.corr_volume_construction(match_left, match_right)

        # att_aggregation
        corr_volume = self.corr_agg_4x_first(corr_volume)
        att_volume = self.corr_feature_att_4(corr_volume, fl[0])
        att_weights = self.att_hourglass_4(att_volume, fl[1:])

        att_weights_prob = F.softmax(att_weights, dim=2)

        # coarse map top k
        _, ind = att_weights_prob.sort(2, True)
        ind_k = ind[:, :, :self.coarse_topk]
        ind_k = ind_k.sort(2, False)[0]

        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.att_weights_only:
            att_topk = torch.gather(att_weights_prob, 2, ind_k)
            # volume
            concat_features_left = self.feature_compress(fl[0])
            concat_features_right = self.feature_compress(fr[0])
            concat_volume = self.sparse_cat_volume(concat_features_left, concat_features_right, disparity_sample_topk)
            volume = att_topk * concat_volume

            # volume aggregation
            volume = self.cat_agg_first(volume)
            volume = self.cat_fea_att_4(volume, fl[0])
            volume_final = self.cat_hourglass_4(volume, fl[1:3])

        # spx
        xspx = self.spx_4(fl[0])
        xspx = self.spx_2(xspx, spxfl_x2)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        if self.training or self.att_weights_only:

            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = torch.sum(att_prob * disparity_sample_topk, dim=1)

            pred_att_up = upfeat(pred_att.unsqueeze(1), spx_pred, 4, 4)
            att_pred = pred_att * 4
            att_up = pred_att_up.squeeze(dim=1) * 4
            if self.att_weights_only:
                return {
                    "preds_pyramid": [att_pred, att_up],
                    "sparse_representation": [att_weights.clone().squeeze(1)],
                    "sparse_index": [ind_k.clone().squeeze(1)],
                }
            
        pred_x4, pred_prob = regression_topk_sparse(volume_final.squeeze(1), self.topk, disparity_sample_topk)
        pred_up = upfeat(pred_x4.unsqueeze(dim=1), spx_pred, 4, 4)
        pred4 = pred_x4 * 4
        pred4_up = pred_up.squeeze(dim=1) * 4

        if self.training:
            return {
                "preds_pyramid": [att_pred, att_up, pred4, pred4_up],
            }
        else:
            return {
                "preds_pyramid": [pred4_up],
                "sparse_representation": [volume_final.squeeze(1)],
                "sparse_index": [ind_k.clone().squeeze(1)],
            }
        
    def forward_step(self, batch_data, device=None):
        batch_inputs = self.prepare_inputs(batch_data, device)
        outputs = self.forward(batch_inputs)
        return outputs
    
    def get_name(self):
        return self.name
    
    def init_parameters(self):
        pass
