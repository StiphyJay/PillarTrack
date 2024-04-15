import torch
import torch.nn as nn
from .pvtv2_backbone import PyramidVisionTransformerV2

class PVTNetV2(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.cfg = model_cfg
        self.build_swin(self.cfg, input_channels)

    def build_swin(self, cfg, in_channels):
        self.pvt_backbone = PyramidVisionTransformerV2(img_size=cfg.FEATURE_SIZE, patch_size=cfg.PATCH_SIZE,
                                        in_chans=in_channels, embed_dims=cfg.EMBED_DIM, 
                                        num_heads=cfg.NUM_HEADS, mlp_ratios=cfg.MLP_RATIOS,
                                        depths=cfg.DEPTHS, reshape_back=cfg.RESHAPE)
        self.feats_list = self.pvt_backbone.embed_dims

    def forward(self, batch_dict):
        x_features = batch_dict['x_spatial_features']
        t_features = batch_dict['t_spatial_features']
        
        x_feats_list = self.pvt_backbone(x_features)
        t_feats_list = self.pvt_backbone(t_features)
        
        batch_dict['search_feats_lists'] = x_feats_list
        batch_dict['template_feats_lists'] = t_feats_list

        return batch_dict
