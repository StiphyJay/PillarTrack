import torch
import torch.nn as nn
from .swin_backbone import SwinTransformer

class SwinNet(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.cfg = model_cfg
        self.build_swin(self.cfg, input_channels)

    def build_swin(self, cfg, in_channels):
        self.swin_backbone = SwinTransformer(pretrain_img_size=cfg.FEATURE_SIZE, 
            in_chans=in_channels,num_heads=cfg.NUM_HEADS,depths=cfg.DEPTHS,
                 out_indices=cfg.OUT_INDICES,embed_dim=cfg.EMBED_DIM, 
                 window_size=cfg.WINDOW_SIZE, patch_size=cfg.PATCH_SIZE, reshape_back=cfg.RESHAPE)
        num_layer = len(cfg.DEPTHS)
        self.feats_list = [2**i*cfg.EMBED_DIM for i in range(num_layer)]
        # self.swin_backbone.init_weights()

    def forward(self, batch_dict):
        if self.training:
            self.swin_backbone.train()
        
        x_features = batch_dict['x_spatial_features']
        t_features = batch_dict['t_spatial_features']

        x_feats_list = self.swin_backbone(x_features)
        t_feats_list = self.swin_backbone(t_features)

        batch_dict['search_feats_lists'] = x_feats_list
        batch_dict['template_feats_lists'] = t_feats_list

        return batch_dict
