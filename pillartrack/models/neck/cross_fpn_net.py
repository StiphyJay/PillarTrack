import torch
import torch.nn as nn
from .cross_fpn import CrossFPN

class CrossFPNNet(nn.Module):
    def __init__(self, model_cfg, input_channels_lists):
        super().__init__()
        self.cfg = model_cfg
        self.reshape = model_cfg.RESHAPE
        self.build_fpn(input_channels_lists)

    def build_fpn(self, input_channels_lists):
        self.fpn = CrossFPN(in_channels=input_channels_lists, out_channels=self.cfg.OUT_CHANNEL, 
                            num_outs=self.cfg.NUM_OUT, cross_heads=self.cfg.HEADS)
        self.feats_list=[self.cfg.OUT_CHANNEL for _ in range(self.cfg.NUM_OUT)]

    def forward(self, batch_dict):
        x_features = batch_dict['search_feats_lists']
        t_features = batch_dict['template_feats_lists']

        x_out = self.fpn(x_features, t_features)
        
        # True: 4D Tensor; False: 3D Tensor
        if self.reshape:
            batch_dict['search_feats_lists'] = x_out
        else:
            x_feats_list = [x.flatten(2).transpose(2,1) for x in x_out]
            batch_dict['search_feats_lists'] = x_feats_list

        return batch_dict
