import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention_blocks import SelfBlock, CrossBlock

class CrossFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 cross_heads,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 upsample_cfg=dict(mode='nearest')):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.cross_lateral = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                1)
            cross_attention = CrossBlock(
                out_channels,
                out_channels,
                cross_heads)
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.cross_lateral.append(cross_attention)
            self.fpn_convs.append(fpn_conv)

    def forward(self, search_feats, template_feats):
        """Forward function."""
        assert len(search_feats) == len(template_feats) == len(self.in_channels)

        proj_x = [
            lateral_conv(search_feats[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        proj_t = [
            lateral_conv(template_feats[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build laterals
        ##########################
        laterals = []
        for i, cross_attention in enumerate(self.cross_lateral):
            x = proj_x[i + self.start_level]
            t = proj_t[i + self.start_level]
            B, _, H, W = x.shape
            x = x.flatten(2).transpose(1,2)
            t = t.flatten(2).transpose(1,2)
            out = cross_attention(x, t).view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            laterals.append(out)
        ##########################

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return tuple(outs)