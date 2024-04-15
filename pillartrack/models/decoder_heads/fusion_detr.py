import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention_blocks import SelfBlock, CrossBlock
from ..attention_blocks import MLP
import math
import warnings
import copy
from .matcherfg import HungarianMatcherFG
from .set_criterionfg import SetCriterionFG
from ...utils import loss_utils, box_utils

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FusionDetr(nn.Module):
    def __init__(self, model_cfg, input_channels_lists, grid_size, point_cloud_range, voxel_size):
        super().__init__()
        
        input_channel = input_channels_lists[0]
        self.linear_fuse = nn.Conv2d(
            in_channels=input_channel*4,
            out_channels=input_channel,
            kernel_size=1,
        )

        self.two_stage = model_cfg.TWO_STAGE

        self.two_stage_num_proposals = model_cfg.NUM_PROPOSALS
        self.pos_dim = model_cfg.POS_DIM if model_cfg.POS_DIM is not None else 128

        ################
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size
        ################

        encoder = []
        decoder = []
        for _ in range(model_cfg.NLAYERS):
            encoder.append(
                SelfBlock(input_channel, model_cfg.DIM_FFN, model_cfg.NHEADS),
            )
            decoder.append(
                CrossBlock(input_channel, model_cfg.DIM_FFN, model_cfg.NHEADS),
            )
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        self.reg_layer = nn.Linear(input_channel, 5)
        self.cls_layer = nn.Linear(input_channel, 2)

        if self.two_stage:
            self.enc_output = nn.Linear(input_channel, input_channel)
            self.enc_output_norm = nn.LayerNorm(input_channel)

            self.encoder_reg = nn.Linear(input_channel, 5)
            self.encoder_cls = nn.Linear(input_channel, 2)

            self.pos_trans = nn.Linear(self.pos_dim*5, input_channel)
            self.pos_trans_norm = nn.LayerNorm(input_channel)
            self.tgt_proj = nn.Linear(input_channel*2, input_channel)
            self.track_query = None
        else:
            self.track_query = nn.Parameter(torch.randn(self.two_stage_num_proposals, input_channel))
            
        self.build_losses(model_cfg.MATCH_CONFIG, model_cfg.LOSS_CONFIG)

    def build_losses(self, match_config, loss_config):
        matcher = HungarianMatcherFG(match_config.CLASS, match_config.BOX, match_config.IOU, match_config.RY)
        losses = ['labels', 'boxes', 'cardinality']
        weight_dict = {'loss_ce': loss_config.CLASS, 'loss_bbox': loss_config.BOX}
        weight_dict['loss_iou'] = loss_config.IOU

        if self.two_stage:
            aux_weight_dict = {}
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        box_loss_weight = loss_config.LOSS_WEIGHTS
        self.loss_set = SetCriterionFG(2, matcher, weight_dict, losses, box_loss_weight)

    def get2d_fg(self, gt_box, feat_w, feat_h):
        dtype = gt_box.dtype
        device = gt_box.device

        heatmap = gt_box.new_zeros(feat_h*feat_w,1)

        fg_mask = box_utils.get_bev_box_mask(gt_box.cpu().numpy(), feat_h, feat_w, self.pc_range, [x*self.feature_map_stride for x in self.voxel_size])
        fg_mask = torch.from_numpy(fg_mask).to(device)
        heatmap[fg_mask]=1
        heatmap = heatmap.reshape(feat_h, feat_w)
        fg2d = torch.nonzero(heatmap)
        fg2d = torch.cat((fg2d[:,1:2],fg2d[:,0:1]),dim=1)
        return fg2d

    def label_assign(self, input_dict):
        gt_boxes = input_dict['gt_boxes']
        center_offset = input_dict['center_offset']

        local_gt = gt_boxes.clone().squeeze(1)[:,:7]
        local_gt[:,:3] -= center_offset
        target_label = []
        for k in range(local_gt.shape[0]):
            fg_2d = self.get2d_fg(local_gt[k],self.feat_shape,self.feat_shape)
            target_cls = local_gt.new_ones(fg_2d.shape[0]).long()
            target_reg = local_gt[k].unsqueeze(0).repeat(fg_2d.shape[0], 1)
            target_label.append({
                'boxes': target_reg,
                'labels':target_cls
            })
        return target_label

    def get_proposal_pos_embed(self, proposals):
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(self.pos_dim, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / self.pos_dim)
        # N, L, 5
        proposals = proposals.sigmoid() * scale
        # N, L, 5, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 5, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4)
        return pos.flatten(2)

    def get_loss(self):
        pred_dict = self.forward_ret_dict['predict']
        target_dict = self.forward_ret_dict['target']
        loss_dict = self.loss_set(pred_dict, target_dict)
        weight_dict = self.loss_set.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return loss, loss_dict

    def gen_encoder_output_proposals(self, memory, spatial_shapes):
        B, N, C = memory.shape

        voxel_size = torch.tensor(self.voxel_size, dtype=memory.dtype, device=memory.device)
        pc_range = torch.tensor(self.pc_range, dtype=memory.dtype, device=memory.device)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, spatial_shapes[0] - 1, spatial_shapes[0], dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, spatial_shapes[1] - 1, spatial_shapes[1], dtype=torch.float32, device=memory.device))
        point_x = (grid_x + 0.5) * self.feature_map_stride * voxel_size[0] + pc_range[0]
        point_y = (grid_y + 0.5) * self.feature_map_stride * voxel_size[1] + pc_range[1]
        point_xy = torch.cat([point_x.unsqueeze(-1), point_y.unsqueeze(-1)], -1)
        point_proposal = point_xy.flatten(0,1).unsqueeze(0).repeat(B, 1, 1)

        output_memory = memory
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, point_proposal

    def forward(self, batch_dict):
        assert self.two_stage or self.track_query is not None

        x_features = batch_dict['search_feats_lists']
        c1, c2, c3, c4 = x_features

        self.feat_shape = c1.shape[2]
        self.feature_map_stride = int(self.grid_size[0] / self.feat_shape)

        c4 = resize(c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        c3 = resize(c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        c2 = resize(c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # B, N, C
        feats = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1)).flatten(2).transpose(2,1)

        for self_attn in self.encoder:
            feats = self_attn(feats)

        if self.two_stage:
            feats, proposal_xy = self.gen_encoder_output_proposals(feats, c1.size()[2:])

            encoder_reg = self.encoder_reg(feats) # B, N, 5
            encoder_cls = self.encoder_cls(feats) # B, N, 2
            encoder_reg[:,:,:2] += proposal_xy
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(encoder_cls[..., 0], topk, dim=1)[1]
            topk_coords = torch.gather(encoder_reg, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5))
            topk_feat = torch.gather(feats, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, feats.shape[2]))
            topk_feat = topk_feat.detach()
            topk_coords = topk_coords.detach()
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords)))
            tgt = self.tgt_proj(torch.cat((topk_feat, pos_trans_out),dim=2)) # B, K, C
        else:
            B = feats.shape[0]
            tgt = self.track_query.unsqueeze(0).repeat(B, 1, 1) # B, K, C

        for cross_attn in self.decoder:
            tgt = cross_attn(tgt, feats)

        decoder_feat = tgt

        reg_out = self.reg_layer(decoder_feat) # B, K, 5
        cls_out = self.cls_layer(decoder_feat) # B, K, 2

        batch_dict['pred_boxes'] = reg_out
        batch_dict['pred_logits'] = cls_out

        ret_dict = {
            'predict':
                {
                'pred_boxes': reg_out,
                'pred_logits': cls_out,
                }
        }
        if self.two_stage:
            batch_dict['enc_outputs'] = {'pred_boxes': encoder_reg, 'pred_logits': encoder_cls}
            ret_dict['predict'].update({
                        'enc_outputs': 
                            {'pred_boxes': encoder_reg, 'pred_logits': encoder_cls},
                        })

        if self.training:
            label_dict = self.label_assign(batch_dict)
            ret_dict['target'] = label_dict
            self.forward_ret_dict = ret_dict
        else:
            batch_dict = self.post_process_top(batch_dict)

        return batch_dict
    
    @torch.no_grad()
    def post_process(self, batch_dict):
        reg_predicts = batch_dict['pred_boxes'] # B, N, 5
        cls_predicts = batch_dict['pred_logits'].sigmoid() # B, N, 2
        top_cls = torch.topk(cls_predicts[..., 0], 1, dim=1)[1]
        topk_reg = torch.gather(reg_predicts, 1, top_cls.unsqueeze(-1).repeat(1, 1, 5)).squeeze(1)
        whl = batch_dict['object_dim'] # B, 3
        ry_pred = torch.atan2(topk_reg[:,3:4], topk_reg[:,4:5])
        final_box = torch.cat((topk_reg[:,:3], whl, ry_pred),dim=1)
        batch_dict['predict_box'] = final_box 
        return batch_dict

    @torch.no_grad()
    def post_process_top(self, batch_dict):
        out_bbox = batch_dict['pred_boxes'] # B, N, 5
        out_logits = batch_dict['pred_logits']
        
        prob = out_logits.sigmoid() # B, N, 2
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 16, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,5))

        whl = batch_dict['object_dim'] # B, 3
        box_list = []
        for k in range(boxes.shape[0]):
            tscore = scores[k][labels[k]==1]
            tindex = torch.topk(tscore,1,dim=0)[1]
            tboxes = boxes[k][labels[k]==1][tindex]

            xyz = tboxes[:,:3]
            ry_pred = torch.atan2(tboxes[:,3:4], tboxes[:,4:5])
            final_box = torch.cat((xyz, whl[k].unsqueeze(0), ry_pred),dim=1)
            box_list.append(final_box)
        final_box = torch.cat((box_list),dim=0) # B, 7
        batch_dict['predict_box'] = final_box 
        return batch_dict
