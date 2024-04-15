import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, transformer_backbone, neck, decoder_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils

class SetTracker(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'transformer_backbone', 'neck', 'decoder_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }

        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )

        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        # model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        model_info_dict['features_lists'] = backbone_2d_module.feats_list
        return backbone_2d_module, model_info_dict

    def build_transformer_backbone(self, model_info_dict):
        if self.model_cfg.get('TRANSFORMER_BACKBONE', None) is None:
            return None, model_info_dict

        transformer_backbone_module = transformer_backbone.__all__[self.model_cfg.TRANSFORMER_BACKBONE.NAME](
            model_cfg=self.model_cfg.TRANSFORMER_BACKBONE,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(transformer_backbone_module)
        model_info_dict['features_lists'] = transformer_backbone_module.feats_list
        return transformer_backbone_module, model_info_dict

    def build_neck(self, model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict

        neck_module = neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK,
            input_channels_lists=model_info_dict['features_lists']
        )
        model_info_dict['module_list'].append(neck_module)
        model_info_dict['features_lists'] = neck_module.feats_list
        return neck_module, model_info_dict
    
    def build_decoder_head(self, model_info_dict):
        if self.model_cfg.get('DECODER_HEAD', None) is None:
            return None, model_info_dict

        decoder_head_module = decoder_heads.__all__[self.model_cfg.DECODER_HEAD.NAME](
            model_cfg=self.model_cfg.DECODER_HEAD,
            input_channels_lists=model_info_dict['features_lists'],
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
        )
        model_info_dict['module_list'].append(decoder_head_module)
        return decoder_head_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        object_dim = batch_dict['object_dim']
        box_preds = batch_dict['pred_bbox']

        final_boxes = torch.cat((box_preds[:,:3],object_dim,box_preds[:,3].view(-1,1)),dim=1) # 7

        return final_boxes

    def post_processing_simple(self, batch_dict):
        object_dim = batch_dict['object_dim']

        track_center = batch_dict['track_center']
        track_ry = batch_dict['track_ry']
        track_cls = batch_dict['track_cls'].sigmoid()

        id = track_cls.argmax()
        object_dim = object_dim.repeat(track_center.shape[0],1)
        ry = torch.atan2(track_ry[:,1],track_ry[:,0]).view(-1,1)
        final_boxes = torch.cat((track_center, object_dim, ry),dim=1) # 7
        max_box = final_boxes[id]

        return max_box

    def post_processing_decode(self, batch_dict):
        object_dim = batch_dict['object_dim']

        pred_center = batch_dict['pred_center']
        pred_ry = batch_dict['pred_ry']
        pred_cls = batch_dict['pred_cls'].sigmoid()
        point_coords = batch_dict['search_point_coords']

        id = pred_cls.argmax()
        max_cls = pred_cls[id]
        object_dim = object_dim.repeat(pred_center.shape[0],1)
        
        ry = torch.atan2(pred_ry[:,1], pred_ry[:,0]).view(-1,1)

        # pred_center += point_coords[:,1:]
        final_boxes = torch.cat((pred_center, object_dim, ry),dim=1) # 7
        max_box = final_boxes[id]

        return max_box

    def post_2d(self, batch_dict):
        final_box = batch_dict['predict_box']
        return final_box

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
