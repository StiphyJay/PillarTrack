import pickle
import time

import numpy as np
import torch
import tqdm
import copy
from pillartrack.models import load_data_to_gpu
from pillartrack.utils import common_utils
from pillartrack.ops.iou3d_nms import iou3d_nms_utils
from pillartrack.utils import box_utils
from .track_eval_metrics import AverageMeter, Success_torch, Precision_torch

def eval_track_one_epoch(model, dataloader, epoch_id, logger, dataset_cls, dist_test=False, save_to_file=False, result_dir=None):
    dataset = dataloader.dataset
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    first_index = dataset.first_frame_index
    model.eval()
    Success_main = Success_torch()
    Precision_main = Precision_torch()

    pbar = tqdm.tqdm(total=len(first_index)-1, leave=False, desc='tracklets', dynamic_ncols=True)
    for f_index in range(len(first_index)-1):
        st = first_index[f_index]
        if f_index == len(first_index)-2:
            ov = first_index[f_index+1]+1
        else:
            ov = first_index[f_index+1]
        
        first_point = dataset[st+1]['template_voxels']   
        length = ov - st - 1
        
        if length > 0:
            for index in range(st+1, ov):
                data = dataset[index]
                if index == st+1:
                    previou_box = data['template_gt_box'].reshape(7)
                    first_point = data['or_template_points']
                    Success_main.add_overlap(torch.ones(1).cuda())
                    Precision_main.add_accuracy(torch.zeros(1).cuda())

                batch_dict = dataset.collate_batch([data])
                template_voxels = batch_dict['template_voxels']
                search_voxels = batch_dict['search_voxels']
                
                load_data_to_gpu(batch_dict)
                gt_box = batch_dict['gt_boxes'].view(-1)[:7]
                center_offset = batch_dict['center_offset'][0]
                

                try:
                    with torch.no_grad():           
                        pred_box = model(batch_dict).view(-1)
                    if dataset_cls=='nus':
                        pred_box[:3]+=center_offset[:3] #nus
                    else:
                        pred_box[:2]+=center_offset[:2] #kitti
        
                except BaseException:
                    pred_box = torch.from_numpy(previou_box).float().cuda()

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_box.view(1,-1), gt_box.view(1,-1))
                iou3d = iou3d.squeeze()
                accuracy = torch.norm(pred_box[:3] - gt_box[:3])
                Success_main.add_overlap(iou3d)
                Precision_main.add_accuracy(accuracy)
                dataset.set_first_points(first_point)
                dataset.set_refer_box(pred_box.cpu().numpy())
                previou_box = pred_box.cpu().numpy()
            dataset.reset_all()
        pbar.update()
    pbar.close()
    avs = Success_main.average.item()
    avp = Precision_main.average.item()
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    ret_dict = {}

    logger.info('Success: %f' % (avs))
    logger.info('Precision: %f' % (avp))

    ret_dict['test/Success'] = avs
    ret_dict['test/Precision'] = avp

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
