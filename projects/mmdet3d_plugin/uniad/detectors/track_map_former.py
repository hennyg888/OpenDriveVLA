# ---------------------------------------------------------------------------------#
# Track_Map_Former: Simplified UniAD for Tracking and Map Segmentation only       #
# ---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
from mmdet.models.builder import build_head
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D
from mmdet3d.core import bbox3d2result

from .uniad_track import UniADTrack
from .uniad_e2e import match_bbox, pop_elem_in_result 


@DETECTORS.register_module()
class Track_Map_Former(UniADTrack):
    """
    Track_Map_Former: Simplified version of UniAD for Tracking and Map Segmentation.
    """
    def __init__(
        self,
        seg_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
        ),
        **kwargs,
    ):
        super(Track_Map_Former, self).__init__(**kwargs)
        
        if seg_head:
            self.seg_head = build_head(seg_head)
        else:
            self.seg_head = None

        self.occ_head = None
        self.motion_head = None
        self.planning_head = None

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {'track', 'map'}


    @property
    def with_planning_head(self):
        return False
    
    @property
    def with_occ_head(self):
        return False

    @property
    def with_motion_head(self):
        return False

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def get_results_for_vlm(self, img_metas, result_track, result_seg, 
                            sdc_planning, sdc_planning_mask, command, 
                            in_uniad_train=False, **kwargs):
        
        if not in_uniad_train:
            if 'gt_bboxes_3d' in kwargs:
                gt_bboxes_3d = kwargs['gt_bboxes_3d'][0][0][0].tensor
                gt_labels_3d = kwargs['gt_labels_3d'][0][0][0]
                gt_inds = kwargs['gt_inds'][0].squeeze(0)
                
                detected_bboxes = (result_track['track_bbox_results'][0][0].tensor).to(gt_bboxes_3d)[:-1, :7] 
                
                matched_idx, iou_matrix = match_bbox(detected_bboxes, gt_bboxes_3d[:,:7])
                
                track_gt_inds_to_embed_idx = {}
                for embed_idx, gt_inds_idx in enumerate(matched_idx):
                    if len(gt_inds) == 0:
                        continue
                    if gt_inds_idx != -1:
                        track_gt_inds_to_embed_idx[int(gt_inds[gt_inds_idx])] = embed_idx

                result_track['gt_bboxes_3d'] = gt_bboxes_3d
                result_track['gt_labels_3d'] = gt_labels_3d
                result_track['gt_inds'] = gt_inds
                result_track['iou_matrix'] = iou_matrix
                result_track['matched_idx'] = matched_idx
                result_track['track_gt_inds_to_embed_idx'] = track_gt_inds_to_embed_idx

        _img_metas = {}
        _result_track = {}
        _result_seg = {}

        save_keys_img_metas = ['filename', 'ori_shape', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'scene_token', 'can_bus']
        save_keys_result_seg = ['output_query_things', 'output_query_stuff', 'chosen_output_query_things']

        pop_keys_result_track = ['bev_pos']
        
        _img_metas.update({key: img_metas[key] for key in img_metas.keys() if key in save_keys_img_metas})

        _result_track.update({key: result_track[key] for key in result_track.keys() if key not in pop_keys_result_track})
        
        if in_uniad_train:
            _result_track.update({"track_query_embeddings_all": result_track["track_instances"].output_embedding})  
        else:
            _result_track.update({"track_query_embeddings_all": result_track["track_instances_fordet"].output_embedding})  
        
        if "track_query_embeddings_all" in _result_track and _result_track["track_query_embeddings_all"] is not None:
            mask = torch.ones(_result_track["track_query_embeddings_all"].shape[0], dtype=torch.bool, 
                             device=_result_track["track_query_embeddings_all"].device)
            if mask.shape[0] > 900:
                mask[900] = False
                _result_track["track_query_embeddings_all"] = _result_track["track_query_embeddings_all"][mask]

        if self.with_seg_head:
            _result_seg.update({key: result_seg[key] for key in result_seg.keys() if key in save_keys_result_seg})
        else:
            _result_seg = {} 

        if not in_uniad_train:
            if _result_track['track_query_embeddings'] is not None and _result_track['track_query_embeddings'].shape[0] > 1:
                _result_track['track_query_embeddings'] = _result_track['track_query_embeddings'][:-1]
            elif _result_track['track_query_embeddings'] is not None and _result_track['track_query_embeddings'].shape[0] == 1:
                _result_track['track_query_embeddings'] = None


        results_for_vlm = dict(
            scene_token=_img_metas['scene_token'],
            sample_token=_img_metas['sample_idx'],
            img_metas=_img_metas,
            result_track=_result_track,
            result_seg=_result_seg,
            planning_gt=dict(
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
        )
        return results_for_vlm

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      **kwargs,
                      ):
        """Forward training function."""
        losses = dict()
        len_queue = img.size(1)
        
        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        outs_track = self.upsample_bev_if_tiny(outs_track)
        bev_embed = outs_track["bev_embed"]
        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = [dict()]
        if self.with_seg_head:          
            losses_seg, outs_seg_raw = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            outs_seg[0].update(outs_seg_raw) 
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)

        results_for_vlm = self.get_results_for_vlm(img_metas[0], outs_track, outs_seg[0], 
                                                   sdc_planning[0], sdc_planning_mask[0], command[0], 
                                                   in_uniad_train=True, **kwargs)

        return losses, results_for_vlm 
    
    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     **kwargs
                    ):
        """Test function"""
        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        bev_embed = result_track[0]["bev_embed"]

        result_seg = [dict() for _ in range(len(img_metas))]
        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)
        
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding', 'track_instances_fordet', 'img_feat_2D']
        seg_pop_list = ['args_tuple', 'ret_iou', 'output_query_things', 'output_query_stuff', 'chosen_output_query_things']
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update({k: v for k, v in result_track[i].items() if k not in pop_track_list})
            if self.with_seg_head:
                res.update({k: v for k, v in result_seg[i].items() if k not in seg_pop_list})

        results_for_vlm = self.get_results_for_vlm(img_metas[0], result_track[0], result_seg[0], 
                                                   sdc_planning[0], sdc_planning_mask[0], command[0], 
                                                   in_uniad_train=False, **kwargs)

        return result, results_for_vlm 