import torch
from mmdet.models import DETECTORS, build_head
from .uniad_track import UniADTrack
from typing import Dict
from mmcv.runner import auto_fp16
from .uniad_e2e import match_bbox

@DETECTORS.register_module()
class Track_Map_Former(UniADTrack):
    """
    Track + Map (Seg) joint detector.
    Inherits ALL tracking logic from UniADTrack.
    """

    def __init__(
        self,
        seg_head=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.seg_head = build_head(seg_head) if seg_head is not None else None


    @auto_fp16()
    def forward_test(
        self,
        bev_embed=None,
        img_feat_2D=None,
        img_metas=None,
        timestamp=None,
        l2g_r_mat=None,
        l2g_t=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        **kwargs,
    ):

        assert bev_embed is not None, "bev_embed is required"
        assert img_metas is not None, "img_metas is required"
        assert len(img_metas) == 1, "only single batch test is supported"

        if bev_embed.dim() == 4:
            B, C, H, W = bev_embed.shape
            bev_hwbc = bev_embed.flatten(2).permute(2, 0, 1).contiguous()
            bev_map = bev_embed
        else:
            bev_hwbc = bev_embed
            H = self.pts_bbox_head.bev_h
            W = self.pts_bbox_head.bev_w
            B = bev_hwbc.shape[1]
            C = bev_hwbc.shape[2]
            bev_map = bev_hwbc.permute(1, 2, 0).contiguous().view(B, C, H, W)

        assert bev_hwbc.shape[0] == self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w

        track_instances = self._generate_empty_tracks()

        det_output = self.pts_bbox_head.get_detections(
            bev_hwbc,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        query_feats = det_output["query_feats"]
        last_ref_pts = det_output["last_ref_points"]

        track_instances.pred_logits = output_classes[-1, 0]
        track_instances.pred_boxes = output_coords[-1, 0]
        track_instances.output_embedding = query_feats[-1][0]
        track_instances.ref_pts = last_ref_pts[0]

        with torch.no_grad():
            track_instances.scores = track_instances.pred_logits.sigmoid().max(dim=-1).values

        sdc_idx = getattr(self, "num_query", None)
        if sdc_idx is None:
            sdc_idx = track_instances.obj_idxes.numel() - 1
        track_instances.obj_idxes[sdc_idx] = -2

        if hasattr(self, "track_base") and self.track_base is not None:
            self.track_base.update(track_instances, None)
            filter_thresh = self.track_base.filter_score_thresh
        else:
            filter_thresh = 0.0

        active_index = (track_instances.obj_idxes >= 0) & (track_instances.scores >= filter_thresh)

        result_track = self.select_active_track_query(track_instances, active_index, img_metas)
        result_track["img_feat_2D"] = img_feat_2D
        result_track["track_instances_fordet"] = track_instances

        if hasattr(self, "select_sdc_track_query"):
            try:
                result_track.update(self.select_sdc_track_query(track_instances[sdc_idx], img_metas))
            except Exception:
                pass


        if gt_bboxes_3d is not None and gt_inds is not None:
            detected_boxes3d = result_track["track_bbox_results"][0][0].tensor  # LiDARInstance3DBoxes.tensor
            if detected_boxes3d.shape[0] > 0:
                detected_bboxes = detected_boxes3d.to(gt_bboxes_3d[0][0][0].tensor)[:-1, :7]  # drop sdc

                gt_boxes = gt_bboxes_3d[0][0][0].tensor[:, :7]
                matched_idx, iou_matrix = match_bbox(detected_bboxes, gt_boxes)

                track_gt_inds_to_embed_idx: Dict[int, int] = {}
                gt_inds_frame = gt_inds[0][0]
                for embed_idx, gt_idx in enumerate(matched_idx):
                    if 0 <= int(gt_idx) < len(gt_inds_frame):
                        track_gt_inds_to_embed_idx[int(gt_inds_frame[int(gt_idx)])] = int(embed_idx)

                result_track["track_gt_inds_to_embed_idx"] = track_gt_inds_to_embed_idx

        result_seg = None
        for meta in img_metas:
            if 'pts_filename' not in meta:
                meta['pts_filename'] = meta.get('token', 'dummy') + '.bin'
        if gt_lane_masks is not None and gt_lane_masks.dim() == 4:
            gt_lane_masks = gt_lane_masks.unsqueeze(1)

        if gt_lane_labels is not None and gt_lane_labels.dim() == 2:
            gt_lane_labels = gt_lane_labels.unsqueeze(1)
        if self.seg_head is not None:
            seg_results = self.seg_head.forward_test(
                pts_feats=bev_hwbc,
                gt_lane_labels=gt_lane_labels,
                gt_lane_masks=gt_lane_masks,
                img_metas=img_metas,
                rescale=rescale,
            )
            result_seg = seg_results[0] if isinstance(seg_results, (list, tuple)) else seg_results

        
        # if not hasattr(self, "_printed_test_keys"):
        #     self._printed_test_keys = True

        #     print("\n================ Track_Map_Former forward_test outputs ================")

        #     if isinstance(result_track, dict):
        #         print("[result_track] keys:")
        #         for k, v in result_track.items():
        #             if torch.is_tensor(v):
        #                 print(f"  - {k}: Tensor {tuple(v.shape)}")
        #             elif isinstance(v, list):
        #                 print(f"  - {k}: list (len={len(v)})")
        #             elif isinstance(v, dict):
        #                 print(f"  - {k}: dict (keys={list(v.keys())})")
        #             else:
        #                 print(f"  - {k}: {type(v)}")
        #     else:
        #         print("[result_track] type:", type(result_track))

        #     if result_seg is None:
        #         print("\n[result_seg] None")
        #     elif isinstance(result_seg, dict):
        #         print("\n[result_seg] keys:")
        #         for k, v in result_seg.items():
        #             if torch.is_tensor(v):
        #                 print(f"  - {k}: Tensor {tuple(v.shape)}")
        #             elif isinstance(v, list):
        #                 print(f"  - {k}: list (len={len(v)})")
        #             elif isinstance(v, dict):
        #                 print(f"  - {k}: dict (keys={list(v.keys())})")
        #             else:
        #                 print(f"  - {k}: {type(v)}")

        #     print("=====================================================================\n")
    
            # ================ Track_Map_Former forward_test outputs ================
            # [result_track] keys:
            #   - boxes_3d: <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
            #   - scores_3d: Tensor (6,)
            #   - labels_3d: Tensor (6,)
            #   - track_scores: Tensor (6,)
            #   - bbox_index: Tensor (6,)
            #   - track_ids: Tensor (6,)
            #   - mask: Tensor (6,)
            #   - track_bbox_results: list (len=1)
            #   - track_query_embeddings: Tensor (6, 256) # dynamic track query embeddings after update
            #   - track_query_matched_idxes: Tensor (6,)
            #   - img_feat_2D: Tensor (1, 6, 256, 32, 88)
            #   - track_instances_fordet: <class 'projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance.Instances'>
            #   - sdc_boxes_3d: <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
            #   - sdc_scores_3d: Tensor (1,)
            #   - sdc_track_scores: Tensor (1,)
            #   - sdc_track_bbox_results: list (len=1)
            #   - sdc_embedding: Tensor (256,)

            # [result_seg] keys:
            #   - pts_bbox: dict (keys=['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list', 'output_query_things', 'output_query_stuff', 'chosen_output_query_things'])
            #   - ret_iou: dict (keys=['drivable_intersection', 'drivable_union', 'lanes_intersection', 'lanes_union', 'divider_intersection', 'divider_union', 'crossing_intersection', 'crossing_union', 'contour_intersection', 'contour_union', 'drivable_iou', 'lanes_iou', 'divider_iou', 'crossing_iou', 'contour_iou'])
            #   - args_tuple: list (len=7)
            #   - output_query_things: Tensor (100, 256)
            #   - output_query_stuff: Tensor (1, 256)
            #   - chosen_output_query_things: Tensor (9, 256) # also dynamic map queries after update
            # =====================================================================


            # [Track_Map_Former]
            #   keys: dict_keys(['result_track', 'result_seg'])
            #   track queries: torch.Size([6, 256])
            #   map queries: torch.Size([9, 256])
            
        return dict(
            result_track=result_track,
            result_seg=result_seg,
        )
