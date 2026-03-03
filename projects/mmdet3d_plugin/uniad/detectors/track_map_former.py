import copy
import torch
from mmdet.models import DETECTORS, build_head
from .uniad_track import UniADTrack
from typing import Dict
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import auto_fp16
from .uniad_e2e import match_bbox
from ..dense_heads.track_head_plugin import Instances

#referencing OpenDriveVLA/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py for track logic
#referencing OpenDriveVLA/projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py for map seg logic
#referencing OpenDriveVLA/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py for multi-task training logic and loss weighting

@DETECTORS.register_module()
class Track_Map_Former(UniADTrack):
    """
    Track + Map (Seg) joint detector.
    Inherits ALL tracking logic from UniADTrack.
    """

    def __init__(
        self,
        seg_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.seg_head = build_head(seg_head) if seg_head is not None else None

        self.task_loss_weight = task_loss_weight

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def get_history_bev(self, bev_embeds_queue, img_metas_list):
        self.eval()
        with torch.no_grad():
            prev_bev = None
            #assuming bs=1
            bs, len_queue, C, H, W = bev_embeds_queue.shape
            bev_embeds_queue = bev_embeds_queue.reshape(bs * len_queue, C, H, W)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                this_bev_embed = bev_embeds_queue[i]
                # Convert [C, H, W] to [H*W, 1, C] for get_bev_embed_with_history
                this_bev_embed = this_bev_embed.flatten(1).permute(1, 0).unsqueeze(1)  # [C, H*W] -> [H*W, C] -> [H*W, 1, C]
                prev_bev, _ = self.pts_bbox_head.get_bev_embed_with_history(
                    current_bev_embed=this_bev_embed, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    def get_bevs(self, bev_embed, img_metas, prev_bev=None, past_bev=None, prev_img_metas=None):
        if past_bev is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(past_bev, prev_img_metas)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_embed_with_history(
                    current_bev_embed=bev_embed, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_embed_with_history(
                    current_bev_embed=bev_embed, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)
    
    @auto_fp16(apply_to=("bev_embed", "prev_bev"))
    def _forward_single_frame_train(
        self,
        bev_embed,
        past_bev,
        prev_img_metas,
        img_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        if past_bev is not None:
            bev_embed = self.get_bevs(
                bev_embed,
                img_metas,
                past_bev=past_bev,
                prev_img_metas=prev_img_metas
            )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            # "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        # out["img_feat_2D"] = img_feat_2D
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(self,
                            bev_embed,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = bev_embed.size(0)
        # init gt instances!
        gt_instances_list = []

        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(bev_embed.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)
            
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(bev_embed.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        for i in range(num_frame):
            # img_single = torch.stack([img_[i] for img_ in img], dim=0)
            #bev_embeds do not contain history
            past_bev = bev_embed[:i, ...] if i != 0 else None
            
            # Convert past_bev from [i, H*W, 1, C] to [1, i, C, H, W] for get_history_bev
            if past_bev is not None:
                len_queue, num_query, bs, embed_dims = past_bev.shape
                H = self.pts_bbox_head.bev_h
                W = self.pts_bbox_head.bev_w
                # [i, H*W, 1, C] -> [i, H*W, C] -> [i, C, H*W] -> [i, C, H, W] -> [1, i, C, H, W]
                past_bev = past_bev.squeeze(2).permute(0, 2, 1).reshape(len_queue, embed_dims, H, W).unsqueeze(0)
            
            prev_img_metas = copy.deepcopy(img_metas)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            frame_res = self._forward_single_frame_train(
                bev_embed[i, ...],
                past_bev,
                prev_img_metas,
                img_metas_single,
                track_instances,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]
        
        get_keys = ["bev_embed",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        get_keys += ["track_instances"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        
        return losses, out

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(
        self,
        bev_embed=None,
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
        #planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        **kwargs):
        losses = dict()
        len_queue = bev_embed.size(0)

        losses_track, outs_track = self.forward_track_train(bev_embed, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]

        img_metas = [each[len_queue-1] for each in img_metas]
        
        # Extract last frame for GT lane data (matching img_metas extraction)
        if gt_lane_labels and len(gt_lane_labels) > 0 and isinstance(gt_lane_labels[0], list):
            gt_lane_labels = [each[len_queue-1] for each in gt_lane_labels]
            gt_lane_bboxes = [each[len_queue-1] for each in gt_lane_bboxes]
            gt_lane_masks = [each[len_queue-1] for each in gt_lane_masks]

        if self.with_seg_head:
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)
        
        # results_for_vlm = self.get_results_for_vlm(img_metas[0], outs_track, outs_seg[0], sdc_planning[0], sdc_planning_mask[0], command[0], in_uniad_train=True, **kwargs)
        return losses #, results_for_vlm

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def _forward_single_frame_track_inference(
        self,
        bev_embed,
        img_metas,
        track_instances,
        prev_bev,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ): 
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        #bev_embed, bev_pos, img_feat_2D = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        #getting bev_embed directly from bevfusion now
        #print("track_instances.query: ", track_instances.query)
        #print("track_instances.ref_pts: ", track_instances.ref_pts)
        bev_embed = self.get_bevs(bev_embed, img_metas, prev_bev=prev_bev)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            #"bev_pos": bev_pos, only used in motion former, not needed
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, 0.5)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        #out["img_feat_2D"] = img_feat_2D no longer needed by map seg former
        return out


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
        print("bev_embed shape:", bev_embed.shape)  # debug print
        print("self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w:", self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w)  # debug print

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

        print("bev_hwbc.shape: ", bev_hwbc.shape)  # debug print
        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):  
            print("*****WARNING starting new track WARNING*****")
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
        else:
            print("using old track")
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_track_inference(
            bev_hwbc,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]
        #print("track_instances_fordet:", track_instances_fordet)  # debug print
        #print("track_instances:", track_instances)  # debug print

        self.test_track_instances = track_instances
        result_track = [dict()]
        get_keys = ["bev_embed",
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        #get_keys += ["img_feat_2D"]
        get_keys += ["track_instances_fordet"]
        get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results"]
        if self.with_motion_head:
            get_keys += ["sdc_embedding"]
        result_track[0].update({k: frame_res[k] for k in get_keys})
        result_track = self._det_instances2results(track_instances_fordet, result_track, img_metas)

        # normalize for local inspection (keep return value as list for callers)
        rt = result_track[0] if isinstance(result_track, list) and len(result_track) > 0 else result_track

        # --- debug & sanity checks (helps explain "AMOTA==0" during evaluation) ---
        if rt is None:
            print(f"[WARN] empty track result for scene={img_metas[0].get('scene_token','N/A')}")
        else:
            try:
                # count predicted boxes (if any)
                num_boxes = 0
                if 'track_bbox_results' in rt and rt['track_bbox_results']:
                    tb = rt['track_bbox_results']
                    if isinstance(tb, list) and len(tb) and isinstance(tb[0], list) and len(tb[0]):
                        num_boxes = tb[0][0].tensor.shape[0]
                track_ids = rt.get('track_ids', None)
                track_scores = rt.get('track_scores', None)
                min_score = None
                max_score = None
                if track_scores is not None:
                    try:
                        min_score = float(track_scores.min())
                        max_score = float(track_scores.max())
                    except Exception:
                        min_score, max_score = None, None
                print(f"[DEBUG] scene={img_metas[0].get('scene_token','N/A')}, num_boxes={num_boxes} track_ids_len={0 if track_ids is None else len(track_ids)} track_scores_minmax={(min_score,max_score)}")
                print(f"[DEBUG] track_ids: {track_ids}")
                print(f"[DEBUG] track_scores: {track_scores}")
                print(f"[DEBUG] track_bbox_results: {rt.get('track_bbox_results', None)}")
                # quick consistency check
                if 'track_ids' in rt and 'track_bbox_results' in rt and num_boxes>0:
                    tid_len = len(rt['track_ids']) if hasattr(rt['track_ids'], '__len__') else 0
                    if tid_len != num_boxes:
                        print(f"[WARN] mismatch track_ids ({tid_len}) vs boxes ({num_boxes}) — tracker -> eval mismatch")
                if 'track_scores' in rt and track_scores is not None:
                    # check score range
                    try:
                        if (track_scores < 0).any() or (track_scores > 1).any():
                            print(f"[WARN] track_scores out of [0,1] range; this can break thresholding in evaluator")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[DEBUG] failed to introspect result_track: {e}")

        if gt_bboxes_3d is not None and gt_inds is not None:
            # use normalized rt for safety
            if rt is None:
                detected_boxes3d = None
            else:
                detected_boxes3d = rt["track_bbox_results"][0][0].tensor  # LiDARInstance3DBoxes.tensor
            if detected_boxes3d is not None and detected_boxes3d.shape[0] > 0:
                detected_bboxes = detected_boxes3d.to(gt_bboxes_3d[0][0][0].tensor)[:-1, :7]  # drop sdc

                gt_boxes = gt_bboxes_3d[0][0][0].tensor[:, :7]
                matched_idx, iou_matrix = match_bbox(detected_bboxes, gt_boxes)

                track_gt_inds_to_embed_idx: Dict[int, int] = {}
                gt_inds_frame = gt_inds[0][0]
                for embed_idx, gt_idx in enumerate(matched_idx):
                    if 0 <= int(gt_idx) < len(gt_inds_frame):
                        track_gt_inds_to_embed_idx[int(gt_inds_frame[int(gt_idx)])] = int(embed_idx)

                # attach to the dict form for downstream usage
                if isinstance(result_track, list) and len(result_track) > 0 and isinstance(result_track[0], dict):
                    result_track[0]["track_gt_inds_to_embed_idx"] = track_gt_inds_to_embed_idx
                elif isinstance(result_track, dict):
                    result_track["track_gt_inds_to_embed_idx"] = track_gt_inds_to_embed_idx
                else:
                    # keep backward-compatible: add to top-level return dict later
                    pass

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