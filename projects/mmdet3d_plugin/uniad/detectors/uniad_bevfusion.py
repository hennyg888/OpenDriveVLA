import torch
from torch import nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models import build_detector

from ...models.builder import build_fusion_model


@DETECTORS.register_module()
class UniADBevFusion(nn.Module):
    """
    BEVFusion -> Track_Map_Former.
    """
    def __init__(
        self,
        bevfusion=None,
        track_map_former=None,
        bev_in_hw=180,
        bev_out_hw=200,
        freeze_bevfusion=False,
        freeze_bevfusion_bn=False,
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.freeze_bevfusion = freeze_bevfusion
        self.freeze_bevfusion_bn = freeze_bevfusion_bn
        self.bev_in_hw = bev_in_hw
        self.bev_out_hw = bev_out_hw
        self.bev_linear = nn.Linear(bev_in_hw * bev_in_hw, bev_out_hw * bev_out_hw)

        self.bevfusion = (
            build_fusion_model(bevfusion, train_cfg=train_cfg, test_cfg=test_cfg)
            if bevfusion is not None
            else None
        )
        self.track_map_former = (
            build_detector(track_map_former, train_cfg=train_cfg, test_cfg=test_cfg)
            if track_map_former is not None
            else None
        )

        if self.freeze_bevfusion:
            self._freeze_bevfusion()

    def init_weights(self):
        if self.bevfusion is not None:
            self.bevfusion.init_weights()
        if hasattr(self.track_map_former, "init_weights"):
            self.track_map_former.init_weights()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bevfusion and self.bevfusion is not None:
            if self.freeze_bevfusion_bn:
                self.bevfusion.eval()
            else:
                self.bevfusion.train(False)
        return self

    def _freeze_bevfusion(self):
        if self.freeze_bevfusion_bn:
            self.bevfusion.eval()
        for param in self.bevfusion.parameters():
            param.requires_grad = False

    def _resize_bev_feat(self, bev_feat):
        if bev_feat is None:
            return None
        if bev_feat.dim() == 5:
            # B, N, C, H, W -> B, C, HW
            b, n, c, h, w = bev_feat.shape
            bev_feat = bev_feat.view(b, n * c, h, w)
            bev_feat = self._resize_bev_feat(bev_feat)
            bev_feat = bev_feat.view(b, n, c, self.bev_out_hw, self.bev_out_hw)
            return bev_feat
        if bev_feat.dim() == 4:
            # B, C, H, W -> B, C, HW
            b, c, h, w = bev_feat.shape
            if h == self.bev_out_hw and w == self.bev_out_hw:
                return bev_feat
            bev_flat = bev_feat.view(b, c, h * w)
            bev_flat = self.bev_linear(bev_flat)
            return bev_flat.view(b, c, self.bev_out_hw, self.bev_out_hw)
        if bev_feat.dim() == 3:
            # HW, B, C -> B, C, HW
            hw, b, c = bev_feat.shape
            if hw == self.bev_out_hw * self.bev_out_hw:
                return bev_feat
            bev_flat = bev_feat.permute(1, 2, 0).contiguous()
            bev_flat = self.bev_linear(bev_flat)
            return bev_flat.view(b, c, self.bev_out_hw, self.bev_out_hw)
        raise ValueError(f"Unsupported bev_feat shape: {bev_feat.shape}")

    def extract_feat(
        self,
        img=None,
        points=None,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        img_metas=None,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if self.bevfusion is None:
            raise RuntimeError("bevfusion is required for feature extraction.")

        if self.freeze_bevfusion:
            with torch.no_grad():
                bev_feat, img_feat_2d = self.bevfusion(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    img_metas,
                    depths,
                    radar=radar,
                    gt_masks_bev=gt_masks_bev,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    **kwargs,
                )
        else:
            bev_feat, img_feat_2d = self.bevfusion(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                depths,
                radar=radar,
                gt_masks_bev=gt_masks_bev,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                **kwargs,
            )

        bev_feat = self._resize_bev_feat(bev_feat)
        return bev_feat, img_feat_2d

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    @auto_fp16(apply_to=("img", "points"))
    def forward_train(
        self,
        img=None,
        points=None,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        img_metas=None,
        depths=None,
        radar=None,
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
        **kwargs,
    ):
        bev_feats = []
        img_feat_2ds = []
        for i in range(self.track_map_former.queue_length):
            bev_feat, img_feat_2d = self.extract_feat(
                img=img[i],
                points=points[i],
                camera2ego=camera2ego[i],
                lidar2ego=lidar2ego[i],
                lidar2camera=lidar2camera[i],
                lidar2image=lidar2image[i],
                camera_intrinsics=camera_intrinsics[i],
                camera2lidar=camera2lidar[i],
                img_aug_matrix=img_aug_matrix[i],
                lidar_aug_matrix=lidar_aug_matrix[i],
                img_metas=img_metas[i],
                depths=depths[i],  
                radar=radar[i],
                gt_bboxes_3d=gt_bboxes_3d[i],
                gt_labels_3d=gt_labels_3d[i],
                **kwargs,
            )
            bev_feats.append(bev_feat)
            img_feat_2ds.append(img_feat_2d)

        bev_feat = torch.stack(bev_feats, dim=0)
        img_feat_2d = torch.stack(img_feat_2ds, dim=0)

        if self.track_map_former is None:
            raise RuntimeError("track_map_former is required for forward_train.")

        losses = self.track_map_former.forward(
            bev_embed=bev_feat,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            timestamp=timestamp,
            gt_lane_labels=gt_lane_labels,
            gt_lane_bboxes=gt_lane_bboxes,
            gt_lane_masks=gt_lane_masks,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
            **kwargs,
        )
        return losses

    @auto_fp16(apply_to=("img", "points"))
    def forward_test(
        self,
        img=None,
        points=None,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        img_metas=None,
        depths=None,
        radar=None,
        rescale=False,
        **kwargs,
    ):
        bev_feat, img_feat_2d = self.extract_feat(
            img=img,
            points=points,
            camera2ego=camera2ego,
            lidar2ego=lidar2ego,
            lidar2camera=lidar2camera,
            lidar2image=lidar2image,
            camera_intrinsics=camera_intrinsics,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            img_metas=img_metas,
            depths=depths,
            radar=radar,
            **kwargs,
        )

        if self.track_map_former is None:
            raise RuntimeError("track_map_former is required for forward_test.")

        return self.track_map_former.forward_test(
            bev_embed=bev_feat,
            img_feat_2D=img_feat_2d,
            img_metas=img_metas,
            rescale=rescale,
            **kwargs,
        )

    def simple_test(self, *args, **kwargs):
        pass

    def aug_test(self, *args, **kwargs):
        pass

