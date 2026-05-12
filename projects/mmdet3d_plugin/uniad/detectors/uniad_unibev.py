import torch
import torch.utils.checkpoint
from collections import OrderedDict
import torch.distributed as dist
from torch import nn
from mmcv.runner import auto_fp16
from mmcv.runner import load_checkpoint
from mmdet.models import DETECTORS
from mmdet3d.models import build_detector


@DETECTORS.register_module()
class UniADUniBEV(nn.Module):
    """
    UniADUniBEV: UniBEV BEV encoder -> Track_Map_Former (track + seg heads).

    UniBEV produces fused BEV embeddings [bs, H*W, 256] from camera and LiDAR.
    These are fed directly to Track_Map_Former via the external_bev path,
    bypassing the BEVFormerTrackHead's internal encoder entirely.
    """

    def __init__(
        self,
        unibev=None,
        track_map_former=None,
        pretrained_unibev=None,
        freeze_unibev=True,
        freeze_unibev_bn=True,
        use_checkpoint=False,
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.freeze_unibev = freeze_unibev
        self.freeze_unibev_bn = freeze_unibev_bn
        self.use_checkpoint = use_checkpoint
        self.pretrained_unibev = pretrained_unibev

        self.unibev = (
            build_detector(unibev)
            if unibev is not None
            else None
        )
        self.track_map_former = (
            build_detector(track_map_former, train_cfg=train_cfg, test_cfg=test_cfg)
            if track_map_former is not None
            else None
        )

        if self.freeze_unibev:
            self._freeze_unibev()

    def init_weights(self):
        """Load UniBEV checkpoint; Track_Map_Former is randomly initialized."""
        if self.unibev is not None and self.pretrained_unibev is not None:
            # UniBEV checkpoint keys have no prefix; add 'unibev.' to match our state dict
            ckpt = torch.load(self.pretrained_unibev, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # Remap: 'key' -> 'unibev.key'
            remapped = {'unibev.' + k: v for k, v in state_dict.items()}
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            print(f'[UniADUniBEV] Loaded UniBEV checkpoint: '
                  f'{len(missing)} missing, {len(unexpected)} unexpected keys')
        elif self.unibev is not None:
            self.unibev.init_weights()

        if self.track_map_former is not None and hasattr(self.track_map_former, 'init_weights'):
            self.track_map_former.init_weights()

    def train(self, mode=True):
        super().train(mode)
        if self.unibev is not None:
            if self.freeze_unibev:
                # Keep UniBEV in eval mode so BN stats are frozen
                self.unibev.eval()
            elif self.freeze_unibev_bn:
                for m in self.unibev.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        if m.weight is not None:
                            m.weight.requires_grad = False
                        if m.bias is not None:
                            m.bias.requires_grad = False
        return self

    def _freeze_unibev(self):
        """Freeze all UniBEV parameters."""
        if self.unibev is not None:
            self.unibev.eval()
            for param in self.unibev.parameters():
                param.requires_grad = False

    def extract_feat(self, img, points, img_metas):
        """Extract fused BEV features using UniBEV encoder for a single frame.

        Args:
            img (Tensor | None): [bs, N_cam, C, H, W]
            points (list[Tensor] | None): list of point clouds, one per batch sample
            img_metas (list[dict]): image metadata

        Returns:
            Tensor: [H*W, bs, embed_dims] — matches Track_Map_Former external_bev format
        """
        if self.freeze_unibev:
            with torch.no_grad():
                img_feats = self.unibev.extract_img_feat(img, img_metas) if img is not None else None
                pts_feats = self.unibev.extract_pts_feat(points) if points is not None else None
                fused_bev = self.unibev.pts_bbox_head.encode_bev(img_feats, pts_feats, img_metas)
            # Detach completely from UniBEV computation graph
            fused_bev = fused_bev.detach()
        else:
            img_feats = self.unibev.extract_img_feat(img, img_metas) if img is not None else None
            pts_feats = self.unibev.extract_pts_feat(points) if points is not None else None
            fused_bev = self.unibev.pts_bbox_head.encode_bev(img_feats, pts_feats, img_metas)

        # fused_bev: [bs, H*W, embed_dims] -> permute to [H*W, bs, embed_dims]
        return fused_bev.permute(1, 0, 2).contiguous()

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img',))
    def forward_train(
        self,
        img=None,
        points=None,
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
        queue_len = self.track_map_former.queue_length

        # Unwrap [[item0, item1, ...]] -> [item0, item1, ...] from DataContainer collation
        def unwrap_if_nested(param, expected_len):
            if param is not None and isinstance(param, (list, tuple)):
                if len(param) == 1 and isinstance(param[0], (list, tuple)) and len(param[0]) == expected_len:
                    return param[0]
            return param

        points_unwrapped = unwrap_if_nested(points, queue_len)

        # Detach img when UniBEV is frozen (no gradient needed through image)
        if self.freeze_unibev and img is not None:
            img = img.detach()

        # Extract img_metas map for multi-frame indexing
        # NuScenesE2EDataset produces img_metas[0] = {0: meta_frame0, 1: meta_frame1, ...}
        if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
            metas_map = img_metas[0]
            use_frame_dict = isinstance(metas_map, dict) and 0 in metas_map
        else:
            metas_map = img_metas
            use_frame_dict = False

        bev_feat_all = None

        for i in range(queue_len):
            # img: [bs, queue_len, N_cam, C, H, W] -> [bs, N_cam, C, H, W]
            cur_img = img[:, i] if img is not None else None

            # Extract point cloud for frame i (batch size = 1)
            if points_unwrapped is not None:
                frame_points = points_unwrapped[i]
                # Flatten nested list wrapping to get the actual tensor
                while isinstance(frame_points, (list, tuple)) and len(frame_points) == 1:
                    frame_points = frame_points[0]
                cur_points = [frame_points] if torch.is_tensor(frame_points) else frame_points
            else:
                cur_points = None

            cur_img_metas = [metas_map[i]] if use_frame_dict else img_metas

            if not self.freeze_unibev and self.use_checkpoint:
                # Gradient checkpointing: saves activation memory at cost of recomputation
                def _ckpt_extract(img_f, _pts=cur_points, _metas=cur_img_metas):
                    return self.extract_feat(img_f, _pts, _metas)
                bev_frame = torch.utils.checkpoint.checkpoint(
                    _ckpt_extract, cur_img, use_reentrant=True)
            else:
                bev_frame = self.extract_feat(cur_img, cur_points, cur_img_metas)

            # bev_frame: [H*W, 1, C]
            if bev_feat_all is None:
                bev_feat_all = bev_frame.new_zeros(queue_len, *bev_frame.shape)
            bev_feat_all[i] = bev_frame

            del cur_img, cur_points

        if torch.cuda.is_available() and self.freeze_unibev:
            torch.cuda.empty_cache()

        # bev_feat_all: [queue_len, H*W, 1, C] — same layout as UniADBevFusion output
        # Track head expects bev_embed[i] = [H*W, 1, C], seg head expects [H*W, bs, C]
        bev_embed = bev_feat_all.contiguous()

        if self.track_map_former is None:
            raise RuntimeError('track_map_former is required for forward_train.')

        losses = self.track_map_former.forward(
            bev_embed=bev_embed,
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

    @auto_fp16(apply_to=('img',))
    def forward_test(
        self,
        img=None,
        points=None,
        img_metas=None,
        rescale=False,
        **kwargs,
    ):
        # Single-frame inference: flatten point list
        if isinstance(points, (list, tuple)):
            while isinstance(points, (list, tuple)) and len(points) == 1:
                points = points[0]
            cur_points = [points] if torch.is_tensor(points) else points
        else:
            cur_points = points

        cur_metas = img_metas[0] if isinstance(img_metas, (list, tuple)) else img_metas

        bev_feat = self.extract_feat(img, cur_points, [cur_metas] if isinstance(cur_metas, dict) else cur_metas)
        # bev_feat: [H*W, 1, C]

        if self.track_map_former is None:
            raise RuntimeError('track_map_former is required for forward_test.')

        return self.track_map_former.forward_test(
            bev_embed=bev_feat,
            img_metas=img_metas,
            rescale=rescale,
            **kwargs,
        )

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def simple_test(self, *args, **kwargs):
        pass

    def aug_test(self, *args, **kwargs):
        pass
