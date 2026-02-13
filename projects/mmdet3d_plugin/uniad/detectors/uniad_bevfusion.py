import torch
from collections import OrderedDict
import torch.distributed as dist
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
        # Use bilinear interpolation instead of linear layer to save memory
        # No trainable parameters, purely geometric transformation

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
        """
        Resize BEV features using bilinear interpolation (no trainable parameters).
        Supports multiple input formats: [B, C, H, W], [B, N, C, H, W], or [HW, B, C].
        """
        if bev_feat is None:
            return None
        
        if bev_feat.dim() == 5:
            # B, N, C, H, W -> B, N*C, H, W
            b, n, c, h, w = bev_feat.shape
            if h == self.bev_out_hw and w == self.bev_out_hw:
                return bev_feat
            bev_feat = bev_feat.view(b, n * c, h, w)
            # Bilinear interpolate: [B, N*C, H, W] -> [B, N*C, H', W']
            bev_feat = torch.nn.functional.interpolate(
                bev_feat, size=(self.bev_out_hw, self.bev_out_hw),
                mode='bilinear', align_corners=False
            )
            bev_feat = bev_feat.view(b, n, c, self.bev_out_hw, self.bev_out_hw)
            return bev_feat
        
        if bev_feat.dim() == 4:
            # B, C, H, W
            b, c, h, w = bev_feat.shape
            if h == self.bev_out_hw and w == self.bev_out_hw:
                return bev_feat
            # Bilinear interpolate: [B, C, H, W] -> [B, C, H', W']
            return torch.nn.functional.interpolate(
                bev_feat, size=(self.bev_out_hw, self.bev_out_hw),
                mode='bilinear', align_corners=False
            )
        
        if bev_feat.dim() == 3:
            # HW, B, C -> B, C, H, W
            hw, b, c = bev_feat.shape
            if hw == self.bev_out_hw * self.bev_out_hw:
                return bev_feat
            h = w = int(hw ** 0.5)
            assert h * w == hw, f"Cannot reshape {hw} to square"
            bev_feat = bev_feat.permute(1, 2, 0).contiguous()  # [B, C, HW]
            bev_feat = bev_feat.view(b, c, h, w)  # [B, C, H, W]
            # Bilinear interpolate
            bev_feat = torch.nn.functional.interpolate(
                bev_feat, size=(self.bev_out_hw, self.bev_out_hw),
                mode='bilinear', align_corners=False
            )
            # [B, C, H', W'] -> [H'*W', B, C]
            bev_feat = bev_feat.view(b, c, -1).permute(2, 0, 1).contiguous()
            return bev_feat
        
        raise ValueError(f"Unsupported bev_feat shape: {bev_feat.shape}")

    def _extract_calib_from_metas(self, img_metas):
        """
        Extract camera2ego / lidar2ego / lidar2camera / lidar2image /
        camera_intrinsics / camera2lidar / img_aug_matrix / lidar_aug_matrix
        from img_metas.

        Supports:
        - Training (NuScenesE2EDataset + union2one): img_metas[0] is a
          dict indexed by frame idx: {0: meta_0, 1: meta_1, ...}
        - Single-frame test: img_metas[0] is a meta dict.
        """
        assert img_metas is not None, "img_metas is required to extract calibration."

        # bs=1 is assumed in UniAD / Track_Map_Former
        if isinstance(img_metas, (list, tuple)):
            assert len(img_metas) == 1, "Only batch size = 1 is supported here."
            metas = img_metas[0]
        else:
            metas = img_metas

        # Multi-frame case: metas is a dict indexed by frame idx
        if isinstance(metas, dict) and 0 in metas:
            num_frames = self.track_map_former.queue_length
            frames = [metas[i] for i in range(num_frames)]

            camera2ego = [f["camera2ego"] for f in frames]
            lidar2ego = [f["lidar2ego"] for f in frames]
            lidar2camera = [f["lidar2camera"] for f in frames]
            lidar2image = [f["lidar2image"] for f in frames]
            camera_intrinsics = [f["camera_intrinsics"] for f in frames]
            camera2lidar = [f["camera2lidar"] for f in frames]
            img_aug_matrix = [f["img_aug_matrix"] for f in frames]
            lidar_aug_matrix = [f["lidar_aug_matrix"] for f in frames]
        else:
            # Single-frame meta dict
            f = metas
            camera2ego = f["camera2ego"]
            lidar2ego = f["lidar2ego"]
            lidar2camera = f["lidar2camera"]
            lidar2image = f["lidar2image"]
            camera_intrinsics = f["camera_intrinsics"]
            camera2lidar = f["camera2lidar"]
            img_aug_matrix = f["img_aug_matrix"]
            lidar_aug_matrix = f["lidar_aug_matrix"]

        return dict(
            camera2ego=camera2ego,
            lidar2ego=lidar2ego,
            lidar2camera=lidar2camera,
            lidar2image=lidar2image,
            camera_intrinsics=camera_intrinsics,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
        )

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

        # Normalize calibration parameters to BEVFusion expected format:
        # Per-camera params: list of N_cam tensors [tensor0, tensor1, ..., tensorN]
        # Single params: just a tensor (NOT a list)
        
        device = img.device if img is not None else 'cuda'
        
        def to_list_of_tensors(param):
            """
            Convert list of N_cam arrays to list of N_cam tensors
            """
            if param is None:
                return None
            if isinstance(param, list):
                return [torch.as_tensor(x, device=device) if not torch.is_tensor(x) else x.to(device) 
                       for x in param]
            if isinstance(param, torch.Tensor):
                # Already a stacked tensor [N_cam, ...], split into list
                return [param[i].to(device) for i in range(param.shape[0])]
            return param
        
        def to_tensor(param):
            """
            Convert single array/tensor to tensor (NOT a list)
            """
            if param is None:
                return None
            if not torch.is_tensor(param):
                return torch.as_tensor(param, device=device)
            return param.to(device)
        
        # Per-camera parameters: list of N_cam tensors, _to_BN44 will handle conversion
        camera2ego = to_list_of_tensors(camera2ego)
        lidar2camera = to_list_of_tensors(lidar2camera)
        lidar2image = to_list_of_tensors(lidar2image)
        camera_intrinsics = to_list_of_tensors(camera_intrinsics)
        camera2lidar = to_list_of_tensors(camera2lidar)
        img_aug_matrix = to_list_of_tensors(img_aug_matrix)
        
        # Single parameters: just single tensors, _to_BN44 will handle conversion to [B, 4, 4]
        lidar2ego = to_tensor(lidar2ego)
        lidar_aug_matrix = to_tensor(lidar_aug_matrix)

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
            # Detach to completely cut off gradient graph from BEVFusion
            bev_feat = bev_feat.detach()
            if img_feat_2d is not None:
                img_feat_2d = img_feat_2d.detach()
            
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

        # Apply bilinear interpolation to resize BEV features (no trainable parameters)
        bev_feat = self._resize_bev_feat(bev_feat)
        
        return bev_feat, img_feat_2d

    def forward(self, return_loss=True, **kwargs):
        # Unpack calibration matrices from img_metas if not explicitly provided
        img_metas = kwargs.get("img_metas", None)
        needs_calib = (
            "camera2ego" not in kwargs
            or kwargs["camera2ego"] is None
        )
        if img_metas is not None and needs_calib and self.bevfusion is not None:
            calib = self._extract_calib_from_metas(img_metas)
            kwargs.update(calib)

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
        # Unwrap nested data structures from [[item0, item1, ...]] to [item0, item1, ...]
        # Many parameters come as [[frames...]] due to DataContainer collation
        def unwrap_if_nested(param, expected_len):
            if param is not None and isinstance(param, (list, tuple)):
                if len(param) == 1 and isinstance(param[0], (list, tuple)) and len(param[0]) == expected_len:
                    return param[0]
            return param
        
        queue_len = self.track_map_former.queue_length
        
        # Unwrap for loop indexing, but keep originals for track_map_former
        points_unwrapped = unwrap_if_nested(points, queue_len)
        gt_bboxes_3d_unwrapped = unwrap_if_nested(gt_bboxes_3d, queue_len)
        gt_labels_3d_unwrapped = unwrap_if_nested(gt_labels_3d, queue_len)
        gt_inds_unwrapped = unwrap_if_nested(gt_inds, queue_len)
        
        # Detach img if bevfusion is frozen (already done in previous optimization)
        # Detach calibration matrices to save memory (they don't need gradients)
        if camera2ego is not None:
            camera2ego = [c.detach() if torch.is_tensor(c) else c for c in camera2ego]
        if lidar2ego is not None:
            lidar2ego = [l.detach() if torch.is_tensor(l) else l for l in lidar2ego]
        if lidar2camera is not None:
            lidar2camera = [l.detach() if torch.is_tensor(l) else l for l in lidar2camera]
        if lidar2image is not None:
            lidar2image = [l.detach() if torch.is_tensor(l) else l for l in lidar2image]
        if camera_intrinsics is not None:
            camera_intrinsics = [c.detach() if torch.is_tensor(c) else c for c in camera_intrinsics]
        if camera2lidar is not None:
            camera2lidar = [c.detach() if torch.is_tensor(c) else c for c in camera2lidar]
        if img_aug_matrix is not None:
            img_aug_matrix = [i.detach() if torch.is_tensor(i) else i for i in img_aug_matrix]
        if lidar_aug_matrix is not None:
            lidar_aug_matrix = [l.detach() if torch.is_tensor(l) else l for l in lidar_aug_matrix]
        
        # Detach img if bevfusion is frozen to save memory (no gradients needed)
        if self.freeze_bevfusion and img is not None:
            img = img.detach()
        
        bev_feats = []
        
        # Extract metas_map for multi-frame indexing
        # img_metas is typically [metas_map] where metas_map = {0: meta0, 1: meta1, ...}
        if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
            metas_map = img_metas[0]
            if isinstance(metas_map, dict) and 0 in metas_map:
                # Multi-frame case
                use_frame_dict = True
            else:
                # Single frame case or already frame meta
                use_frame_dict = False
        else:
            metas_map = img_metas
            use_frame_dict = False
        
        for i in range(queue_len):
            # Slice along queue_length dimension, keep batch dimension
            # img: [batch_size, queue_length, N_cam, C, H, W] -> img[:, i] -> [batch_size, N_cam, C, H, W]
            # points: use unwrapped version for indexing
            cur_img = img[:, i] if img is not None else None
            if points_unwrapped is not None:
                frame_points = points_unwrapped[i]
                
                # BEVFusion expects: list of batch_size point clouds (each a single tensor)
                # After unwrapping, frame_points might be:
                # 1. A single point cloud tensor -> wrap in list [tensor]
                # 2. Still needs handling
                
                if isinstance(frame_points, (list, tuple)):
                    # frame_points is a list, could be nested batch structure
                    # Flatten to get the actual point cloud tensor
                    while isinstance(frame_points, (list, tuple)) and len(frame_points) == 1:
                        frame_points = frame_points[0]
                    
                    if torch.is_tensor(frame_points):
                        # After unwrapping, got a tensor
                        cur_points = [frame_points]
                    else:
                        # Shouldn't reach here, but handle gracefully
                        cur_points = frame_points if isinstance(frame_points, (list, tuple)) else [frame_points]
                else:
                    # Single point cloud tensor
                    cur_points = [frame_points]
            else:
                cur_points = None
            
            # Get frame i's img_metas and wrap in list for BEVFusion
            if use_frame_dict:
                cur_img_metas = [metas_map[i]]  # BEVFusion expects a list
            else:
                cur_img_metas = img_metas if i == 0 else None  # fallback
            
            bev_feat, _ = self.extract_feat(
                img=cur_img,
                points=cur_points,
                camera2ego=camera2ego[i],
                lidar2ego=lidar2ego[i],
                lidar2camera=lidar2camera[i],
                lidar2image=lidar2image[i],
                camera_intrinsics=camera_intrinsics[i],
                camera2lidar=camera2lidar[i],
                img_aug_matrix=img_aug_matrix[i],
                lidar_aug_matrix=lidar_aug_matrix[i],
                img_metas=cur_img_metas,
                depths=depths[i] if depths is not None else None,
                radar=radar[i] if radar is not None else None,
                gt_bboxes_3d=gt_bboxes_3d_unwrapped[i] if gt_bboxes_3d_unwrapped is not None else None,
                gt_labels_3d=gt_labels_3d_unwrapped[i] if gt_labels_3d_unwrapped is not None else None,
                **kwargs,
            )
            # extract_feat returns [1, C, H, W], remove batch dim for stacking
            if bev_feat.dim() == 4 and bev_feat.shape[0] == 1:
                bev_feat = bev_feat.squeeze(0)  # [C, H, W]
            bev_feats.append(bev_feat)
            
            # Clear intermediate variables to free memory
            del cur_img, cur_points

        # Stack: [queue_length, C, H, W]
        bev_feat = torch.stack(bev_feats, dim=0)
        
        # Delete intermediate list immediately
        del bev_feats
        
        # Clear cache to free memory after BEVFusion forward (BEVFusion is freeze and detached)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reshape for track_map_former: [queue_length, C, H, W] -> [queue_length, H*W, 1, C]
        # The extra dimension is batch (=1), needed for transformer decoder
        queue_length, C, H, W = bev_feat.shape
        bev_embed = bev_feat.flatten(2).permute(0, 2, 1).unsqueeze(2)  # [queue_length, H*W, 1, C]
        
        # Delete bev_feat to save memory before converting to FP16
        del bev_feat
        
        # Keep in FP32 for numerical stability (FP16 causes NaN in loss)
        bev_embed = bev_embed.contiguous()

        if self.track_map_former is None:
            raise RuntimeError("track_map_former is required for forward_train.")

        # Use original (non-unwrapped) parameters for track_map_former
        # They already have the correct [[frames...]] format that track_map_former expects
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

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
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
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def simple_test(self, *args, **kwargs):
        pass

    def aug_test(self, *args, **kwargs):
        pass

