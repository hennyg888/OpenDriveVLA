_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

queue_length = 5
file_client_args = dict(backend="disk")

# Unified point_cloud_range matching UniBEV checkpoint
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
patch_size = [108.0, 108.0]  # 2 * 54

# embed_dims=256 to match UniBEV output; Track_Map_Former trained from scratch
_dim_ = 256
_pos_dim_ = _dim_ // 2   # 128
_ffn_dim_ = _dim_ * 2    # 512
_num_levels_ = 1          # UniBEV uses 1 FPN level
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
past_steps = 4
fut_steps = 4
image_size = [256, 704]

seed = 0
deterministic = False
total_epochs = 20
max_epochs = 6

dataset_type = "NuScenesE2EDataset"
data_root = "data/nuscenes/"
info_root = "data/infos/"
ann_file_train = info_root + "nuscenes_infos_temporal_train.pkl"
ann_file_val = info_root + "nuscenes_infos_temporal_val.pkl"

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

gt_paste_stop_epoch = -1
load_dim = 5
use_dim = 5

class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
object_classes = class_names
map_classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']

occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)

planning_evaluation_strategy = "uniad"
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

# Caffe-style normalization to match UniBEV's ResNet-101 pretrained weights
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True),
)
augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5,
)

model = dict(
    type="UniADUniBEV",
    pretrained_unibev=None,  # Set to path of UniBEV checkpoint before training
    freeze_unibev=True,
    freeze_unibev_bn=True,
    use_checkpoint=False,  # No gradient checkpointing needed when UniBEV is frozen
    unibev=dict(
        type="UniBEV",
        use_grid_mask=True,
        pts_voxel_layer=dict(
            max_num_points=10,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(90000, 120000)),
        pts_voxel_encoder=dict(
            type='HardSimpleVFE',
            num_features=5),
        pts_middle_encoder=dict(
            type='SparseEncoder',
            in_channels=5,
            sparse_shape=[41, 1440, 1440],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32),
                              (32, 32, 64),
                              (64, 64, 128),
                              (128, 128)),
            encoder_paddings=((0, 0, 1),
                              (0, 0, 1),
                              (0, 0, [0, 1, 1]),
                              (0, 0)),
            block_type='basicblock'),
        pts_backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),
        pts_neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            upsample_strides=[1, 2],
            out_channels=[_dim_ // 2, _dim_ // 2],  # [128, 128] -> concat -> 256
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),
        img_backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type='BN2d', requires_grad=False),
            norm_eval=True,
            style='caffe',
            with_cp=True,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[2048],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=_num_levels_,
            relu_before_extra_convs=True),
        pts_bbox_head=dict(
            type='UniBEV_Head',
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=10,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type='UniBEVTransformer',
                embed_dims=_dim_,
                fusion_method='linear',
                drop_modality=0.5,
                feature_norm='ChannelNormWeights',
                img_encoder=dict(
                    type='ImgEncoder',
                    num_layers=3,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='ImgLayer',
                        attn_cfgs=[
                            dict(type='MultiScaleDeformableAttention',
                                 embed_dims=_dim_, num_levels=1),
                            dict(type='SpatialCrossAttentionImg',
                                 pc_range=point_cloud_range,
                                 deformable_attention=dict(
                                     type='MSDeformableAttention3DImg',
                                     embed_dims=_dim_, num_points=8,
                                     num_levels=_num_levels_),
                                 embed_dims=_dim_)
                        ],
                        ffn_cfgs=dict(type='FFN', embed_dims=_dim_),
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
                pts_encoder=dict(
                    type='PtsEncoder',
                    num_layers=3,
                    pc_range=point_cloud_range,
                    num_points_in_pillar_lidar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='PtsLayer',
                        attn_cfgs=[
                            dict(type='MultiScaleDeformableAttention',
                                 embed_dims=_dim_, num_levels=1),
                            dict(type='SpatialCrossAttentionPts',
                                 pc_range=point_cloud_range,
                                 deformable_attention=dict(
                                     type='MSDeformableAttention3DPts',
                                     embed_dims=_dim_, num_points=8,
                                     num_levels=_num_levels_),
                                 embed_dims=_dim_)
                        ],
                        ffn_cfgs=dict(type='FFN', embed_dims=_dim_),
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
                # Detection decoder is required by UniBEV_Head._init_layers()
                # but is never called via encode_bev() path
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(type='MultiheadAttention',
                                 embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type='CustomMSDeformableAttention',
                                 embed_dims=_dim_, num_levels=1),
                        ],
                        ffn_cfgs=dict(type='FFN', embed_dims=_dim_),
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-66.0, -66.0, -10.0, 66.0, 66.0, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_),
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True,
                gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
        train_cfg=dict(pts=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range)))),
    track_map_former=dict(
        type="Track_Map_Former",
        gt_iou_threshold=0.3,
        queue_length=queue_length,
        use_grid_mask=True,
        video_test_mode=True,
        num_query=900,
        num_classes=10,
        pc_range=point_cloud_range,
        img_backbone=None,
        img_neck=None,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        score_thresh=0.4,
        filter_score_thresh=0.35,
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-66.0, -66.0, -10.0, 66.0, 66.0, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=True,
            iou_thres=0.5,
        ),
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=True,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        loss_cfg=dict(
            type="ClipMatcher",
            num_classes=10,
            weight_dict=None,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3DTrack",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=point_cloud_range,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_past_traj_weight=0.0,
        ),
        pts_bbox_head=dict(
            type='BEVFormerTrackHead',
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=10,
            in_channels=_dim_,   # 256 — matches UniBEV embed_dims
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            past_steps=4,
            fut_steps=4,
            transformer=dict(
                type='PerceptionTransformer',
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                # encoder=None: disabled — BEV features come from UniBEV via external_bev path
                encoder=None,
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(type='MultiheadAttention',
                                 embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type='CustomMSDeformableAttention',
                                 embed_dims=_dim_, num_levels=1)
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_ffn_dim_,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True)),
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-66.0, -66.0, -10.0, 66.0, 66.0, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,   # 128
                row_num_embed=bev_h_,
                col_num_embed=bev_w_),
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True,
                gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
        seg_head=dict(
            type="PansegformerHead",
            bev_h=bev_h_,
            bev_w=bev_w_,
            canvas_size=canvas_size,
            pc_range=point_cloud_range,
            num_query=300,
            num_classes=4,
            num_things_classes=3,
            num_stuff_classes=1,
            in_channels=_dim_,   # 256
            sync_cls_avg_factor=True,
            as_two_stage=False,
            with_box_refine=True,
            transformer=dict(
                type="SegDeformableTransformer",
                encoder=dict(
                    type="DetrTransformerEncoder",
                    num_layers=6,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=dict(
                            type="MultiScaleDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                        feedforward_channels=_feed_dim_,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_feed_dim_,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=("self_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DeformableDetrTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention",
                                 embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type="MultiScaleDeformableAttention",
                                 embed_dims=_dim_, num_levels=1),
                        ],
                        feedforward_channels=_feed_dim_,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_feed_dim_,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            positional_encoding=dict(
                type="SinePositionalEncoding",
                num_feats=_dim_half_,   # 128
                normalize=True,
                offset=-0.5,
            ),
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True,
                gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            loss_mask=dict(type="DiceLoss", loss_weight=2.0),
            thing_transformer_head=dict(
                type="SegMaskHead", d_model=_dim_, nhead=8, num_decoder_layers=4),
            stuff_transformer_head=dict(
                type="SegMaskHead", d_model=_dim_, nhead=8, num_decoder_layers=6, self_attn=True),
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                assigner_with_mask=dict(
                    type="HungarianAssigner_multi_info",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                    mask_cost=dict(type="DiceCost", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                sampler_with_mask=dict(type="PseudoSampler_segformer"),
            ),
        ),
        task_loss_weight=dict(
            track=1.1,
            map=1.0,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# UniBEV detection decoder params are unused in encode_bev() path
find_unused_parameters = True

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D_E2E',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False,
         with_future_anns=True,
         with_ins_inds_3d=True,
         ins_inds_add_1=True),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=False,
        is_train=False),
    dict(
        type="GlobalRotScaleTrans_3D",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    # Caffe-style normalization for ResNet-101 pretrained backbone
    dict(type="ImageNormalize",
         mean=[103.530, 116.280, 123.675],
         std=[1.0, 1.0, 1.0],
         to_tensor=False),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False),
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="PointsToTensor"),
    dict(type="CustomCollect3D",
         keys=["img",
               "points",
               "timestamp",
               "l2g_r_mat",
               "l2g_t",
               "gt_lane_labels",
               "gt_lane_bboxes",
               "gt_lane_masks",
               "gt_segmentation",
               "gt_inds",
               "gt_bboxes_3d",
               "gt_labels_3d",
               "gt_fut_traj",
               "gt_fut_traj_mask",
               "gt_past_traj",
               "gt_past_traj_mask",
               "gt_sdc_bbox",
               "gt_sdc_label",
               "gt_sdc_fut_traj",
               "gt_sdc_fut_traj_mask",
               "gt_instance",
               "gt_centerness",
               "gt_offset",
               "gt_flow",
               "gt_backward_flow",
               "gt_occ_has_invalid_frame",
               "gt_occ_img_is_valid",
               "gt_future_boxes",
               "gt_future_labels",
               "sdc_planning",
               "sdc_planning_mask",
               "command"],
         meta_keys=(
             "filename", "ori_shape", "img_shape", "lidar2img", "depth2img",
             "cam2img", "pad_shape", "scale_factor", "flip",
             "pcd_horizontal_flip", "pcd_vertical_flip", "box_mode_3d",
             "box_type_3d", "img_norm_cfg", "pcd_trans", "sample_idx",
             "prev_idx", "next_idx", "pcd_scale_factor", "pcd_rotation",
             "pts_filename", "transformation_3d_flow", "scene_token", "can_bus",
             "camera2ego", "lidar2ego", "lidar2camera", "lidar2image",
             "camera_intrinsics", "camera2lidar", "img_aug_matrix", "lidar_aug_matrix")),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D_E2E',
         with_bbox_3d=False,
         with_label_3d=False,
         with_attr_label=False,
         with_future_anns=True,
         with_ins_inds_3d=False,
         ins_inds_add_1=True),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=False,
        is_train=False),
    dict(
        type="GlobalRotScaleTrans_3D",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    dict(type="ImageNormalize",
         mean=[103.530, 116.280, 123.675],
         std=[1.0, 1.0, 1.0]),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False),
    dict(type="PointsToTensor"),
    dict(
        type="CustomCollect3D",
        keys=["img", "points", "timestamp", "l2g_r_mat", "l2g_t",
              "gt_lane_labels", "gt_lane_bboxes", "gt_lane_masks", "gt_segmentation"],
        meta_keys=(
            "camera2ego", "lidar2ego", "lidar2camera", "lidar2image",
            "camera_intrinsics", "camera2lidar", "img_aug_matrix", "lidar_aug_matrix",
            "pts_filename", "scene_token", "can_bus", "box_type_3d", "box_mode_3d",
            "img_shape", "sample_idx")),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track', 'map'],
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=object_classes,
        modality=input_modality,
        box_type_3d="LiDAR",
        eval_mod=['map', 'track'],
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# Only Track_Map_Former parameters are optimized; UniBEV is frozen
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=4,
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 7
evaluation = dict(
    interval=8,
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ]
)
checkpoint_config = dict(interval=1)
load_from = None
resume_from = None
