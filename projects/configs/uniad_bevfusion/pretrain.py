_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]
        
queue_length = 5
file_client_args = dict(backend="disk")

bevfusion_point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
bevfusion_voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
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
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

image_size = [256, 704]

augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(
        prob=0.0,
        fixed_prob=True,
    )
)

augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5,
)


object_classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

map_classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']

input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)

planning_evaluation_strategy = "uniad"
### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

## occflow setting	
occ_n_future = 4	
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])	

model = dict(
    type="UniADBevFusion",
    freeze_bevfusion=True,
    freeze_bevfusion_bn=True,
    bev_in_hw=180,
    bev_out_hw=200,
    bevfusion=dict(
        type="BEVFusion",
        encoders=dict(
            camera=dict(
                backbone=dict(
                    type="SwinTransformer",
                    embed_dims=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.2,
                    patch_norm=True,
                    out_indices=(1, 2, 3),
                    with_cp=False,
                    convert_weights=True,
                    init_cfg=dict(
                        type="Pretrained",
                        checkpoint="/home/hhguo/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth",
                    ),
                ),
                neck=dict(
                    type="GeneralizedLSSFPN",
                    in_channels=[192, 384, 768],
                    out_channels=256,
                    start_level=0,
                    num_outs=3,
                    norm_cfg=dict(type="BN2d", requires_grad=True),
                    act_cfg=dict(type="ReLU", inplace=True),
                    upsample_cfg=dict(type="bilinear", align_corners=False),
                ),
                vtransform=dict(
                    type="DepthLSSTransform",
                    in_channels=256,
                    out_channels=80,
                    image_size=image_size,
                    feature_size=[image_size[0] // 8, image_size[1] // 8],
                    xbound=[-54.0, 54.0, 0.3],
                    ybound=[-54.0, 54.0, 0.3],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 60.0, 0.5],
                    downsample=2,
                ),
            ),
            lidar=dict(
                voxelize=dict(
                    max_num_points=10,
                    point_cloud_range=bevfusion_point_cloud_range,
                    voxel_size=bevfusion_voxel_size,
                    max_voxels=(120000, 160000),
                ),
                backbone=dict(
                    type="SparseEncoder",
                    in_channels=5,
                    sparse_shape=[1440, 1440, 41],
                    output_channels=128,
                    order=["conv", "norm", "act"],
                    encoder_channels=[
                        [16, 16, 32],
                        [32, 32, 64],
                        [64, 64, 128],
                        [128, 128],
                    ],
                    encoder_paddings=[
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, [1, 1, 0]],
                        [0, 0],
                    ],
                    block_type="basicblock",
                ),
            ),
        ),
        fuser=dict(type="ConvFuser", in_channels=[80, 256], out_channels=256),
    ),
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
            type="BEVFormerTrackHead",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=10,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            past_steps=past_steps,
            fut_steps=fut_steps,
            transformer=dict(
                type="PerceptionTransformer",
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                encoder=dict(
                    type="BEVFusionFormerEncoder",
                    num_layers=6,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1),
                            dict(type="TemporalCrossAttention", embed_dims=_dim_, num_levels=_num_levels_),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "temporal_cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
                decoder=dict(
                    type="DetectionTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type="CustomMSDeformableAttention",
                                embed_dims=_dim_,
                                num_levels=1,
                            ),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        ),
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
            in_channels=_dim_,
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
                            num_levels=_num_levels_,
                        ),
                        feedforward_channels=_feed_dim_,
                        ffn_dropout=0.1,
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
                            dict(
                                type="MultiheadAttention",
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type="MultiScaleDeformableAttention",
                                embed_dims=_dim_,
                                num_levels=_num_levels_,
                            ),
                        ],
                        feedforward_channels=_feed_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            positional_encoding=dict(
                type="SinePositionalEncoding",
                num_feats=_dim_half_,
                normalize=True,
                offset=-0.5,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            loss_mask=dict(type="DiceLoss", loss_weight=2.0),
            thing_transformer_head=dict(type="SegMaskHead", d_model=_dim_, nhead=8, num_decoder_layers=4),
            stuff_transformer_head=dict(type="SegMaskHead", d_model=_dim_, nhead=8, num_decoder_layers=6, self_attn=True),
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
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=[0,1,2,3,4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D_E2E', 
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,

        with_future_anns=True,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds 
        ins_inds_add_1=True,    # ins_inds start from 1
    ),
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
    dict(
        type="ImageNormalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
              # Occ gt
              "gt_instance", 
              "gt_centerness", 
              "gt_offset", 
              "gt_flow",
              "gt_backward_flow",
              "gt_occ_has_invalid_frame",
              "gt_occ_img_is_valid",
              # gt future bbox for plan	
              "gt_future_boxes",	
              "gt_future_labels",	
              # planning	
              "sdc_planning",	
              "sdc_planning_mask",	
              "command"],
        meta_keys=(
            # default meta keys (keep existing ones)
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "can_bus",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "lidar2image",
            "camera_intrinsics",
            "camera2lidar",
            "img_aug_matrix",
            "lidar_aug_matrix")
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=[0,1,2,3,4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,
         with_label_3d=False, 
         with_attr_label=False,

         with_future_anns=True,
         with_ins_inds_3d=False,
         ins_inds_add_1=True, # ins_inds start from 1
    ),
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
    dict(
        type="ImageNormalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
    type='GenerateOccFlowLabels',
    grid_conf=occflow_grid_conf,
    ignore_index=255,
    only_vehicle=True,
    filter_invisible=False),
    dict(type="PointsToTensor"),
    dict(
        type="CustomCollect3D",
        keys=["img", 
              "points",
              "timestamp",
              "l2g_r_mat",
              "l2g_t",
              "gt_lane_labels",
              "gt_lane_bboxes",
              "gt_lane_masks",
              "gt_segmentation"],
        meta_keys=(
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "lidar2image",
            "camera_intrinsics",
            "camera2lidar",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "pts_filename",
            # "transformation_3d_flow", 
            "scene_token",
            # "can_bus",
            # "l2g_r_mat",    
        )),
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

        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
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

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 6
evaluation = dict(
    interval=6,
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10, 
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="uniad_bevfusion",
                name="pretrain_resume1",
                tags=["bevfusion_backbone", "track_map_former", "pretrain"],
            ),
            log_artifact=True,
        ),
    ]
)
checkpoint_config = dict(interval=1)
load_from = None
resume_from = "/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain/epoch_1.pth"
find_unused_parameters = True
