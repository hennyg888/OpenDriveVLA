_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

seed = 0
deterministic = False
total_epochs = 20
max_epochs = 6

dataset_type = "NuScenesE2EDataset"
data_root = "data/nuscenes/"

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
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

model = dict(
    type='BEVFusion',
    encoders=dict(
        camera=dict(
            backbone=dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
            ),

            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(type='bilinear', align_corners=False)
            ),
            
            vtransform=dict(
                type='DepthLSSTransform',
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-54.0, 54.0, 0.3], # H: 54 - (-54) / 0.3 = 360
                ybound=[-54.0, 54.0, 0.3], # W: 54 - (-54) / 0.3 = 360
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5], 
                downsample=2 # 180 * 180 * 
            )
        ),
        
        lidar=dict(
            voxelize = dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(120000, 160000)
            ),
            backbone = dict(
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
        )
    ),
    
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256 # B * 256 * 180 * 180
    ),
    
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)
        ),
        
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True
        )
    ),
    
    heads = dict(
        object=dict(
            type="TransFusionHead",
            num_proposals=200,
            auxiliary=True,
            in_channels=512,
            hidden_channel=128,
            num_classes=10,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",

            train_cfg=dict(
                dataset="nuScenes",
                point_cloud_range=point_cloud_range,
                grid_size=[1440, 1440, 41],
                voxel_size=voxel_size,
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                assigner=dict(
                    type="BEVHungarianAssigner3D",
                    iou_calculator=dict(
                        type="BboxOverlaps3D",
                        coordinate="lidar",
                    ),
                    cls_cost=dict(
                        type="FocalLossCost",
                        gamma=2.0,
                        alpha=0.25,
                        weight=0.15,
                    ),
                    reg_cost=dict(
                        type="BBoxBEVL1Cost",
                        weight=0.25,
                    ),
                    iou_cost=dict(
                        type="IoU3DCost",
                        weight=0.25,
                    ))),

            test_cfg=dict(
                dataset="nuScenes",
                grid_size=[1440, 1440, 41],
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2],
                nms_type=None),

            common_heads=dict(
                center=[2, 2],
                height=[1, 2],
                dim=[3, 2],
                rot=[2, 2],
                vel=[2, 2]),

            bbox_coder=dict(
                type="TransFusionBBoxCoder",
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=10),

            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction="mean",
                loss_weight=1.0),

            loss_heatmap=dict(
                type="GaussianFocalLoss",
                reduction="mean",
                loss_weight=1.0),

            loss_bbox=dict(
                type="L1Loss",
                reduction="mean",
                loss_weight=0.25),
        )
    )
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type="ObjectPaste",
        stop_epoch=gt_paste_stop_epoch,
        db_sampler=dict(
            dataset_root=data_root,
            info_path=data_root + "nuscenes_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5,
                ),
            ),
            classes=object_classes,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2,
            ),
            points_loader=dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=load_dim,
                use_dim=use_dim,
            ),
        )),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d['resize'][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=True,
        is_train=True),
    dict(
        type="GlobalRotScaleTrans_3D",
        resize_lim=augment3d['scale'],
        rot_lim=augment3d['rotate'],
        trans_lim=augment3d['translate'],
        is_train=True),
    dict(
      type="LoadBEVSegmentation",
      dataset_root=data_root,
      xbound=[-50.0, 50.0, 0.5],
      ybound=[-50.0, 50.0, 0.5],
      classes=map_classes),
    dict(type="BEVRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=object_classes),
    dict(
        type="ImageNormalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type="GridMask",
        use_h=True,
        use_w=True,
        max_epoch=total_epochs,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=augment2d['gridmask']['prob']),
    dict(type="PointShuffle"),
    dict(
        type="DefaultFormatBundle3D",
        class_names=object_classes),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "camera2lidar",
            "lidar2image",
            "img_aug_matrix",
            "lidar_aug_matrix")),
    dict(type="GTDepth", keyframe_only=True)
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
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
      type="LoadBEVSegmentation",
      dataset_root=data_root,
      xbound=[-50.0, 50.0, 0.5],
      ybound=[-50.0, 50.0, 0.5],
      classes=map_classes),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="ImageNormalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type="DefaultFormatBundle3D",
        class_names=object_classes),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "camera2lidar",
            "lidar2image",
            "img_aug_matrix",
            "lidar_aug_matrix")),
    dict(type="GTDepth", keyframe_only=True)
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            dataset_root=data_root,
            ann_file=data_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            object_classes=object_classes,
            map_classes=map_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d="LiDAR",
        ),
    ),

    val=dict(
        type=dataset_type,
        dataset_root=data_root,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),

    test=dict(
        type=dataset_type,
        dataset_root=data_root,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1e-3,
)
momentum_config = dict(policy="cyclic")

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
)

runner = dict(
    type="CustomEpochBasedRunner",
    max_epochs=max_epochs,
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook")
    ]
)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)

fp16 = dict(
    loss_scale=dict(growth_interval=2000)
)

cudnn_benchmark = False

load_from = None
resume_from = None

find_unused_parameters = True