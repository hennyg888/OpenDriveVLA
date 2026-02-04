_base_ = ["../bevfusion_track_map/bevfusion.py"]

# BEVFormer track/map head settings (from bevfusion_track_map/track_map_former.py)
point_cloud_range_track = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size_track = [0.2, 0.2, 8]
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 5
past_steps = 4
fut_steps = 4

model = dict(
    type="UniADBevFusion",
    freeze_bevfusion=True,
    freeze_bevfusion_bn=False,
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
                        checkpoint="/home/s56cai/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth",
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
                    point_cloud_range=point_cloud_range,
                    voxel_size=voxel_size,
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
        pc_range=point_cloud_range_track,
        img_backbone=dict(
            type="ResNet",
            depth=101,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=4,
            norm_cfg=dict(type="BN2d", requires_grad=False),
            norm_eval=True,
            style="caffe",
            dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
        img_neck=dict(
            type="FPN",
            in_channels=[512, 1024, 2048],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=4,
            relu_before_extra_convs=True,
        ),
        freeze_img_backbone=True,
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
                pc_range=point_cloud_range_track,
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
                    type="BEVFormerEncoder",
                    num_layers=6,
                    pc_range=point_cloud_range_track,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1),
                            dict(
                                type="SpatialCrossAttention",
                                pc_range=point_cloud_range_track,
                                deformable_attention=dict(
                                    type="MSDeformableAttention3D",
                                    embed_dims=_dim_,
                                    num_points=8,
                                    num_levels=_num_levels_,
                                ),
                                embed_dims=_dim_,
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
                pc_range=point_cloud_range_track,
                max_num=300,
                voxel_size=voxel_size_track,
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
            pc_range=point_cloud_range_track,
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
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1)
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"
find_unused_parameters = True
