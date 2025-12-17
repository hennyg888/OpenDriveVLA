from typing import Optional, Union
import torch
import torch.utils.checkpoint
from torch import nn
import os
import os.path as osp

from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from llava.utils import rank0_print, pad_bevfeature
from mmcv.parallel import DataContainer as DC
from mmengine import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.models import build_model as build_fusion_model

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('shapely.geos').setLevel(logging.ERROR)
from transformers import logging
logging.set_verbosity_error()

def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x

def to_tensor_list(x, device):
    out = []
    for a in x:
        if torch.is_tensor(a):
            out.append(a.to(device))
        else:
            out.append(torch.from_numpy(a).to(device))
    return out
def to_device_optional(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [to_device_optional(xx, device) for xx in x]
    return x

def unwrap_metas(img_metas):
    if isinstance(img_metas, DC):
        img_metas = img_metas.data
    while isinstance(img_metas, list):
        img_metas = img_metas[0]
    assert isinstance(img_metas, dict)
    return img_metas, [img_metas]

class BevFusionTrackMapConfig(PretrainedConfig):
    model_type = "bevfusion_track_map_model"

    def __init__(self, bevfusion_config_dict: Optional[dict] = None, track_map_former_config_dict: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)

        self.bevfusion_config_dict = bevfusion_config_dict
        self.track_map_former_config_dict = track_map_former_config_dict

class BEVFusionTrackMapModel(PreTrainedModel):
    config_class = BevFusionTrackMapConfig
    base_model_prefix = "bevfusion_track_map"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _no_split_modules = ["BEVFusion", "TrackMapFormer"]

    def __init__(self, config: BevFusionTrackMapConfig, load_mmdet3d_weights=True, vision_tower_test_mode=False):
        super().__init__(config)

        self.config = config
        self.load_mmdet3d_weights = load_mmdet3d_weights
        self.vision_tower_test_mode = vision_tower_test_mode
        # build the Bevfusion_Track_Map model
        self.bevfusion, self.track_map_former = self.build_bevfusion_track_map_model()

    def build_bevfusion_track_map_model(self):
        bevfusion_config_mmlab = Config()
        track_map_former_mmlab = Config()
        bevfusion_config_mmlab.merge_from_dict(self.config.bevfusion_config_dict)
        track_map_former_mmlab.merge_from_dict(self.config.track_map_former_config_dict)
        
        # import modules from plguin/xx, registry will be updated
        if hasattr(bevfusion_config_mmlab, 'plugin'):
            if bevfusion_config_mmlab.plugin:
                import importlib
                import sys
                # plugin_dir = bevfusion_config_mmlab.plugin_dir
                # _module_dir = osp.dirname(plugin_dir)
                # _module_dir = str(_module_dir).split('/')
                # _module_path = _module_dir[0]

                # for m in _module_dir[1:]:
                #     _module_path = _module_path + '.' + m
                # print(_module_path)
                # plg_lib = importlib.import_module(_module_path)
                if bevfusion_config_mmlab.get("plugin", False):
                    sys.path.insert(0, os.getcwd())
                    sys.path.insert(0, os.path.join(os.getcwd(), bevfusion_config_mmlab.plugin_dir))
                    importlib.import_module("projects.mmdet3d_plugin")

        device = torch.device("cuda", torch.cuda.current_device())
        print(f"Building BEVFusion and TrackMapFormer models...{device}")
        bevfusion_config_mmlab.model.pretrained = None
        bevfusion_config_mmlab.model.train_cfg = None
        bevfusion_model = build_fusion_model(bevfusion_config_mmlab.model)
        track_map_former_model = build_model(track_map_former_mmlab.model)
        bevfusion_model = bevfusion_model.to_empty(device=device)
        track_map_former_model = track_map_former_model.to_empty(device=device)
        if self.load_mmdet3d_weights:
            bevfusion_checkpoint = torch.load('/home/s56cai/ckpt/bevfusion/bevfusion-det.pth', map_location='cpu')
            track_map_former_checkpoint = torch.load('/home/s56cai/ckpt/uniad_stage1/uniad_base_track_map.pth', map_location='cpu')
            bevfusion_model.load_state_dict(bevfusion_checkpoint["state_dict"], strict=False)
            track_map_former_model.load_state_dict(track_map_former_checkpoint["state_dict"], strict=False)
            bevfusion_model = bevfusion_model.to(device)
            track_map_former_model = track_map_former_model.to(device)
            bevfusion_model.float()
            track_map_former_model.float()
            
            bevfusion_model.eval()
            track_map_former_model.eval()
            
            if 'CLASSES' in bevfusion_checkpoint.get('meta', {}):
                bevfusion_model.CLASSES = bevfusion_checkpoint['meta']['CLASSES']
            if 'PALETTE' in bevfusion_checkpoint.get('meta', {}):
                bevfusion_model.PALETTE = bevfusion_checkpoint['meta']['PALETTE']
            if 'CLASSES' in track_map_former_checkpoint.get('meta', {}):
                track_map_former_model.CLASSES = track_map_former_checkpoint['meta']['CLASSES']
            if 'PALETTE' in track_map_former_checkpoint.get('meta', {}):
                track_map_former_model.PALETTE = track_map_former_checkpoint['meta']['PALETTE']
                
        return bevfusion_model, track_map_former_model

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

    def forward(self, data):
        if self.vision_tower_test_mode:
            device = next(self.bevfusion.parameters()).device
            img = unwrap_dc(data["img"])
            if isinstance(img, list):
                img = torch.stack(img, dim=0).unsqueeze(0)
            img = img.to(device, non_blocking=True)

            points = unwrap_dc(data["points"])
            if torch.is_tensor(points):
                if points.dim() == 3:          # [1, N, C]
                    points = [points[0]]
                elif points.dim() == 2:        # [N, C]
                    points = [points]
                else:
                    raise ValueError(f"Unexpected points tensor shape: {points.shape}")
            elif isinstance(points, list):
                assert torch.is_tensor(points[0]), type(points[0])
            else:
                raise TypeError(f"Unexpected points type: {type(points)}")

            points = [p.to(device, non_blocking=True) for p in points]

            meta0, metas = unwrap_metas(data["img_metas"])

            camera2ego = [torch.as_tensor(x, device=device) for x in meta0["camera2ego"]]
            lidar2ego = torch.as_tensor(meta0["lidar2ego"], device=device)
            lidar2camera = [torch.as_tensor(x, device=device) for x in meta0["lidar2camera"]]
            lidar2image = [torch.as_tensor(x, device=device) for x in meta0["lidar2image"]]
            camera_intrinsics = [torch.as_tensor(x, device=device) for x in meta0["camera_intrinsics"]]
            camera2lidar = [torch.as_tensor(x, device=device) for x in meta0["camera2lidar"]]
            img_aug_matrix = [torch.as_tensor(x, device=device) for x in meta0["img_aug_matrix"]]
            lidar_aug_matrix = torch.as_tensor(meta0["lidar_aug_matrix"], device=device)
            
            timestamp = to_device_optional(unwrap_dc(data.get("timestamp", None)), device)
            l2g_r_mat = to_device_optional(unwrap_dc(data.get("l2g_r_mat", None)), device)
            l2g_t = to_device_optional(unwrap_dc(data.get("l2g_t", None)), device)
            gt_lane_labels = to_device_optional((data.get("gt_lane_labels", None)), device)
            gt_lane_bboxes = to_device_optional((data.get("gt_lane_bboxes", None)), device)
            gt_lane_masks = to_device_optional((data.get("gt_lane_masks", None)), device)
            if gt_lane_masks is not None:
                if gt_lane_masks.dim() == 3:
                    gt_lane_masks = gt_lane_masks.unsqueeze(0) 
            if gt_lane_bboxes is not None:
                if gt_lane_bboxes.dim() == 2:
                    gt_lane_bboxes = gt_lane_bboxes.unsqueeze(0)
            if gt_lane_labels is not None:
                if gt_lane_labels.dim() ==1:
                    gt_lane_labels = gt_lane_labels.unsqueeze(0)
            with torch.cuda.amp.autocast(enabled=False):
                bevfeature, camerafeature = self.bevfusion(
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
                    metas=metas,
                    depths=None,
                    radar=None,
                )

            bevfeature = pad_bevfeature(bevfeature, target_size=(200, 200))
            B, C, H, W = bevfeature.shape
            bev_embed = bevfeature.flatten(2).permute(2, 0, 1).contiguous()

            results_for_vlm = self.track_map_former.forward_test(
                bev_embed=bev_embed,
                img_feat_2D=camerafeature,
                img_metas=metas,
                gt_lane_labels=gt_lane_labels,
                gt_lane_bboxes=gt_lane_bboxes,
                gt_lane_masks=gt_lane_masks,
                timestamp=timestamp,
                l2g_r_mat=l2g_r_mat,
                l2g_t=l2g_t,
                rescale=False,
            )
        else:
            bevfeature = self.bevfusion(data)
            padded_bevfeature = pad_bevfeature(bevfeature, target_size=(200, 200))
            _, results_for_vlm = self.track_map_former(padded_bevfeature, return_loss=False, rescale=True)
        return results_for_vlm

class BEVFusionTrackMapVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()
        
        bevfusion_config_dict = Config.fromfile('/home/s56cai/OpenDriveVLA/projects/configs/bevfusion_track_map/bevfusion.py').to_dict()
        track_map_former_config_dict = Config.fromfile('/home/s56cai/OpenDriveVLA/projects/configs/bevfusion_track_map/track_map_former.py').to_dict()
        self.config = BevFusionTrackMapConfig(bevfusion_config_dict=bevfusion_config_dict, track_map_former_config_dict=track_map_former_config_dict)

        self.vision_tower_name = vision_tower
        self.vision_tower: nn.Module = None
        self.is_loaded = False
        self.vision_tower_pretrained = vision_tower_cfg.vision_tower_pretrained
        if hasattr(vision_tower_cfg, "vision_tower_test_mode"):
            self.vision_tower_test_mode = vision_tower_cfg.vision_tower_test_mode
        else:  # set to False when in training mode
            self.vision_tower_test_mode = False

        self.image_processor = None

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()

        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()

        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

        else:
            self.cfg_only = self.config

    def load_model(self, device_map="auto"):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        if self.vision_tower_pretrained:
            # Check if vision_tower_name points to a valid pretrained model path
            load_from_transformers_pretrained = (
                isinstance(self.vision_tower_pretrained, str) and 
                (os.path.exists(self.vision_tower_pretrained) or 
                self.vision_tower_pretrained.startswith('https://') or
                self.vision_tower_pretrained.startswith('http://'))
            )

            if load_from_transformers_pretrained:
                rank0_print(f"Loading BEVFusion and TrackMapFormer transformers checkpoint from: {self.vision_tower_pretrained}")
                self.vision_tower = BEVFusionTrackMapModel.from_pretrained(
                    self.vision_tower_pretrained, 
                    device_map=device_map
                )
                self.vision_tower.vision_tower_test_mode = self.vision_tower_test_mode
            else: # load from mmdet3d checkpoint
                rank0_print("Loading BEVFusion and TrackMapFormer from mmdet3d checkpoint")
                self.vision_tower = BEVFusionTrackMapModel(self.config, load_mmdet3d_weights=True, vision_tower_test_mode=self.vision_tower_test_mode)
        else:
            # only build the model, not load the weights
            rank0_print("Building BEVFusion and TrackMapFormer from config. Weights will be loaded from the llava checkpoint.")
            self.vision_tower = BEVFusionTrackMapModel(self.config, load_mmdet3d_weights=False, vision_tower_test_mode=self.vision_tower_test_mode)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, data):
        return self.vision_tower(data)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device