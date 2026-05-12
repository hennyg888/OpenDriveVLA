from typing import Optional
import torch
import torch.utils.checkpoint
from torch import nn
import os
import os.path as osp

from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from llava.utils import rank0_print

from mmengine import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


class FusionADTrackMapConfig(PretrainedConfig):
    model_type = "fusionad_track_map_model"

    def __init__(self, fusionad_config_dict: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.fusionad_config_dict = fusionad_config_dict


class FusionADTrackMapModel(PreTrainedModel):
    config_class = FusionADTrackMapConfig
    base_model_prefix = "fusionad_track_map"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _no_split_modules = ["FusionAD"]

    def __init__(self, config: FusionADTrackMapConfig, load_mmdet3d_weights=False,
                 checkpoint_path=None, vision_tower_test_mode=False):
        super().__init__(config)

        self.config = config
        self.load_mmdet3d_weights = load_mmdet3d_weights
        self.checkpoint_path = checkpoint_path
        self.vision_tower_test_mode = vision_tower_test_mode
        self.vision_model = self._build_fusionad_model()

    def _build_fusionad_model(self):
        fusionad_cfg = Config()
        fusionad_cfg.merge_from_dict(self.config.fusionad_config_dict)

        # Register plugin modules (fusionad_plugin) so mmdet3d can find them.
        # mmdet3d_plugin may have already registered some of the same class names
        # (e.g. HungarianAssigner3D).  Temporarily patch Registry._register_module
        # to use force=True so that fusionad_plugin's classes silently overwrite
        # any previously registered duplicates instead of raising a KeyError.
        if hasattr(fusionad_cfg, 'plugin') and fusionad_cfg.plugin:
            import importlib
            import sys as _sys
            from mmcv.utils.registry import Registry as _Registry
            plugin_dir = fusionad_cfg.plugin_dir
            _module_dir = osp.dirname(plugin_dir)
            _module_path = '.'.join(str(_module_dir).split('/'))
            if _module_path not in _sys.modules:
                _orig_register = _Registry._register_module
                _Registry._register_module = (
                    lambda self, module, module_name=None, force=False:
                    _orig_register(self, module, module_name=module_name, force=True)
                )
                try:
                    importlib.import_module(_module_path)
                finally:
                    _Registry._register_module = _orig_register

        fusionad_cfg.model.pretrained = None
        fusionad_cfg.model.train_cfg = None
        model = build_model(fusionad_cfg.model, test_cfg=fusionad_cfg.get('test_cfg'))

        if self.load_mmdet3d_weights:
            ckpt_path = self.checkpoint_path or 'checkpoints/fusionad_base_track_map.pth'
            # strict=False: the checkpoint may contain plan/occ head weights that are
            # absent from this model (only track + map heads are used).
            checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint.get('meta', {}):
                model.PALETTE = checkpoint['meta']['PALETTE']

        return model

    def _init_weights(self, module):
        pass

    def forward(self, data):
        return self.vision_model.get_results_for_vlm(data)

    def get_results_for_eval(self, data):
        return self.vision_model.get_results_for_eval(data)


class FusionADVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        fusionad_config_dict = Config.fromfile(
            'projects/configs/fusionad/fusion_base_track_map.py'
        ).to_dict()
        self.config = FusionADTrackMapConfig(fusionad_config_dict=fusionad_config_dict)

        self.vision_tower_name = vision_tower
        self.vision_tower: nn.Module = None
        self.is_loaded = False
        self.vision_tower_pretrained = vision_tower_cfg.vision_tower_pretrained
        if hasattr(vision_tower_cfg, 'vision_tower_test_mode'):
            self.vision_tower_test_mode = vision_tower_cfg.vision_tower_test_mode
        else:
            self.vision_tower_test_mode = False

        self.image_processor = None

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, 'unfreeze_mm_vision_tower', False):
            rank0_print("The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, 'mm_tunable_parts') and 'mm_vision_tower' in vision_tower_cfg.mm_tunable_parts:
            rank0_print("The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map='auto'):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, skipping.")
            return

        if self.vision_tower_pretrained:
            # A raw .pth file is an mmdet3d checkpoint, not a transformers directory.
            _p = self.vision_tower_pretrained
            load_from_transformers = (
                isinstance(_p, str) and
                not _p.endswith('.pth') and
                (os.path.exists(_p) or
                 _p.startswith('https://') or
                 _p.startswith('http://'))
            )
            if load_from_transformers:
                rank0_print(f"Loading FusionAD transformers checkpoint from: {self.vision_tower_pretrained}")
                self.vision_tower = FusionADTrackMapModel.from_pretrained(
                    self.vision_tower_pretrained,
                    device_map=device_map,
                )
                self.vision_tower.vision_tower_test_mode = self.vision_tower_test_mode
            else:
                # If vision_tower_pretrained ends with .pth, treat it as a raw
                # mmdet3d checkpoint file; otherwise fall back to the default path.
                raw_pth = (
                    self.vision_tower_pretrained
                    if (isinstance(self.vision_tower_pretrained, str) and
                        self.vision_tower_pretrained.endswith('.pth'))
                    else None
                )
                rank0_print(f"Loading FusionAD from mmdet3d checkpoint: {raw_pth or '(default)'}")
                self.vision_tower = FusionADTrackMapModel(
                    self.config,
                    load_mmdet3d_weights=True,
                    checkpoint_path=raw_pth,
                    vision_tower_test_mode=self.vision_tower_test_mode,
                )
        else:
            rank0_print("Building FusionAD from config. Weights will be loaded from the llava checkpoint.")
            self.vision_tower = FusionADTrackMapModel(
                self.config,
                load_mmdet3d_weights=False,
                vision_tower_test_mode=self.vision_tower_test_mode,
            )

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, data):
        return self.vision_tower(data)

    def get_results_for_eval(self, data):
        return self.vision_tower.get_results_for_eval(data)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device
