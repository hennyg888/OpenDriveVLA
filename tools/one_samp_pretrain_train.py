import os
import sys
import importlib
from typing import Any, Dict, List

import torch
import mmcv
from mmcv import Config
from mmcv.parallel import DataContainer as DC


def _add_repo_to_syspath():
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())
    return repo_root


_add_repo_to_syspath()

# Configuration
UNIAD_BEVFUSION_CONFIG = "/home/s56cai/OpenDriveVLA/projects/configs/uniad_bevfusion/pretrain.py"
UNIAD_BEVFUSION_CKPT = "/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain/epoch_3.pth"


def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x


def unwrap_metas(img_metas):
    if isinstance(img_metas, DC):
        img_metas = img_metas.data
    return img_metas


def to_torch_list(x, device):
    out = []
    for a in x:
        if torch.is_tensor(a):
            out.append(a.to(device))
        else:
            out.append(torch.from_numpy(a).to(device))
    return out


def to_device(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(xx, device) for xx in x]
    return x


def print_losses(losses_dict, prefix=""):
    """Pretty print loss dictionary"""
    if not isinstance(losses_dict, dict):
        print(f"{prefix}Losses: {losses_dict}")
        return
    
    print(f"\n{prefix}=== Losses ===")
    for key, value in sorted(losses_dict.items()):
        if torch.is_tensor(value):
            print(f"{prefix}{key}: {value.item():.4f}")
        else:
            print(f"{prefix}{key}: {value}")
    print(f"{prefix}==============\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading config...")
    cfg = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    if cfg.get("plugin", False):
        sys.path.insert(0, os.path.join(os.getcwd(), cfg.plugin_dir))
        importlib.import_module("projects.mmdet3d_plugin")
    
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet3d.models import build_model as build_mmdet3d_model
    from mmdet3d.datasets import build_dataset
    
    print("Building model...")
    cfg_model = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    cfg_model.model.pop("train_cfg", None)
    cfg_model.model.pop("test_cfg", None)
    model = build_mmdet3d_model(cfg_model.model).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(UNIAD_BEVFUSION_CKPT):
        print(f"Loading checkpoint: {UNIAD_BEVFUSION_CKPT}")
        ckpt = torch.load(UNIAD_BEVFUSION_CKPT, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    else:
        print("No checkpoint found, using random initialization")
    
    # Set to train mode for forward_train
    model.train()
    print("Model ready (train mode)")
    
    # Build train dataset
    print("Building train dataset...")
    dataset = build_dataset(cfg.data.train)
    
    # Directly get one sample from dataset (bypass dataloader)
    print("Getting sample directly from dataset...")
    sample_idx = 10  # Use index >= queue_length to ensure we get 5 frames
    data = dataset[sample_idx]
    
    
    # Unwrap data from DataContainers
    img = unwrap_dc(data["img"])
    points = unwrap_dc(data["points"])
    img_metas = unwrap_dc(data["img_metas"])
    
    # Check img_metas format and wrap in batch dimension
    if isinstance(img_metas, dict):
        num_frames = len(img_metas)
        meta0 = img_metas[0] if 0 in img_metas else None
        print(f"Loaded {num_frames} frames from scene: {meta0.get('scene_token', 'unknown') if meta0 else 'N/A'}")
        # Wrap in list for batch dimension: {0: meta0, ...} -> [{0: meta0, ...}]
        img_metas = [img_metas]
    # Process points - should be list of queue_length tensors
    if isinstance(points, list):
        points = [p.to(device) if torch.is_tensor(p) else p for p in points]
    elif torch.is_tensor(points):
        points = [points.to(device)]
    print(f"Loaded {len(points)} frames of points")
    
    # Process img - should be tensor [queue_len, N_cam, C, H, W]
    if torch.is_tensor(img):
        img = img.unsqueeze(0).to(device)  # Add batch dimension: [1, queue_len, N_cam, C, H, W]
    print(f"img shape: {list(img.shape)}")
    
    # Prepare GT inputs - add batch dimension
    timestamp = to_device(unwrap_dc(data.get("timestamp", None)), device)
    l2g_r_mat = to_device(unwrap_dc(data.get("l2g_r_mat", None)), device)
    l2g_t = to_device(unwrap_dc(data.get("l2g_t", None)), device)
    
    if isinstance(timestamp, list):
        timestamp = [timestamp]
    if isinstance(l2g_r_mat, list):
        l2g_r_mat = [l2g_r_mat]
    if isinstance(l2g_t, list):
        l2g_t = [l2g_t]
    
    # Tracking GT - add batch dimension for all GT lists
    gt_bboxes_3d = to_device(unwrap_dc(data.get("gt_bboxes_3d", None)), device)
    gt_labels_3d = to_device(unwrap_dc(data.get("gt_labels_3d", None)), device)
    
    # Wrap GT lists in batch dimension: [frame0, frame1, ...] -> [[frame0, frame1, ...]]
    if isinstance(gt_bboxes_3d, list):
        gt_bboxes_3d = [gt_bboxes_3d]
    if isinstance(gt_labels_3d, list):
        gt_labels_3d = [gt_labels_3d]
    gt_past_traj = to_device(unwrap_dc(data.get("gt_past_traj", None)), device)
    gt_past_traj_mask = to_device(unwrap_dc(data.get("gt_past_traj_mask", None)), device)
    gt_inds = to_device(unwrap_dc(data.get("gt_inds", None)), device)
    gt_sdc_bbox = to_device(unwrap_dc(data.get("gt_sdc_bbox", None)), device)
    gt_sdc_label = to_device(unwrap_dc(data.get("gt_sdc_label", None)), device)
    
    # Add batch dimension to all GT lists
    if isinstance(gt_past_traj, list):
        gt_past_traj = [gt_past_traj]
    if isinstance(gt_past_traj_mask, list):
        gt_past_traj_mask = [gt_past_traj_mask]
    if isinstance(gt_inds, list):
        gt_inds = [gt_inds]
    if isinstance(gt_sdc_bbox, list):
        gt_sdc_bbox = [gt_sdc_bbox]
    if isinstance(gt_sdc_label, list):
        gt_sdc_label = [gt_sdc_label]
    
    # Segmentation GT
    gt_lane_labels = to_device(unwrap_dc(data.get("gt_lane_labels", None)), device)
    gt_lane_bboxes = to_device(unwrap_dc(data.get("gt_lane_bboxes", None)), device)
    gt_lane_masks = to_device(unwrap_dc(data.get("gt_lane_masks", None)), device)
    gt_segmentation = to_device(unwrap_dc(data.get("gt_segmentation", None)), device)
    
    # Add batch dimension to segmentation GT lists
    if isinstance(gt_lane_labels, list):
        gt_lane_labels = [gt_lane_labels]
    if isinstance(gt_lane_bboxes, list):
        gt_lane_bboxes = [gt_lane_bboxes]
    if isinstance(gt_lane_masks, list):
        gt_lane_masks = [gt_lane_masks]
    if isinstance(gt_segmentation, list):
        gt_segmentation = [gt_segmentation]
    
    print("\nRunning forward_train...")
    try:
        # Build kwargs - camera params will be extracted from img_metas by model
        kwargs = {
            'return_loss': True,
            'img': img,
            'points': points,
            'img_metas': img_metas,
        }
        
        # Add optional GT parameters if they exist
        if timestamp is not None:
            kwargs['timestamp'] = timestamp
        if l2g_r_mat is not None:
            kwargs['l2g_r_mat'] = l2g_r_mat
        if l2g_t is not None:
            kwargs['l2g_t'] = l2g_t
        if gt_bboxes_3d is not None:
            kwargs['gt_bboxes_3d'] = gt_bboxes_3d
        if gt_labels_3d is not None:
            kwargs['gt_labels_3d'] = gt_labels_3d
        if gt_past_traj is not None:
            kwargs['gt_past_traj'] = gt_past_traj
        if gt_past_traj_mask is not None:
            kwargs['gt_past_traj_mask'] = gt_past_traj_mask
        if gt_inds is not None:
            kwargs['gt_inds'] = gt_inds
        if gt_sdc_bbox is not None:
            kwargs['gt_sdc_bbox'] = gt_sdc_bbox
        if gt_sdc_label is not None:
            kwargs['gt_sdc_label'] = gt_sdc_label
        if gt_lane_labels is not None:
            kwargs['gt_lane_labels'] = gt_lane_labels
        if gt_lane_bboxes is not None:
            kwargs['gt_lane_bboxes'] = gt_lane_bboxes
        if gt_lane_masks is not None:
            kwargs['gt_lane_masks'] = gt_lane_masks
        if gt_segmentation is not None:
            kwargs['gt_segmentation'] = gt_segmentation
        
        losses = model(**kwargs)
        
        print_losses(losses)
        
        # Print summary
        total_loss = sum([v.item() if torch.is_tensor(v) else v 
                        for k, v in losses.items() if 'loss' in k.lower()])
        print(f"Total Loss: {total_loss:.4f}")
        
        # Check for NaN
        has_nan = any([torch.isnan(v).any() if torch.is_tensor(v) else False 
                      for v in losses.values()])
        if has_nan:
            print("\nWARNING: NaN detected in losses!")
        else:
            print("\nNo NaN losses detected")
        
    except Exception as e:
        print(f"\nError during forward_train:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nForward train completed successfully!")


if __name__ == "__main__":
    main()
