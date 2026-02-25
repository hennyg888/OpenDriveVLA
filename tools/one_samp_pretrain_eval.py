import os
import sys
import importlib
from typing import Any, Dict, List

import torch
import mmcv
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import build_dataset


def _add_repo_to_syspath():
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())
    return repo_root


_add_repo_to_syspath()

# copy constants from pretrain_eval for convenience
UNIAD_BEVFUSION_CONFIG = "/home/hhguo/OpenDriveVLA/projects/configs/uniad_bevfusion/pretrain.py"
UNIAD_BEVFUSION_CKPT = "/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain/epoch_5.pth"
EVAL_MOD = ["track", "map"]


def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x


def unwrap_metas(img_metas):
    if isinstance(img_metas, DC):
        img_metas = img_metas.data
    while isinstance(img_metas, list):
        img_metas = img_metas[0]
    return img_metas, [img_metas]


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


def strip_unwanted(d):
    if not isinstance(d, dict):
        return d
    d.pop("occ", None)
    d.pop("motion", None)
    d.pop("planning", None)
    d.pop("result_occ", None)
    d.pop("result_motion", None)
    d.pop("result_planning", None)
    for k, v in list(d.items()):
        if isinstance(v, dict):
            strip_unwanted(v)
    return d


def to_cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_cpu_tree(v) for v in x)
    return x


def pack_for_nuscenes_e2e(out: Dict[str, Any]) -> Dict[str, Any]:
    out = strip_unwanted(out)
    out = to_cpu_tree(out)

    rt = out.get("result_track", {}) or {}
    if isinstance(rt, list):
        rt = rt[0] if len(rt) > 0 and isinstance(rt[0], dict) else {}
    elif not isinstance(rt, dict):
        rt = {}
    rs = out.get("result_seg", {}) or {}

    packed = {}
    packed.update(rt)

    if "ret_iou" in rs:
        packed["ret_iou"] = rs["ret_iou"]

    return packed

def get_classes(ds):
    if hasattr(ds, "CLASSES"):
        return ds.CLASSES
    if hasattr(ds, "dataset"):
        return ds.dataset.CLASSES
    if hasattr(ds, "datasets"):
        return ds.datasets[0].CLASSES
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_bf = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    if cfg_bf.get("plugin", False):
        sys.path.insert(0, os.path.join(os.getcwd(), cfg_bf.plugin_dir))
        importlib.import_module("projects.mmdet3d_plugin")

    cfg_bf.data.test.eval_mod = EVAL_MOD

    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet3d.models import build_model as build_mmdet3d_model

    print("building model...")
    cfg_tm = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    cfg_tm.model.pop("train_cfg", None)
    cfg_tm.model.pop("test_cfg", None)
    model_tm = build_mmdet3d_model(cfg_tm.model).to(device).eval()
    ckpt_tm = torch.load(UNIAD_BEVFUSION_CKPT, map_location="cpu")
    model_tm.load_state_dict(ckpt_tm.get("state_dict", ckpt_tm), strict=False)
    print("model ready")

    dataset = build_dataset(cfg_bf.data.test)

    PRINT_GT = False

    #GT PRINTING
    if PRINT_GT:
#       ata classes:  ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
#       data_infos keys:  dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'gt_inds', 'gt_ins_tokens', 'fut_traj', 'fut_traj_valid_mask', 'visibility_tokens'])
#       annotation keys:  dict_keys(['gt_bboxes_3d', 'gt_labels_3d', 'gt_names', 'gt_inds', 'gt_fut_traj', 'gt_fut_traj_mask', 'gt_past_traj', 'gt_past_traj_mask', 'gt_sdc_bbox', 'gt_sdc_label', 'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask', 'sdc_planning', 'sdc_planning_mask', 'command'])
        print("data classes: ",get_classes(dataset))

        scene_token = "c3ab8ee2c1a54068a72d7eb4cf22e43d"

        print("data_infos keys: ", dataset.data_infos[0].keys())

        scene_indices = [
            i for i, info in enumerate(dataset.data_infos)
            if info["scene_token"] == scene_token
        ]

        
        for idx in scene_indices:
            print("sample token:", dataset.data_infos[idx]["token"])
            print("timestamp:", dataset.data_infos[idx]["timestamp"])

            ann = dataset.get_ann_info(idx)
            #print("annotation keys: ", ann.keys())

            print(f"{len(ann['gt_names'])} Objects in frame:")
            for name, label, box in zip(
                ann["gt_names"],
                ann["gt_labels_3d"],
                ann["gt_bboxes_3d"]
            ):
                print(name, label, box)

    #PRINT INFERENCE RESULT
    if not PRINT_GT:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=getattr(cfg_bf.data, "workers_per_gpu", 0),
            dist=False,
            shuffle=False,
            nonshuffler_sampler=getattr(cfg_bf.data, "nonshuffler_sampler", None),
        )

        # iterate through dataloader and collect everything from the first scene
        scene_token = None
        results: List[Any] = []
        gt_examples: List[Dict[str, Any]] = []
        # keep track of raw tracking bboxes from data for printout
        gt_track_boxes: List[Any] = []
        sample_count = 0
        max_samples = 10

        for data in data_loader:
            #data.keys() dict_keys(['img_metas', 'img', 'points', 'timestamp', 'l2g_r_mat', 'l2g_t', 'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks', 'gt_segmentation'])
            print("timestamp: ", data["timestamp"].item())
            img = unwrap_dc(data["img"])
            points = unwrap_dc(data["points"])
            meta0, metas = unwrap_metas(data["img_metas"])
            current_scene = meta0.get("scene_token")
            if scene_token is None:
                scene_token = current_scene
                print(f"evaluating scene {scene_token} ...")
            elif current_scene != scene_token:
                # finished the scene or moved to next scene
                break

            if sample_count >= max_samples:
                break

            # prepare everything for this sample
            timestamp = to_device(unwrap_dc(data.get("timestamp", None)), device)
            l2g_r_mat = to_device(unwrap_dc(data.get("l2g_r_mat", None)), device)
            l2g_t = to_device(unwrap_dc(data.get("l2g_t", None)), device)

            gt_lane_labels = to_device(unwrap_dc(data.get("gt_lane_labels", None)), device)
            gt_lane_bboxes = to_device(unwrap_dc(data.get("gt_lane_bboxes", None)), device)
            gt_lane_masks = to_device(unwrap_dc(data.get("gt_lane_masks", None)), device)

            # capture tracking boxes if present
            track_boxes = data.get("gt_bboxes_3d", None)
            if track_boxes is not None:
                # try to move to cpu for printing later
                try:
                    if isinstance(track_boxes, DC):
                        track_boxes = track_boxes.data
                    if torch.is_tensor(track_boxes):
                        track_boxes = track_boxes.cpu().numpy()
                except Exception:
                    pass
            gt_track_boxes.append(track_boxes)

            if torch.is_tensor(points):
                points = [points[0].to(device)]
            else:
                points = [p.to(device) for p in points]

            if isinstance(img, list):
                img = torch.stack(img, dim=0).unsqueeze(0).to(device)
            else:
                img = img.to(device)

            camera2ego = to_torch_list(meta0["camera2ego"], device)
            lidar2ego = torch.from_numpy(meta0["lidar2ego"]).to(device)
            lidar2camera = to_torch_list(meta0["lidar2camera"], device)
            camera2lidar = to_torch_list(meta0["camera2lidar"], device)
            lidar2image = to_torch_list(meta0["lidar2image"], device)
            camera_intrinsics = to_torch_list(meta0["camera_intrinsics"], device)
            img_aug_matrix = to_torch_list(meta0["img_aug_matrix"], device)
            lidar_aug_matrix = torch.from_numpy(meta0["lidar_aug_matrix"]).to(device)

            with torch.no_grad():
                out = model_tm(
                    return_loss=False,
                    img=img,
                    points=points,
                    timestamp=timestamp,
                    l2g_r_mat=l2g_r_mat,
                    l2g_t=l2g_t,
                    gt_lane_labels=gt_lane_labels,
                    gt_lane_bboxes=gt_lane_bboxes,
                    gt_lane_masks=gt_lane_masks,
                    camera2ego=camera2ego,
                    lidar2ego=lidar2ego,
                    lidar2camera=lidar2camera,
                    camera2lidar=camera2lidar,
                    lidar2image=lidar2image,
                    camera_intrinsics=camera_intrinsics,
                    img_aug_matrix=img_aug_matrix,
                    lidar_aug_matrix=lidar_aug_matrix,
                    img_metas=metas,
                )

            packed = pack_for_nuscenes_e2e(out)
            #print(packed)
            results.append(packed)
            gt_examples.append({
                "gt_lane_labels": gt_lane_labels,
                "gt_lane_bboxes": gt_lane_bboxes,
                "gt_lane_masks": gt_lane_masks,
                "img_metas": meta0,
            })

            sample_count += 1

        if len(results) == 0:
            print("no samples found for scene", scene_token)
            return


if __name__ == "__main__":
    main()
