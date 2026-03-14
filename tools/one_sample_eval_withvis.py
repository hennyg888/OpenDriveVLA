import os
import sys
import importlib
from typing import Any, Dict, List

import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import build_dataset
from analysis_tools.visualize.utils import AgentPredictionData
from analysis_tools.visualize.render.bev_render import BEVRender
from analysis_tools.visualize.render.cam_render import CameraRender


def _add_repo_to_syspath():
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())
    return repo_root


_add_repo_to_syspath()

# copy constants from pretrain_eval for convenience
UNIAD_BEVFUSION_CONFIG = "/home/s56cai/OpenDriveVLA/projects/configs/uniad_bevfusion/pretrain_decoder.py"
UNIAD_BEVFUSION_CKPT = "/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain_decoder/epoch_6.pth"
EVAL_MOD = ["track", "map"]

# directory to save detection / GT visualization results
OUTPUT_DIR = "/home/s56cai/OpenDriveVLA/detection_results/decoder"


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


def find_sample_token_from_scene_and_time(dataset, scene_token, timestamp_sec):
    """Resolve NuScenes sample_token given scene_token and timestamp in seconds."""
    if scene_token is None or timestamp_sec is None:
        return None
    target_ts = timestamp_sec * 1e6
    for info in dataset.data_infos:
        if info.get("scene_token", None) != scene_token:
            continue
        ts = info.get("timestamp", None)
        if ts is None:
            continue
        if abs(ts - target_ts) < 1e-1:
            return info.get("token", None)
    return None


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


def build_agent_predictions_from_track(track_out: Dict[str, Any],
                                       score_thresh: float = 0.0) -> List[AgentPredictionData]:
    """Convert model track outputs to AgentPredictionData list for visualization."""
    # Prefer using track_bbox_results (the same source used for debug num_boxes)
    bboxes = None
    scores = None
    labels = None
    track_ids = None

    tb_list = track_out.get("track_bbox_results", None)
    if isinstance(tb_list, list) and len(tb_list) > 0 and isinstance(tb_list[0], list) and len(tb_list[0]) >= 3:
        tb = tb_list[0]
        bboxes = tb[0]          # LiDARInstance3DBoxes
        scores = tb[1]          # Tensor of scores
        labels = tb[2]          # Tensor of class ids
        if len(tb) >= 4:
            track_ids = tb[3]   # Tensor of track ids
    else:
        # Fallback to direct boxes_3d / scores_3d / labels_3d if available
        bboxes = track_out.get("boxes_3d", None)
        scores = track_out.get("scores_3d", None)
        labels = track_out.get("labels_3d", None)
        track_ids = track_out.get("track_ids", None)

    if bboxes is None or scores is None or labels is None:
        return []

    track_scores = scores.detach().cpu().numpy()
    track_labels = labels.detach().cpu().numpy()
    track_centers = bboxes.gravity_center.detach().cpu().numpy()
    track_dims = bboxes.dims.detach().cpu().numpy()
    track_yaw = bboxes.yaw.detach().cpu().numpy()
    track_velocity = bboxes.tensor.detach().cpu().numpy()[:, -2:]

    if track_ids is not None and torch.is_tensor(track_ids):
        track_ids = track_ids.detach().cpu().numpy()

    predicted_agent_list: List[AgentPredictionData] = []
    for i in range(track_scores.shape[0]):
        if track_scores[i] < score_thresh:
            continue

        if track_ids is not None:
            track_id = track_ids[i] if i < len(track_ids) else 0
        else:
            track_id = None

        predicted_agent_list.append(
            AgentPredictionData(
                pred_score=track_scores[i],
                pred_label=track_labels[i],
                pred_center=track_centers[i],
                pred_dim=track_dims[i],
                pred_yaw=track_yaw[i],
                pred_vel=track_velocity[i],
                pred_traj=None,
                pred_traj_score=0,
                pred_track_id=track_id,
                pred_occ_map=None,
                past_pred_traj=None,
            )
        )

    return predicted_agent_list

def get_classes(ds):
    if hasattr(ds, "CLASSES"):
        return ds.CLASSES
    if hasattr(ds, "dataset"):
        return ds.dataset.CLASSES
    if hasattr(ds, "datasets"):
        return ds.datasets[0].CLASSES
    return None


def find_sample_token_from_scene_and_time(dataset, scene_token, timestamp_sec):
    """Resolve NuScenes sample_token given scene_token and timestamp in seconds."""
    if scene_token is None or timestamp_sec is None:
        return None
    target_ts = timestamp_sec * 1e6
    for info in dataset.data_infos:
        if info.get("scene_token", None) != scene_token:
            continue
        ts = info.get("timestamp", None)
        if ts is None:
            continue
        if abs(ts - target_ts) < 1e-1:
            return info.get("token", None)
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

    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # prepare renderers (prediction mode by default)
    bev_render = BEVRender(show_gt_boxes=False)
    cam_render = CameraRender(show_gt_boxes=False)

    PRINT_GT = False

    # GT 可视化模式：只画 GT box 到 jpg，不跑模型
    if PRINT_GT:
        from nuscenes.prediction import PredictHelper

        print("data classes: ", get_classes(dataset))
        print("data_infos keys: ", dataset.data_infos[0].keys())

        # 选一个场景（和之前打印 GT 的场景一致，按需可修改）
        scene_token = "c3ab8ee2c1a54068a72d7eb4cf22e43d"
        scene_indices = [
            i for i, info in enumerate(dataset.data_infos)
            if info["scene_token"] == scene_token
        ]

        predict_helper = PredictHelper(dataset.nusc)
        max_samples = 10

        for frame_idx, idx in enumerate(scene_indices):
            if frame_idx >= max_samples:
                break
            sample_token = dataset.data_infos[idx]["token"]
            print(f"[GT_VIS] scene={scene_token}, frame_idx={frame_idx}, sample_token={sample_token}")

            out_prefix = os.path.join(
                OUTPUT_DIR, f"frame_{frame_idx:04d}_gt_{sample_token}"
            )

            # BEV: lidar + GT 3D boxes
            bev_render.reset_canvas(dx=1, dy=1)
            bev_render.set_plot_cfg()
            bev_render.show_lidar_data(sample_token, dataset.nusc)
            bev_render.render_anno_data(sample_token, dataset.nusc, predict_helper)
            bev_render.save_fig(out_prefix + "_bev.jpg")

            # Camera: 多相机图像 + GT 3D boxes
            cam_render.show_gt_boxes = True
            cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
            cam_render.render_image_data(sample_token, dataset.nusc)
            cam_render.render_pred_track_bbox([], sample_token, dataset.nusc)
            cam_render.save_fig(out_prefix + "_cam.jpg")

        return

    # 预测 + 可视化（原来的 eval 模式）
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
            # keep a cpu copy of timestamp (seconds) for resolving sample_token later
            raw_timestamp = unwrap_dc(data.get("timestamp", None))
            timestamp_cpu = float(raw_timestamp) if raw_timestamp is not None else None
            timestamp = to_device(raw_timestamp, device)
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

            # debug: inspect structure of out / result_track once
            if sample_count == 0:
                print("[DEBUG_VIS] out type:", type(out))
                if isinstance(out, dict):
                    print("[DEBUG_VIS] out keys:", list(out.keys()))
                    rt_dbg = out.get("result_track", None)
                    print("[DEBUG_VIS] result_track type:", type(rt_dbg))
                    if isinstance(rt_dbg, list):
                        print("[DEBUG_VIS] len(result_track):", len(rt_dbg))
                        if len(rt_dbg) > 0:
                            print("[DEBUG_VIS] type(result_track[0]):", type(rt_dbg[0]))
                            if isinstance(rt_dbg[0], dict):
                                print("[DEBUG_VIS] result_track[0] keys:", list(rt_dbg[0].keys()))
                else:
                    print("[DEBUG_VIS] out is not dict, repr:", repr(out))

            packed = pack_for_nuscenes_e2e(out)
            #print(packed)
            results.append(packed)
            gt_examples.append({
                "gt_lane_labels": gt_lane_labels,
                "gt_lane_bboxes": gt_lane_bboxes,
                "gt_lane_masks": gt_lane_masks,
                "img_metas": meta0,
            })

            # visualize detection for each sample that has valid track outputs (within this scene / max_samples)
            track_out = out.get("result_track", {}) or {}
            if isinstance(track_out, list):
                track_out = track_out[0] if len(track_out) > 0 and isinstance(track_out[0], dict) else {}

            if isinstance(track_out, dict) and track_out:
                agent_list = build_agent_predictions_from_track(track_out)
                # prefer token attached by the model (from result_track)
                sample_token = track_out.get("token", None)
                # fallback: try to recover sample_token for NuScenes lookup from metadata
                if sample_token is None:
                    meta_token = meta0.get("token", None)
                    sample_idx = meta0.get("sample_idx", None)
                    scene_token_meta = meta0.get("scene_token", None)
                    if meta_token is not None:
                        sample_token = meta_token
                    elif sample_idx is not None:
                        # in this codebase sample_idx is actually the NuScenes sample token
                        sample_token = sample_idx
                    else:
                        # fall back: resolve token from scene_token + timestamp
                        sample_token = find_sample_token_from_scene_and_time(
                            dataset, scene_token_meta, timestamp_cpu
                        )

                print(
                    "[DEBUG_VIS2] "
                    f"meta_token={meta0.get('token', None)}, "
                    f"meta_sample_idx={meta0.get('sample_idx', None)}, "
                    f"meta_scene_token={meta0.get('scene_token', None)}, "
                    f"track_token={track_out.get('token', None)}, "
                    f"resolved_sample_token={sample_token}, "
                    f"len(agent_list)={len(agent_list)}, "
                    f"timestamp_cpu={timestamp_cpu}"
                )

                if sample_token is not None:
                    print(f"[VIS] visualizing detection for sample_token={sample_token}, num_agents={len(agent_list)}")
                    out_prefix = os.path.join(
                        OUTPUT_DIR, f"frame_{sample_count:04d}_{sample_token}"
                    )

                    # BEV view
                    bev_render.reset_canvas(dx=1, dy=1)
                    bev_render.set_plot_cfg()
                    bev_render.show_lidar_data(sample_token, dataset.nusc)
                    bev_render.render_pred_box_data(agent_list)
                    bev_render.save_fig(out_prefix + "_bev.jpg")

                    # camera view
                    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
                    cam_render.render_image_data(sample_token, dataset.nusc)
                    cam_render.render_pred_track_bbox(agent_list, sample_token, dataset.nusc)
                    cam_render.save_fig(out_prefix + "_cam.jpg")

            sample_count += 1

        if len(results) == 0:
            print("no samples found for scene", scene_token)
            return


if __name__ == "__main__":
    main()