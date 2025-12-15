import argparse
import os
import sys
import importlib
import torch

from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import DataContainer as DC
from llava.utils import pad_bevfeature

def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x


def unwrap_metas(img_metas):
    if isinstance(img_metas, DC):
        img_metas = img_metas.data
    while isinstance(img_metas, list):
        img_metas = img_metas[0]
    assert isinstance(img_metas, dict)
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
        return [to_device(xx) for xx in x]
    return x


def parse_args():
    parser = argparse.ArgumentParser("BEVFusion → Track_Map_Former test")
    parser.add_argument("--bevfusion-config", default='projects/configs/bevfusion_track_map/bevfusion.py')
    parser.add_argument("--bevfusion-ckpt", default='/home/s56cai/ckpt/bevfusion/bevfusion-det.pth')
    parser.add_argument("--track-config", default='projects/configs/bevfusion_track_map/track_map_former.py')
    parser.add_argument("--track-ckpt", default='/home/s56cai/ckpt/uniad_stage1/uniad_base_track_map.pth')
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg_bf = Config.fromfile(args.bevfusion_config)

    if cfg_bf.get("plugin", False):
        sys.path.insert(0, os.getcwd())
        sys.path.insert(0, os.path.join(os.getcwd(), cfg_bf.plugin_dir))
        importlib.import_module("projects.mmdet3d_plugin")

    from projects.mmdet3d_plugin.models.builder import build_model

    bevfusion = build_model(cfg_bf.model).to(device).eval()
    ckpt = torch.load(args.bevfusion_ckpt, map_location="cpu")
    bevfusion.load_state_dict(ckpt["state_dict"], strict=False)

    dataset = build_dataset(cfg_bf.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    data = next(iter(dataloader))

    # unwrap inputs
    img = unwrap_dc(data["img"])
    points = unwrap_dc(data["points"])
    meta0, metas = unwrap_metas(data["img_metas"])
    timestamp = unwrap_dc(data.get("timestamp", None))
    l2g_r_mat = unwrap_dc(data.get("l2g_r_mat", None))
    l2g_t     = unwrap_dc(data.get("l2g_t", None))

    gt_lane_labels = unwrap_dc(data.get("gt_lane_labels", None))
    gt_lane_bboxes = unwrap_dc(data.get("gt_lane_bboxes", None))
    gt_lane_masks  = unwrap_dc(data.get("gt_lane_masks", None))

    timestamp = to_device(timestamp, device)
    l2g_r_mat = to_device(l2g_r_mat, device)
    l2g_t     = to_device(l2g_t, device)
    gt_lane_labels = to_device(gt_lane_labels, device)
    gt_lane_bboxes = to_device(gt_lane_bboxes, device)
    gt_lane_masks  = to_device(gt_lane_masks, device)
    print("gt_lane_masks type:", type(data["gt_lane_masks"]))
    print("gt_lane_masks shape:", getattr(data["gt_lane_masks"], "shape", None))
    print("gt_segmentation shape:", data.get("gt_segmentation", None).shape if "gt_segmentation" in data else None)

    # points: Tensor [1, N, 5] -> list[Tensor(N,5)]
    if torch.is_tensor(points):
        points = [points[0].to(device)]
    else:
        points = [p.to(device) for p in points]

    # img: list[Tensor(C,H,W)] -> Tensor [1,6,C,H,W]
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
        bevfeature, camerafeature = bevfusion(
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

    print("[BEVFusion]")
    print("  bevfeature:", bevfeature.shape)
    print("  camerafeature:", camerafeature.shape)
    from mmdet3d.models import build_model as build_mmdet3d_model
    cfg_tm = Config.fromfile(args.track_config)
    cfg_tm.model.pop("train_cfg", None)
    cfg_tm.model.pop("test_cfg", None)
    model_tm = build_mmdet3d_model(cfg_tm.model).to(device).eval()

    ckpt_tm = torch.load(args.track_ckpt, map_location="cpu")
    model_tm.load_state_dict(ckpt_tm["state_dict"], strict=False)
    bevfeature = pad_bevfeature(bevfeature, target_size=(200, 200))
    if bevfeature.dim() == 4:
        B, C, H, W = bevfeature.shape
        bev_embed = bevfeature.flatten(2).permute(2, 0, 1).contiguous()
    else:
        bev_embed = bevfeature

    with torch.no_grad():
        outputs = model_tm.forward_test(
            bev_embed=bev_embed,
            img_feat_2D=camerafeature,
            img_metas=metas,

            # seg eval / IOU needs these
            gt_lane_labels=gt_lane_labels,
            gt_lane_masks=gt_lane_masks,
            gt_lane_bboxes=gt_lane_bboxes,
            rescale=False,

            # optional (usually unused in your feature-only Track_Map_Former)
            timestamp=timestamp,
            l2g_r_mat=l2g_r_mat,
            l2g_t=l2g_t,
        )


    print("\n[Track_Map_Former]")
    print("  keys:", outputs.keys())
    print("  track queries:",
          None if outputs["result_track"]["track_query_embeddings"] is None
          else outputs["result_track"]["track_query_embeddings"].shape)

    if outputs["result_seg"] is not None:
        print("  map queries:",
              outputs["result_seg"]["chosen_output_query_things"].shape)

    print("\n✅ SUCCESS: BEVFusion → Track_Map_Former pipeline runs.")


if __name__ == "__main__":
    main()
