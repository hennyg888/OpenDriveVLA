import argparse
import os
import sys
import importlib
import torch

from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import DataContainer as DC


def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x

def unwrap_metas(img_metas):
    # unwrap DataContainer
    if isinstance(img_metas, DC):
        img_metas = img_metas.data

    # peel lists until we hit dict
    metas = img_metas
    while isinstance(metas, list):
        if len(metas) == 0:
            raise ValueError("Empty img_metas list")
        metas = metas[0]

    if not isinstance(metas, dict):
        raise TypeError(f"Unexpected meta type: {type(metas)}")

    meta0 = metas

    # rebuild metas as List[Dict] (batch dimension)
    return meta0, [meta0]

def to_torch_list(x, device):
    """list[np.ndarray or Tensor] -> list[Tensor] on device"""
    out = []
    for a in x:
        if torch.is_tensor(a):
            out.append(a.to(device))
        else:
            out.append(torch.from_numpy(a).to(device))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test BEVFusion encoder-only checkpoint loading"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-model", action="store_true")
    return parser.parse_args()


def load_partial_checkpoint(model, ckpt_path):
    print("\n[CKPT] Loading PARTIAL checkpoint (encoders + fuser only)")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    keep_prefix = ("encoders.", "fuser.")
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k.startswith(keep_prefix)
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state_dict, strict=False
    )

    print(f"[CKPT] Loaded keys: {len(filtered_state_dict)}")
    print(f"[CKPT] Missing keys: {len(missing_keys)}")
    print(f"[CKPT] Unexpected keys: {len(unexpected_keys)}")

    return ckpt

def main():
    args = parse_args()

    print("=" * 80)
    print(f"Loading config: {args.config}")
    print("=" * 80)

    cfg = Config.fromfile(args.config)

    if getattr(cfg, "plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        project_root = os.getcwd()

        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        plugin_abs_path = os.path.join(project_root, plugin_dir)
        if plugin_abs_path not in sys.path:
            sys.path.insert(0, plugin_abs_path)

        importlib.import_module("projects.mmdet3d_plugin")
        print("[PLUGIN] Plugin imported successfully")

    cfg.model.decoder = None
    cfg.model.heads = None
    cfg.model.train_cfg = None

    if "camera" in cfg.model.encoders:
        cfg.model.encoders.camera.backbone.init_cfg = None

    from projects.mmdet3d_plugin.models.builder import build_model
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    if args.show_model:
        print(model)

    load_partial_checkpoint(model, args.checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()


    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    print(f"[DATA] Test dataset size: {len(dataset)}")

    data = next(iter(dataloader))

    print("\n[INPUT]")
    for k, v in data.items():
        print(f"  {k}: {type(v)}")


    img = unwrap_dc(data["img"])
    points = unwrap_dc(data["points"])
    meta0, metas = unwrap_metas(data["img_metas"])


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

    # no depth supervision in test
    depths = None


    with torch.no_grad():
        bevfeature, camerafeature = model(
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
            depths=depths,
            radar=None,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
        )

    print("\n[OUTPUT]")
    print(f"  bevfeature:   {tuple(bevfeature.shape)}")
    print(f"  camerafeature:{tuple(camerafeature.shape)}")


if __name__ == "__main__":
    main()
