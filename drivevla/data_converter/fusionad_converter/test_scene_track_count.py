"""
Minimal per-scene track-count test for the can_bus-delta fix.

Iterates through ONE complete scene (~40 frames) in the same scene-contiguous,
temporally ordered fashion as pickle_fusionad_pth.py, and prints per-frame:
  - # track_query_embeddings  (tracker's active IDs)
  - # track_gt_inds_to_embed_idx (GT-matched, IoU>0.01)

If the can_bus delta fix worked, track count should monotonically grow for
the first several mid-scene frames instead of staying at ~3-5 flat.

Usage:
    cd /home/s56cai/OpenDriveVLA
    python drivevla/data_converter/fusionad_converter/test_scene_track_count.py
    python drivevla/data_converter/fusionad_converter/test_scene_track_count.py --max_frames 40
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import warnings
import logging

import addict
import torch
from torch.utils.data import DataLoader
from mmengine import Config

from llava.model.multimodal_encoder.fusionad_track_map import FusionADVisionTower
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from drivevla.utils.tensor_utils import move_data_to_device
from drivevla.data_converter.fusionad_converter.pickle_fusionad_pth import (
    collate_no_datacontainer_fusionad,
)

import projects.fusionad_plugin_new.datasets.pipelines.loading  # noqa: F401
import projects.fusionad_plugin_new.datasets.pipelines  # noqa: F401

warnings.filterwarnings("ignore")
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_frames', type=int, default=45,
                    help='Hard cap (scene has ~40 frames; loop also auto-stops at scene change).')
    ap.add_argument('--ckpt', default='/home/s56cai/ckpt/fusionad/fusion_latest.pth')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    dataset = NuScenesE2EDataset(**dataset_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_no_datacontainer_fusionad,
        num_workers=0,
    )

    vt_cfg = addict.Dict()
    vt_cfg.vision_tower_pretrained = args.ckpt
    vt_cfg.vision_tower_test_mode = True
    vision_tower = FusionADVisionTower('fusionad_track_map', vt_cfg).to(device).eval()

    print(f"{'frame':>5}  {'scene_token':<34}  {'sample_token':<34}  "
          f"{'n_embed':>8}  {'n_matched':>9}")
    print('-' * 100)

    first_scene = None
    for i, data in enumerate(dataloader):
        if i >= args.max_frames:
            break

        meta_raw = data['img_metas']
        # Unwrap: collate_no_datacontainer_fusionad already strips outer list,
        # but the remaining shape is [meta_dict] for offline path.
        while isinstance(meta_raw, (list, tuple)) and len(meta_raw) > 0:
            meta_raw = meta_raw[0]
        scene_tok = meta_raw['scene_token']
        sample_tok = meta_raw['sample_idx']

        if first_scene is None:
            first_scene = scene_tok
        elif scene_tok != first_scene:
            print(f"(scene changed at frame {i}, stopping)")
            break

        data = move_data_to_device(data, device)
        with torch.no_grad():
            out = vision_tower(data)

        n_embed = out['result_track']['track_query_embeddings'].shape[0]
        n_matched = len(out['result_track']['track_gt_inds_to_embed_idx'])
        print(f"{i:>5}  {scene_tok:<34}  {sample_tok:<34}  "
              f"{n_embed:>8}  {n_matched:>9}")


if __name__ == '__main__':
    main()
