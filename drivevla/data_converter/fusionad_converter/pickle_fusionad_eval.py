"""
Pickle per-sample FusionAD eval outputs (boxes_3d / scores_3d / labels_3d /
track_ids / ret_iou) for the val split — the full payload needed by
NuScenesE2EDataset.evaluate to compute AMOTA / NDS / map IoU.

Uses the same streaming inference path as pickle_fusionad_pth.py (contiguous
per-rank blocks via the custom DistributedSampler, can_bus delta fix applied
inside get_results_for_eval), but writes forward_test-shaped dicts instead of
vlm-shaped ones.

Launch:
    # single-GPU val
    python drivevla/data_converter/fusionad_converter/pickle_fusionad_eval.py --nuscenes_set val

    # multi-GPU val (val is ~6019 samples, <10 min on 8 GPUs)
    torchrun --nproc_per_node=8 drivevla/data_converter/fusionad_converter/pickle_fusionad_eval.py --nuscenes_set val
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
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmengine import Config

from llava.model.multimodal_encoder.fusionad_track_map import FusionADVisionTower
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import (
    DistributedSampler as ContiguousDistributedSampler,
)
from drivevla.utils.tensor_utils import move_data_to_device
from drivevla.data_converter.fusionad_converter.pickle_fusionad_pth import (
    collate_no_datacontainer_fusionad,
    setup_distributed,
)

import projects.fusionad_plugin_new.datasets.pipelines.loading  # noqa: F401
import projects.fusionad_plugin_new.datasets.pipelines  # noqa: F401

warnings.filterwarnings("ignore")
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description='Pickle FusionAD eval outputs')
    parser.add_argument('--nuscenes_set', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='data/fusionad_eval_results')
    parser.add_argument('--ckpt', type=str,
                        default='/home/s56cai/ckpt/fusionad/fusion_latest.pth')
    parser.add_argument('--zero-camera', action='store_true',
                        help='Zero camera features at get_bevs — modality ablation')
    parser.add_argument('--zero-lidar', action='store_true',
                        help='Zero lidar features at get_bevs — modality ablation')
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    gpu_id = local_rank if distributed else args.gpu
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    if args.nuscenes_set == 'val':
        dataset_cfg['ann_file'] = cfg.ann_file_val

    dataset = NuScenesE2EDataset(**dataset_cfg)

    if distributed:
        sampler = ContiguousDistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_no_datacontainer_fusionad,
        num_workers=4,
        pin_memory=True,
    )

    vt_cfg = addict.Dict()
    vt_cfg.vision_tower_pretrained = args.ckpt
    vt_cfg.vision_tower_test_mode = True
    vision_tower = FusionADVisionTower('fusionad_track_map', vt_cfg).to(device).eval()
    # FusionADVisionTower.vision_tower -> FusionADTrackMapModel.vision_model -> FusionAD
    vision_tower.vision_tower.vision_model.zero_camera = args.zero_camera
    vision_tower.vision_tower.vision_model.zero_lidar = args.zero_lidar

    if args.zero_camera and args.zero_lidar:
        ablation_suffix = '_no_both'
    elif args.zero_camera:
        ablation_suffix = '_no_camera'
    elif args.zero_lidar:
        ablation_suffix = '_no_lidar'
    else:
        ablation_suffix = ''
    output_dir = os.path.join(args.out_dir + ablation_suffix, args.nuscenes_set)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    desc = f'eval-pickling {args.nuscenes_set} (rank {rank}/{world_size})'
    iterator = tqdm(dataloader, desc=desc) if rank == 0 else dataloader
    for data in iterator:
        data = move_data_to_device(data, device)

        with torch.no_grad():
            res = vision_tower.get_results_for_eval(data)

        sample_token = res['token']
        output_path = os.path.join(output_dir, f'{sample_token}.pth')
        # Keep fp32: eval uses NuScenes detection API which compares small
        # bbox numbers — no upside to fp16 here, and eval tables use fp32.
        torch.save(res, output_path)

    if distributed:
        dist.barrier()
    if rank == 0:
        saved = len([f for f in os.listdir(output_dir) if f.endswith('.pth')])
        print(f"Done. {saved} eval .pth files written to {output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
