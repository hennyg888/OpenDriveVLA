"""
Offline feature extraction for FusionAD vision tower.

Runs FusionAD inference on each NuScenes sample and saves results_for_vlm
as individual .pth files (one per sample token) for fast loading during
LLM training.

Multi-GPU launch:
    torchrun --nproc_per_node=4 pickle_fusionad_pth.py --nuscenes_set train

Single-GPU launch:
    python pickle_fusionad_pth.py --nuscenes_set train
"""

import os
import sys

# Ensure the repo root is on sys.path regardless of where the script is launched from
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import warnings
import logging

import addict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from mmengine import Config
from mmcv.parallel import collate

from llava.model.multimodal_encoder.fusionad_track_map import FusionADVisionTower
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from drivevla.utils.remove_mmlab_datacontainer import remove_datacontainer
from drivevla.utils.tensor_utils import move_data_to_device, change_tensor_to_float16

warnings.filterwarnings("ignore")
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def collate_no_datacontainer_fusionad(instances):
    """Collate a single-element batch, unwrap DataContainers including 'points'."""
    assert len(instances) == 1, (
        "Only batch_size=1 is supported for FusionADVisionTower inference."
    )

    batch = collate(instances)
    batch = remove_datacontainer(batch)

    # Unwrap the outer list dimension added by collate for test-mode data
    batch['img_metas'] = batch['img_metas'][0]
    batch['img'] = batch['img'][0]

    # Unwrap points — stored as list of lists after collate
    if 'points' in batch:
        points = batch['points']
        # points is [[DC]] after collate; remove_datacontainer unwraps DC,
        # leaving [[Tensor]] — unwrap the two list levels
        if isinstance(points, (list, tuple)) and len(points) > 0:
            inner = points[0]
            if isinstance(inner, (list, tuple)) and len(inner) > 0:
                batch['points'] = [inner[0]]
            else:
                batch['points'] = [inner]

    return batch


def setup_distributed():
    """Initialize the default process group if torchrun was used."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        return True
    return False


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def main():
    parser = argparse.ArgumentParser(description='Offline FusionAD feature extraction')
    parser.add_argument('--nuscenes_set', type=str, choices=['train', 'val'], required=True,
                        help='NuScenes split to process')
    args = parser.parse_args()

    is_distributed = setup_distributed()
    rank = get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------ dataset
    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    if args.nuscenes_set == 'val':
        dataset_cfg['ann_file'] = cfg.ann_file_val

    dataset = NuScenesE2EDataset(**dataset_cfg)

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_no_datacontainer_fusionad,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_no_datacontainer_fusionad,
            num_workers=4,
            pin_memory=True,
        )

    # ----------------------------------------------------------- vision tower
    vision_tower_cfg = addict.Dict()
    vision_tower_cfg.vision_tower_pretrained = '/home/s56cai/ckpt/fusionad/fusion_latest.pth'
    vision_tower_cfg.vision_tower_test_mode = True

    vision_tower = FusionADVisionTower('fusionad_track_map', vision_tower_cfg)
    vision_tower = vision_tower.to(device)
    vision_tower.eval()

    # ----------------------------------------------------------- output dir
    output_dir = f'data/fusionad_results_for_vlm/{args.nuscenes_set}'
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Wait until rank 0 has created the directory before other ranks start writing
    if is_distributed:
        dist.barrier()

    # ----------------------------------------------------------- extraction loop
    for data in tqdm(dataloader, desc=f'[rank {rank}] pickling {args.nuscenes_set}',
                     disable=(rank != 0)):
        data = move_data_to_device(data, device)

        with torch.no_grad():
            results_for_vlm = vision_tower(data)

        sample_token = results_for_vlm['sample_token']
        output_path = os.path.join(output_dir, f'{sample_token}.pth')

        results_for_vlm = change_tensor_to_float16(results_for_vlm)
        torch.save(results_for_vlm, output_path)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        saved = len([f for f in os.listdir(output_dir) if f.endswith('.pth')])
        print(f"Done. {saved} .pth files written to {output_dir}")


if __name__ == '__main__':
    main()
