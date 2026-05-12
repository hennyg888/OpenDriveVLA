"""
Offline feature extraction for FusionAD vision tower (multi-GPU capable).

Runs FusionAD inference on each NuScenes sample and saves results_for_vlm
as individual .pth files (one per sample token) for fast loading during
LLM training.

Multi-GPU is safe here because the repo's custom DistributedSampler at
`projects/mmdet3d_plugin/datasets/samplers/distributed_sampler.py:36-38`
slices indices as contiguous per-rank blocks (NOT the stock PyTorch stride
pattern):

    per_replicas = total_size // num_replicas
    indices[rank * per_replicas : (rank + 1) * per_replicas]

The pkl is sorted by global timestamp and NuScenes scenes are temporally
disjoint, so each rank's block contains whole scenes in order. Combined
with the can_bus delta fix in get_results_for_vlm, the streaming tracker's
prev_bev / test_track_instances state stays valid within each rank's slice.
Only the first frame of each rank's first (possibly partial) scene pays a
cold-start cost.

Launch:
    # single-GPU
    python drivevla/data_converter/fusionad_converter/pickle_fusionad_pth.py --nuscenes_set train

    # multi-GPU (e.g. 8 GPUs)
    torchrun --nproc_per_node=8 drivevla/data_converter/fusionad_converter/pickle_fusionad_pth.py --nuscenes_set train
"""

import os
import sys

# Ensure the repo root is on sys.path regardless of where the script is launched from.
# Script lives at: <REPO>/drivevla/data_converter/fusionad_converter/pickle_fusionad_pth.py
# Three '..' walk back up: fusionad_converter -> data_converter -> drivevla -> REPO root.
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
from mmcv.parallel import collate

from llava.model.multimodal_encoder.fusionad_track_map import FusionADVisionTower
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import (
    DistributedSampler as ContiguousDistributedSampler,
)
from drivevla.utils.remove_mmlab_datacontainer import remove_datacontainer
from drivevla.utils.tensor_utils import move_data_to_device, change_tensor_to_float16

# Plugin registrations required by fusion_base_track_map.py's pipeline
# (LoadMultiViewImageFromFilesInCeph, LoadPointsFromFileInCeph, etc.).
# The config has `plugin=True, plugin_dir="projects/fusionad_plugin/"` but
# Config.fromfile does NOT auto-import plugin modules — we must do it here.
import projects.fusionad_plugin_new.datasets.pipelines.loading  # noqa: F401
import projects.fusionad_plugin_new.datasets.pipelines  # noqa: F401

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
    """Init torch.distributed if launched under torchrun; else return single-GPU defaults."""
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        dist.init_process_group(backend='nccl', init_method='env://',
                                rank=rank, world_size=world_size)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def main():
    parser = argparse.ArgumentParser(description='Offline FusionAD feature extraction')
    parser.add_argument('--nuscenes_set', type=str, choices=['train', 'val'], required=True,
                        help='NuScenes split to process')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Single-GPU only: which GPU to use (ignored under torchrun).')
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    gpu_id = local_rank if distributed else args.gpu
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------ dataset
    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    if args.nuscenes_set == 'val':
        dataset_cfg['ann_file'] = cfg.ann_file_val

    dataset = NuScenesE2EDataset(**dataset_cfg)

    # The custom DistributedSampler slices as contiguous per-rank blocks, so
    # scene continuity is preserved within each rank's slice. shuffle=False
    # makes it iterate the pkl in its native timestamp-sorted order.
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
    if distributed:
        dist.barrier()

    # ----------------------------------------------------------- extraction loop
    # Only rank 0 shows the tqdm bar so the log stays readable.
    desc = f'pickling {args.nuscenes_set} (rank {rank}/{world_size})'
    iterator = tqdm(dataloader, desc=desc) if rank == 0 else dataloader
    for data in iterator:
        data = move_data_to_device(data, device)

        with torch.no_grad():
            results_for_vlm = vision_tower(data)

        sample_token = results_for_vlm['sample_token']
        output_path = os.path.join(output_dir, f'{sample_token}.pth')

        # Extract planning_gt and img_metas from the data batch (available
        # because train_fusionad_with_track_gt loads GT annotations).
        img_metas_raw = data.get('img_metas', [[{}]])[0]  # first temporal step
        if isinstance(img_metas_raw, (list, tuple)):
            img_metas_raw = img_metas_raw[0]
        results_for_vlm['img_metas'] = {
            'filename': img_metas_raw.get('filename', []),
            'can_bus':  img_metas_raw.get('can_bus', None),
            'sample_idx': img_metas_raw.get('sample_idx', sample_token),
        }

        sdc_planning = data.get('sdc_planning', None)
        command      = data.get('command', None)
        if sdc_planning is not None:
            # sdc_planning arrives as a list-of-tensors (queue); keep last step
            if isinstance(sdc_planning, (list, tuple)):
                sdc_planning = sdc_planning[-1]
            results_for_vlm['planning_gt'] = {
                'sdc_planning': sdc_planning,
                'command':      command,
            }

        results_for_vlm = change_tensor_to_float16(results_for_vlm)
        torch.save(results_for_vlm, output_path)

    if distributed:
        dist.barrier()
    if rank == 0:
        saved = len([f for f in os.listdir(output_dir) if f.endswith('.pth')])
        print(f"Done. {saved} .pth files written to {output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
