import torch
import os
import warnings
import os.path as osp
from tqdm import tqdm  
from typing import Dict, Sequence
import argparse
import addict

from torch.utils.data import DataLoader

from mmengine import Config
from mmcv.parallel import collate

from llava.model.multimodal_encoder.uniad_track_map import UniadTrackMapVisionTower
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from drivevla.utils.remove_mmlab_datacontainer import remove_datacontainer
from drivevla.utils.tensor_utils import move_data_to_device, change_tensor_to_float16

import logging
logging.getLogger('shapely.geos').setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

def pickle_results_for_vlm(output_dir, **kwargs):
    """
    Save E2E AD model intermediate outputs in .pth format
    
    Args:
        output_dir: directory to save the results
        kwargs: key-value pairs of E2E AD model intermediate outputs
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(kwargs, f'{output_dir}/{kwargs["sample_token"]}.pth')

def collate_no_datacontainer(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    assert len(instances) == 1, "Currently only one instance (batch_size=1) is supported for UniADTrackMapVisionTower inference"

    uniad_data = collate(instances)

    # remove DataContainer to avoid GPU memory leak
    uniad_data = remove_datacontainer(uniad_data)
    uniad_data['img_metas'] = uniad_data['img_metas'][0]
    uniad_data['img'] = uniad_data['img'][0]
    
    return uniad_data

def pickle_uniad_pth(pickle_nuscenes_set):

    cfg = Config.fromfile('./projects/configs/stage1_track_map/base_track_map.py')

    cfg.data.train_llava_with_track_gt.pop('type')
    if pickle_nuscenes_set == 'train':
        dataset = NuScenesE2EDataset(**cfg.data.train_llava_with_track_gt)
    else: # val
        cfg.data.train_llava_with_track_gt.ann_file = cfg.ann_file_test
        dataset = NuScenesE2EDataset(**cfg.data.train_llava_with_track_gt)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_no_datacontainer,
        num_workers=4,
        pin_memory=True
    )
    
    vision_tower_cfg = addict.Dict()
    vision_tower_cfg.vision_tower_pretrained='checkpoints/uniad_track_map'
    vision_tower_cfg.vision_tower_test_mode = True
    vision_tower = UniadTrackMapVisionTower(vision_tower="uniad_track_map", vision_tower_cfg=vision_tower_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in tqdm(dataloader, ncols=80, desc=f'pickling {pickle_nuscenes_set} set'):
        data = move_data_to_device(data, device)
        results_for_vlm = vision_tower(data)
        results_for_vlm = change_tensor_to_float16(results_for_vlm)
        pickle_results_for_vlm(
            f'data/uniad_results_for_vlm/{pickle_nuscenes_set}',
            **results_for_vlm
        )

def pickle_map_gt(pickle_nuscenes_set):

    cfg = Config.fromfile('./projects/configs/stage1_track_map/base_track_map.py')

    # if pickle_nuscenes_set == 'train':
    #     cfg.data.test.ann_file = cfg.ann_file_train
        
    cfg.data.train_llava_with_track_gt.pop('type')
    if pickle_nuscenes_set == 'train':
        dataset = NuScenesE2EDataset(**cfg.data.train_llava_with_track_gt)
    else: # val
        cfg.data.train_llava_with_track_gt.ann_file = cfg.ann_file_test
        dataset = NuScenesE2EDataset(**cfg.data.train_llava_with_track_gt)
           
    # cfg.data.train.pop('type')
    # dataset = NuScenesE2EDataset(**cfg.data.train)
    data_dict = {}
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_no_datacontainer,
        num_workers=4,
        pin_memory=False
    )
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    for data in tqdm(dataloader, ncols=80):
        data = move_data_to_device(data, device)
        sample_token = data['img_metas'][0][0]['sample_idx']
        single_data_dict = {
            'img_metas': data['img_metas'],
            'gt_lane_labels': data['gt_lane_labels'],
            'gt_lane_bboxes': data['gt_lane_bboxes'],
            'gt_lane_masks': data['gt_lane_masks']
        }
        output_path = f'data/uniad_results_for_vlm/{pickle_nuscenes_set}/{sample_token}.pth'
        if not os.path.exists(output_path): 
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(single_data_dict, output_path)
        else:
            print(f'{output_path} already exists')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pickle NuScenes dataset for VLM')
    parser.add_argument('--uniad_pth', action='store_true', help='Run pickle_uniad_pth')
    parser.add_argument('--map_gt', action='store_true', help='Run pickle_map_gt')
    parser.add_argument('--nuscenes_set', type=str, choices=['train', 'val'], required=True, help='Dataset split to process')
    
    args = parser.parse_args()
    
    if not args.uniad_pth and not args.map_gt:
        parser.error("At least one of --uniad_pth or --map_gt must be specified")
    
    if args.uniad_pth:
        pickle_uniad_pth(pickle_nuscenes_set=args.nuscenes_set)
    
    if args.map_gt:
        pickle_map_gt(pickle_nuscenes_set=args.nuscenes_set)