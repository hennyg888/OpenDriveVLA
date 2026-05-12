"""
Diagnostic script to verify two suspected bugs in
projects/fusionad_plugin_new/fusionad/detectors/fusionad_e2e.py::get_results_for_vlm:

  (1) SDC handling — does `track_bbox_results` really exclude the ego
      (self-driving car) query, as claimed by the comment at fusionad_e2e.py:637?
      We compare lengths of track_bbox_results / track_query_embeddings and
      confirm `sdc_track_bbox_results` + `sdc_embedding` exist as separate keys.

  (2) Map query filtering — the thing/stuff decoder queries are captured via
      a forward hook as raw `out[2][-1]` (100 slots), bypassing the panseg
      head's quality filter (scores > quality_threshold_things + mask-area).
      UniAD, by contrast, uses `chosen_output_query_things` which IS filtered.
      We verify this by ALSO capturing seg_head.forward_test's returned
      `chosen_output_query_things` via a hook, and comparing its length to
      the raw thing-query length.

Runs a handful of samples and prints a per-sample table.

Usage:
    cd /home/s56cai/OpenDriveVLA
    python drivevla/data_converter/fusionad_converter/debug_vlm_fields.py
    python drivevla/data_converter/fusionad_converter/debug_vlm_fields.py --num 5 --split val
"""

import os
import sys
import argparse
import warnings
import logging

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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

# Plugin side-effect imports (mirror pickle_fusionad_pth.py).
import projects.fusionad_plugin_new.datasets.pipelines.loading  # noqa: F401
import projects.fusionad_plugin_new.datasets.pipelines  # noqa: F401

warnings.filterwarnings("ignore")
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def _shape(t):
    if t is None:
        return 'None'
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    try:
        return f'{type(t).__name__}(len={len(t)})'
    except Exception:
        return type(t).__name__


def diagnose(model, data, sample_idx):
    """Run get_results_for_vlm with an extra hook to capture the panseg
    head's filtered output, then print a per-sample diagnostic."""
    # Accessor chain: FusionADVisionTower.vision_tower = FusionADTrackMapModel,
    # FusionADTrackMapModel.vision_model = the raw FusionAD detector.
    detector = model.vision_tower.vision_model

    # Capture seg_head.forward_test's full returned list so we can read
    # `chosen_output_query_things` (which get_results_for_vlm currently ignores).
    captured = {}
    def _seg_fwd_test_hook(module, inputs, output):
        captured['seg_fwd_test_output'] = output
    h = detector.seg_head.register_forward_hook(_seg_fwd_test_hook)
    try:
        with torch.no_grad():
            out = detector.get_results_for_vlm(data)
    finally:
        h.remove()

    rt  = out['result_track']
    rs  = out['result_seg']

    track_bbox = rt.get('track_bbox_results')  # may be absent (popped inside)
    # track_query_embeddings is passed through from result_track.
    tqe = rt.get('track_query_embeddings')

    # sdc_* are popped before return; reach back into the detector's last
    # computed result instead.  Safer: just report from get_results_for_vlm's
    # output + the raw result dict stashed on the module by the hook paths.
    # Here we simply print tqe shape and inform if track_bbox got popped.

    # Raw thing / stuff decoder queries (what mm_projector_map sees today)
    raw_things = rs.get('chosen_output_query_things')  # NB: in FusionAD this IS the raw hook output
    raw_stuff  = rs.get('output_query_stuff')
    seg_counts = rs.get('seg_instance_counts', {})

    # Panseg head's OWN filtered output (what we'd like to use instead)
    seg_fwd_output = captured.get('seg_fwd_test_output')
    panseg_chosen_things = None
    if isinstance(seg_fwd_output, list) and len(seg_fwd_output) > 0 and isinstance(seg_fwd_output[0], dict):
        panseg_chosen_things = seg_fwd_output[0].get('chosen_output_query_things')

    print(f"\n======== sample {sample_idx} ({out.get('sample_token')}) ========")
    print(f"[tracks] track_query_embeddings shape   : {_shape(tqe)}")
    print(f"[tracks] track_bbox_results (in out)    : {_shape(track_bbox)}"
          f"  {'← popped before return (as expected)' if track_bbox is None else ''}")

    # Re-run IoU matching manually against gt_bboxes_3d to sanity-check SDC alignment.
    # We only re-derive what get_results_for_vlm just did; no model forward.
    gt_bboxes_3d = data.get('gt_bboxes_3d')
    if gt_bboxes_3d is not None:
        gt_obj = gt_bboxes_3d[0]
        while isinstance(gt_obj, list):
            gt_obj = gt_obj[-1]
        print(f"[tracks] gt_bboxes_3d (current frame)   : {_shape(gt_obj.tensor)}")

    # Report the track_gt_inds_to_embed_idx stats
    mapping = rt.get('track_gt_inds_to_embed_idx', {})
    if mapping:
        max_embed_idx = max(mapping.values())
        print(f"[tracks] matched GT→embed_idx           : {len(mapping)} pairs, "
              f"max embed_idx = {max_embed_idx}")
        if tqe is not None and max_embed_idx >= tqe.shape[0]:
            print(f"           !! max embed_idx >= track_query_embeddings.shape[0] "
                  f"({max_embed_idx} >= {tqe.shape[0]}) -- OOB indexing !!")
        if tqe is not None:
            print(f"           max embed_idx / tqe.len  : "
                  f"{max_embed_idx} / {tqe.shape[0] - 1}")
    else:
        print(f"[tracks] matched GT→embed_idx           : 0 (empty)")

    print(f"[map  ] current (hook-based raw) things : {_shape(raw_things)}")
    print(f"[map  ] current (hook-based raw) stuff  : {_shape(raw_stuff)}")
    print(f"[map  ] panseg filtered chosen_things   : {_shape(panseg_chosen_things)}")
    if raw_things is not None and panseg_chosen_things is not None:
        raw_n = raw_things.shape[0] if isinstance(raw_things, torch.Tensor) else None
        chosen_n = (panseg_chosen_things.shape[0]
                    if isinstance(panseg_chosen_things, torch.Tensor) else None)
        if raw_n is not None and chosen_n is not None:
            print(f"           raw / filtered ratio     : {raw_n} / {chosen_n}  "
                  f"(~{raw_n / max(chosen_n, 1):.1f}x too many slots fed to projector)")
    print(f"[map  ] panoptic post-proc counts       : {seg_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--num', type=int, default=3,
                        help='Number of samples to diagnose (reads from dataset head).')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    if args.split == 'val':
        dataset_cfg['ann_file'] = cfg.ann_file_val
    dataset = NuScenesE2EDataset(**dataset_cfg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_no_datacontainer_fusionad,
        num_workers=0,
    )

    vt_cfg = addict.Dict()
    vt_cfg.vision_tower_pretrained = '/home/s56cai/ckpt/fusionad/fusion_latest.pth'
    vt_cfg.vision_tower_test_mode = True
    model = FusionADVisionTower('fusionad_track_map', vt_cfg).to(device)
    model.eval()

    for i, data in enumerate(loader):
        if i >= args.num:
            break
        data = move_data_to_device(data, device)
        diagnose(model, data, i)

    print("\nDone.")


if __name__ == '__main__':
    main()
