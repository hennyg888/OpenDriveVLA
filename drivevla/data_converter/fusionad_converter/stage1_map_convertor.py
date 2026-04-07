"""
Build Stage 1 map caption training data from FusionAD seg predictions.

Caption text uses predicted instance counts from result_seg['seg_pred_labels']
stored in per-sample .pth files (produced by pickle_fusionad_pth.py).

FusionAD panseg_head has 3 thing classes (num_things_classes=3):
  class 0 → divider  (road_divider + lane_divider merged, same as VectorizedLocalMap CLASS2LABEL)
  class 1 → ped_crossing
  class 2 → contour  (road boundary)

Caption format:

  Q: Please provide a caption for the following map: <map_start><MAP><map_end>
  A: This scene contains N lane-related elements:
     There are X instances of divider such as lane dividers or road dividers.
     There are Y instances of pedestrian crosswalk marked with white stripes.
     There are Z instances of road boundary marking the edge of the drivable area.

The <MAP> token embedding is supplied at training time from result_seg in the .pth file.

Usage:
    export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH
    python drivevla/data_converter/fusionad_converter/stage1_map_convertor.py \
        --split train --output_dir data/stage1
"""

import os
import json
import argparse
import torch
from tqdm import tqdm

FUSIONAD_PTH_DIR = "data/fusionad_results_for_vlm"

MAP_QUESTION = (
    "Please provide a caption for the following map: "
    "<map_start><MAP><map_end>"
)

# FusionAD panseg_head thing class index → caption label
# class 0: divider (road_divider + lane_divider merged — same CLASS2LABEL as VectorizedLocalMap)
# class 1: ped_crossing
# class 2: contour (road boundary)
CLASS_LABELS = {
    0: 'divider such as lane dividers or road dividers',
    1: 'pedestrian crosswalk marked with white stripes',
    2: 'road boundary marking the edge of the drivable area',
}


def count_from_seg_pred_labels(seg_pred_labels) -> dict:
    """Count predicted instances per class from seg_pred_labels tensor."""
    counts = {0: 0, 1: 0, 2: 0}
    if seg_pred_labels is None:
        return counts
    if not isinstance(seg_pred_labels, torch.Tensor):
        seg_pred_labels = torch.tensor(seg_pred_labels)
    for cls_idx in [0, 1, 2]:
        counts[cls_idx] = int((seg_pred_labels == cls_idx).sum().item())
    return counts


def generate_caption(counts: dict) -> str | None:
    """Build answer text. Returns None if no map elements are present."""
    total = sum(counts.values())
    if total == 0:
        return None

    lines = []
    for cls_idx, label in CLASS_LABELS.items():
        n = counts[cls_idx]
        if n == 0:
            continue
        if n == 1:
            lines.append(f'There is 1 instance of {label}.')
        else:
            lines.append(f'There are {n} instances of {label}.')

    return f'This scene contains {total} lane-related elements:\n' + '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',      choices=['train', 'val'], default='train')
    parser.add_argument('--output_dir', default='data/stage1')
    parser.add_argument('--pth_dir',    default=FUSIONAD_PTH_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pth_split_dir = os.path.join(args.pth_dir, args.split)
    pth_files = [fn for fn in os.listdir(pth_split_dir) if fn.endswith('.pth')]

    entries        = []
    skipped_no_seg = 0
    skipped_empty  = 0

    for fn in tqdm(pth_files, desc=f'[map] {args.split}'):
        token = fn.replace('.pth', '')
        pth_path = os.path.join(pth_split_dir, fn)

        try:
            data = torch.load(pth_path, map_location='cpu')
        except Exception as e:
            import warnings
            warnings.warn(f'Failed to load {pth_path}: {e}')
            continue

        result_seg = data.get('result_seg', {})
        seg_pred_labels = result_seg.get('seg_pred_labels', None)

        if seg_pred_labels is None:
            skipped_no_seg += 1
            continue

        counts  = count_from_seg_pred_labels(seg_pred_labels)
        caption = generate_caption(counts)

        if caption is None:
            skipped_empty += 1
            continue

        entries.append({
            'id':        token,
            'uniad_pth': pth_path,
            'conversations': [
                {'from': 'human', 'value': MAP_QUESTION},
                {'from': 'gpt',   'value': caption},
            ],
        })

    out_path = os.path.join(args.output_dir, f'stage1_map_{args.split}.json')
    with open(out_path, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f'[map] {args.split}: {len(entries)} entries → {out_path}')
    print(f'       skipped (no seg_pred_labels): {skipped_no_seg}')
    print(f'       skipped (empty scene):        {skipped_empty}')

    yaml_path = os.path.join(args.output_dir, f'stage1_combined_{args.split}.yaml')
    if os.path.exists(yaml_path):
        import yaml
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        existing = {d['json_path'] for d in yaml_data['datasets']}
        if out_path not in existing:
            yaml_data['datasets'].append({'json_path': out_path, 'sampling_strategy': 'all'})
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            print(f'Updated YAML → {yaml_path}')


if __name__ == '__main__':
    main()
