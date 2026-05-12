"""
Rank-0 evaluation driver: load per-sample eval .pth files, reorder them to
match `NuScenesE2EDataset.data_infos`, and call `dataset.evaluate(...)` to
compute AMOTA / NDS / map IoU.

This assumes pickle_fusionad_eval.py has already been run against the same
split.

Launch:
    python drivevla/data_converter/fusionad_converter/eval_from_pth.py --nuscenes_set val
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import warnings
import logging
import time
import os.path as osp

import torch
from tqdm import tqdm
from mmengine import Config

from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset

import projects.fusionad_plugin_new.datasets.pipelines.loading  # noqa: F401
import projects.fusionad_plugin_new.datasets.pipelines  # noqa: F401

warnings.filterwarnings("ignore")
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nuscenes_set', choices=['train', 'val'], default='val')
    ap.add_argument('--pth_dir', default='data/fusionad_eval_results')
    ap.add_argument('--out', default=None,
                    help='Optional path to save the merged results list as .pkl (for re-eval).')
    ap.add_argument('--metric', nargs='+', default=['bbox'],
                    help='Metric arg forwarded to dataset.evaluate(). Usually ["bbox"].')
    ap.add_argument('--jsonfile_prefix', default=None,
                    help='Output dir for intermediate NuScenes eval JSON (default: temp dir).')
    args = ap.parse_args()

    # --- dataset (needed for evaluate + sample ordering) ---
    cfg = Config.fromfile('./projects/configs/fusionad/fusion_base_track_map.py')
    dataset_cfg = cfg.data.train_fusionad_with_track_gt.copy()
    dataset_cfg.pop('type')
    if args.nuscenes_set == 'val':
        dataset_cfg['ann_file'] = cfg.ann_file_val
    dataset = NuScenesE2EDataset(**dataset_cfg)

    pth_dir = os.path.join(args.pth_dir, args.nuscenes_set)
    avail = {f[:-4] for f in os.listdir(pth_dir) if f.endswith('.pth')}
    missing = []

    # --- rebuild the ordered results list in data_infos order ---
    results = []
    for i, info in enumerate(tqdm(dataset.data_infos, desc='loading pths')):
        token = info['token']
        p = os.path.join(pth_dir, f'{token}.pth')
        if token not in avail:
            missing.append(token)
            results.append(dict(token=token))  # placeholder; evaluate skips when no boxes_3d
            continue
        d = torch.load(p, map_location='cpu', weights_only=False)
        results.append(d)

    print(f"Loaded {len(results) - len(missing)} / {len(results)} samples "
          f"({len(missing)} missing)")

    if args.out:
        import pickle
        with open(args.out, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved merged results list to {args.out}")

    # --- run NuScenes evaluate ---
    kwargs = {}
    if args.jsonfile_prefix is None:
        kwargs['jsonfile_prefix'] = osp.join(
            'test',
            args.nuscenes_set,
            time.strftime('%Y%m%d_%H%M%S'),
        )
    else:
        kwargs['jsonfile_prefix'] = args.jsonfile_prefix

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.metric, **kwargs))

    print("Running dataset.evaluate() ...")
    out = dataset.evaluate(results, **eval_kwargs)
    print("\n====== Results ======")
    for k, v in sorted(out.items()):
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
