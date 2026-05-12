"""
Count GT-matched tracked queries across all saved .pth files.

For each sample, `result_track.track_gt_inds_to_embed_idx` is a dict mapping
GT instance index -> tracker embedding index (IoU > 0.01 matched entries).
Its length = number of GT-matched tracks for that sample.

Usage:
    python drivevla/data_converter/fusionad_converter/count_matched_tracks.py
    python drivevla/data_converter/fusionad_converter/count_matched_tracks.py --split val
"""
import argparse
import os
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--dir', default='data/fusionad_results_for_vlm')
    args = ap.parse_args()

    pth_dir = os.path.join(args.dir, args.split)
    files = sorted(f for f in os.listdir(pth_dir) if f.endswith('.pth'))

    total_matched = 0
    total_embed = 0
    samples_with_zero = 0
    load_errors = 0
    histogram = {}

    for fn in tqdm(files, desc=f'scanning {args.split}'):
        try:
            d = torch.load(os.path.join(pth_dir, fn), map_location='cpu', weights_only=False)
        except Exception:
            load_errors += 1
            continue

        rt = d.get('result_track', {})
        matched = len(rt.get('track_gt_inds_to_embed_idx', {}))
        embed = rt.get('track_query_embeddings')
        embed_n = embed.shape[0] if embed is not None else 0

        total_matched += matched
        total_embed += embed_n
        if matched == 0:
            samples_with_zero += 1
        histogram[matched] = histogram.get(matched, 0) + 1

    n = len(files)
    print(f"\nSplit:                       {args.split}")
    print(f"Total .pth files:            {n}")
    print(f"Load errors:                 {load_errors}")
    print(f"Samples with 0 matched:      {samples_with_zero}")
    print(f"Total matched (sum len):     {total_matched}")
    print(f"Total embed queries:         {total_embed}")
    print(f"Avg matched / sample:        {total_matched / max(1, n - load_errors):.3f}")
    print(f"Avg embed   / sample:        {total_embed   / max(1, n - load_errors):.3f}")
    print(f"Match rate (matched/embed):  "
          f"{(total_matched / max(1, total_embed)) * 100:.1f}%")

    print(f"\nHistogram of matched-per-sample (top 20):")
    for k in sorted(histogram)[:20]:
        print(f"  {k:3d}  : {histogram[k]}")
    tail = sorted(histogram)[-5:]
    if tail and tail[0] > 19:
        print("  ...")
        for k in tail:
            print(f"  {k:3d}  : {histogram[k]}")


if __name__ == '__main__':
    main()
