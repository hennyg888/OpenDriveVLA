"""
Build NuScenes-QA val entries in the LLaVANuScenesDataset-expected schema,
with the exact Stage 2 prompt so the trained VLA sees a familiar input.

Mirrors `stage2_convertor.convert_nuscenes_qa` but:
  - reads the VAL questions (NuScenes_val_questions.json) instead of train.
  - points `uniad_pth` at data/fusionad_results_for_vlm/val/<token>.pth.
    Required so inference can use `--use-uniad-pth` -> turns off
    in_nuscenes_order dedup in the dataset and keeps ALL ~83k QA entries
    (otherwise the sample_token-keyed reorder dict collapses to 1 Q/sample).
  - carries top-level sidecar metadata (template_type, num_hop,
    gt_answer, sample_token, question) for the post-inference scorer.

Prereq: run pickle_fusionad_pth.py --nuscenes_set val first, so the val
pths exist under data/fusionad_results_for_vlm/val/.

Output: data/stage2_val_qa/val_qa.json
"""

import os
import sys
import json
import pickle
import argparse

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tqdm import tqdm

from drivevla.data_converter.fusionad_converter.stage2_convertor import (
    generate_ego_and_history,
    build_human_turn,
    build_available_pth_set,
    CACHED_NUSCENES_PKL,
)

VAL_QUESTIONS_PATH = "data/nuscenes-QA/NuScenes_val_questions.json"
FUSIONAD_PTH_DIR_VAL = "data/fusionad_results_for_vlm/val"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/stage2_val_qa/val_qa.json')
    parser.add_argument('--val_questions', default=VAL_QUESTIONS_PATH)
    parser.add_argument('--cached', default=CACHED_NUSCENES_PKL)
    parser.add_argument('--pth_dir', default=FUSIONAD_PTH_DIR_VAL)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading cached NuScenes info from {args.cached} ...")
    with open(args.cached, 'rb') as f:
        cached = pickle.load(f)

    print(f"Scanning available val pths in {args.pth_dir} ...")
    if not os.path.isdir(args.pth_dir):
        raise SystemExit(
            f"ERROR: {args.pth_dir} does not exist. Run "
            f"'pickle_fusionad_pth.py --nuscenes_set val' first."
        )
    available = build_available_pth_set(args.pth_dir)
    print(f"  {len(available)} val pths found")

    print(f"Loading val questions from {args.val_questions} ...")
    with open(args.val_questions) as f:
        val_data = json.load(f)
    questions = val_data['questions']

    entries = []
    skipped_no_cache = 0
    skipped_no_pth = 0
    for idx, q in enumerate(tqdm(questions, desc='[NuScenes-QA val]')):
        token = q['sample_token']
        if token not in cached:
            skipped_no_cache += 1
            continue
        if token not in available:
            skipped_no_pth += 1
            continue

        ego_msg, his_msg = generate_ego_and_history(cached[token])
        human_turn = build_human_turn(ego_msg, his_msg, q['question'])

        entries.append({
            "id": f"{token}_nuqa_{idx:06d}",
            "uniad_pth": os.path.join(args.pth_dir, f"{token}.pth"),
            "conversations": [
                {"from": "human", "value": human_turn},
                {"from": "gpt",   "value": q['answer']},
            ],
            "template_type": q['template_type'],
            "num_hop":       q['num_hop'],
            "gt_answer":     q['answer'],
            "sample_token":  token,
            "question":      q['question'],
        })

    print(f"[NuScenes-QA val] {len(entries)} entries "
          f"({skipped_no_cache} no-cache, {skipped_no_pth} no-pth skipped)")

    with open(args.output, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    main()
