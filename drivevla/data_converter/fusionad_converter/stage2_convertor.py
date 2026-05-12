"""
Build Stage 2 instruction tuning data from 3 sources:
  1. NuScenes-QA  — perception/reasoning QA (376604 questions, 5 types)
  2. nuCaption    — scene caption QA (161845 questions)
  3. NuX          — narration + reasoning (28130 samples)

Each entry uses the full prompt with real ego state and history info
filled at conversion time from cached_nuscenes_info.pkl (Option A).

Prompt format (human turn):
    Scene information: <scene_start><SCENE><scene_end>
    Object-wise tracking information: <track_start><TRACK><track_end>
    Map information: <map_start><MAP><map_end>
    Ego states: {ego_message}
    Historical trajectory (last 2 seconds): {his_message}
    Please answer the following question: {question}

Output per entry:
    {
        "id": "<sample_token>_<source>_<idx>",
        "uniad_pth": "data/fusionad_results_for_vlm/train/<sample_token>.pth",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt",   "value": "..."}
        ]
    }

Usage:
    python drivevla/data_converter/fusionad_converter/stage2_convertor.py \
        --output_dir data/stage2
"""

import os
import json
import pickle
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from llava.constants import (
    DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_MAP_END_TOKEN,
)

FUSIONAD_PTH_DIR    = "data/fusionad_results_for_vlm/train"
CACHED_NUSCENES_PKL = "data/nuscenes/cached_nuscenes_info.pkl"
NUSCENES_QA_PATH    = "data/nuscenes-QA/NuScenes_train_questions.json"
NUCAPTION_PATH      = "data/nuCaption/train.json"
NUX_PATH            = "data/nuX/Nu_X_train.json"


def build_available_pth_set(pth_dir: str) -> set:
    return {fn.replace(".pth", "") for fn in os.listdir(pth_dir) if fn.endswith(".pth")}


def generate_ego_and_history(data_dict: dict) -> tuple[str, str]:
    """Extract ego state and history strings from cached_nuscenes_info entry."""
    lcf = data_dict['gt_ego_lcf_feat']
    his_diff = data_dict['gt_ego_his_diff']
    his_trajs = data_dict['gt_ego_his_trajs']

    vx    = lcf[0] * 0.5
    vy    = lcf[1] * 0.5
    v_yaw = lcf[4]
    ax    = his_diff[-1, 0] - his_diff[-2, 0]
    ay    = his_diff[-1, 1] - his_diff[-2, 1]
    cx    = lcf[2]
    cy    = lcf[3]
    vhead = lcf[7] * 0.5
    steer = lcf[8]

    ego_msg = (
        f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
        f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
        f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
        f" - Can Bus: ({cx:.2f},{cy:.2f})"
        f" - Heading Speed: ({vhead:.2f})"
        f" - Steering: ({steer:.2f})"
    )

    pts = his_trajs[:4]   # [4, 2]
    his_msg = "[" + ",".join(f"({p[0]:.2f},{p[1]:.2f})" for p in pts) + "]"

    return ego_msg, his_msg


def build_human_turn(ego_msg: str, his_msg: str, question: str) -> str:
    return (
        f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
        f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        f"Ego states: {ego_msg}\n"
        f"Historical trajectory (last 2 seconds): {his_msg}\n"
        f"Please answer the following question: {question}"
    )


def convert_nuscenes_qa(cached: dict, available: set) -> list:
    with open(NUSCENES_QA_PATH) as f:
        data = json.load(f)

    entries = []
    skipped_no_pth = 0
    skipped_no_cache = 0

    for q in tqdm(data['questions'], desc='[NuScenes-QA]'):
        token = q['sample_token']
        if token not in available:
            skipped_no_pth += 1
            continue
        if token not in cached:
            skipped_no_cache += 1
            continue

        ego_msg, his_msg = generate_ego_and_history(cached[token])
        entries.append({
            "id": f"{token}_nuqa_{q['template_type']}",
            "uniad_pth": os.path.join(FUSIONAD_PTH_DIR, f"{token}.pth"),
            "conversations": [
                {"from": "human", "value": build_human_turn(ego_msg, his_msg, q['question'])},
                {"from": "gpt",   "value": q['answer']},
            ],
        })

    print(f"[NuScenes-QA] {len(entries)} entries "
          f"({skipped_no_pth} no-pth, {skipped_no_cache} no-cache skipped)")
    return entries


def convert_nucaption(cached: dict, available: set) -> list:
    with open(NUCAPTION_PATH) as f:
        data = json.load(f)

    entries = []
    skipped_no_pth = 0
    skipped_no_cache = 0

    for item in tqdm(data, desc='[nuCaption]'):
        token = item['sample_token']
        if token not in available:
            skipped_no_pth += 1
            continue
        if token not in cached:
            skipped_no_cache += 1
            continue

        ego_msg, his_msg = generate_ego_and_history(cached[token])
        entries.append({
            "id": f"{token}_nucap",
            "uniad_pth": os.path.join(FUSIONAD_PTH_DIR, f"{token}.pth"),
            "conversations": [
                {"from": "human", "value": build_human_turn(ego_msg, his_msg, item['question'])},
                {"from": "gpt",   "value": item['answer']},
            ],
        })

    print(f"[nuCaption] {len(entries)} entries "
          f"({skipped_no_pth} no-pth, {skipped_no_cache} no-cache skipped)")
    return entries


def convert_nux(cached: dict, available: set) -> list:
    with open(NUX_PATH) as f:
        data = json.load(f)

    NUX_QUESTION = "Please provide a narration and reasoning for the current driving behavior."

    entries = []
    skipped_no_pth = 0
    skipped_no_cache = 0

    for token, val in tqdm(data.items(), desc='[NuX]'):
        if token not in available:
            skipped_no_pth += 1
            continue
        if token not in cached:
            skipped_no_cache += 1
            continue

        answer = f"Narration: {val['narration']}\nReasoning: {val['reasoning']}."
        ego_msg, his_msg = generate_ego_and_history(cached[token])
        entries.append({
            "id": f"{token}_nux",
            "uniad_pth": os.path.join(FUSIONAD_PTH_DIR, f"{token}.pth"),
            "conversations": [
                {"from": "human", "value": build_human_turn(ego_msg, his_msg, NUX_QUESTION)},
                {"from": "gpt",   "value": answer},
            ],
        })

    print(f"[NuX] {len(entries)} entries "
          f"({skipped_no_pth} no-pth, {skipped_no_cache} no-cache skipped)")
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/stage2')
    parser.add_argument('--pth_dir',    default=FUSIONAD_PTH_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading cached NuScenes info...")
    with open(CACHED_NUSCENES_PKL, 'rb') as f:
        cached = pickle.load(f)

    print("Building available .pth set...")
    available = build_available_pth_set(args.pth_dir)
    print(f"  {len(available)} .pth files found")

    nuqa_entries   = convert_nuscenes_qa(cached, available)
    nucap_entries  = convert_nucaption(cached, available)
    nux_entries    = convert_nux(cached, available)

    all_entries = nuqa_entries + nucap_entries + nux_entries
    print(f"\nTotal: {len(nuqa_entries)} NuQA + {len(nucap_entries)} nuCaption "
          f"+ {len(nux_entries)} NuX = {len(all_entries)}")

    out_path = os.path.join(args.output_dir, 'stage2_train.json')
    with open(out_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"Saved → {out_path}")

    yaml_path = os.path.join(args.output_dir, 'stage2_train.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(
            {'datasets': [{'json_path': out_path, 'sampling_strategy': 'all'}]},
            f, default_flow_style=False
        )
    print(f"Saved → {yaml_path}")


if __name__ == '__main__':
    main()
