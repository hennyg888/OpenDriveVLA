"""
Convert Stage 1 QA data (scene captions + agent captions) to training format
compatible with LLaVANuScenesDataset (use_uniad_pth=True, skip_build_conversation=True).

Agent entries are pre-filtered: only instances that FusionAD actually tracked
(i.e. instance_ind+1 appears in track_gt_inds_to_embed_idx of the .pth file)
are included. This avoids the runtime _rand_another retry in the dataset.

Output format per entry:
    {
        "id": "<sample_token or bbox_token>",
        "uniad_pth": "data/fusionad_results_for_vlm/train/<sample_token>.pth",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt",   "value": "..."}
        ],
        # agent entries only:
        "instance_token": "<instance_token>"
    }

Usage:
    python drivevla/data_converter/fusionad_converter/stage1_fusionad_convertor.py \
        --split train \
        --output_dir data/stage1
"""

import json
import os
import argparse
import yaml
import torch
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

FUSIONAD_PTH_DIR = "data/fusionad_results_for_vlm"
SCENE_DATA_PATH  = "data/nuCaption/stage1_scene_data.json"
AGENT_DATA_PATH  = "data/TOD3DCap/stage1_agent_data.jsonl"
BBOX_TOKEN_MAP   = "data/TOD3DCap/final_caption_bbox_token.json"
NUSCENES_ROOT    = "data/nuscenes"
NUSCENES_VER     = "v1.0-trainval"

# ins_inds_add_1=True in FusionAD pipeline config → keys in track_gt_inds_to_embed_idx
# are nusc.getind('instance', token) + 1
INS_INDS_ADD_1 = True

# Role name mapping: role/content → from/value
_ROLE_MAP = {"user": "human", "assistant": "gpt"}


def _to_from_value(conv_list):
    """Convert role/content conversation list to from/value format."""
    result = []
    for turn in conv_list:
        if "from" in turn:
            result.append({"from": turn["from"], "value": turn["value"]})
        else:
            from_key = _ROLE_MAP.get(turn["role"], turn["role"])
            result.append({"from": from_key, "value": turn["content"]})
    return result


def convert_scene_data(split, pth_dir):
    with open(SCENE_DATA_PATH) as f:
        raw = json.load(f)

    pth_split_dir = os.path.join(pth_dir, split)
    available_pth = set(
        fn.replace(".pth", "")
        for fn in os.listdir(pth_split_dir)
        if fn.endswith(".pth")
    )

    entries = []
    skipped = 0
    for item in raw:
        if item["split"] != split:
            continue
        token = item["sample_token"]
        if token not in available_pth:
            skipped += 1
            continue
        entries.append({
            "id": token,
            "uniad_pth": os.path.join(pth_split_dir, f"{token}.pth"),
            "conversations": _to_from_value(item["conversation"]),
        })

    print(f"[scene] {split}: {len(entries)} entries ({skipped} skipped – no .pth)")
    return entries


def convert_agent_data(split, pth_dir, nusc):
    with open(BBOX_TOKEN_MAP) as f:
        bbox_map = json.load(f)  # bbox_token → {sample_token, instance_token, ...}

    pth_split_dir = os.path.join(pth_dir, split)

    # Load all track_gt_inds_to_embed_idx from .pth files into memory
    # (keyed by sample_token for fast lookup)
    print(f"[agent] Loading .pth track indices for {split} split...")
    pth_track_inds = {}  # sample_token → set of tracked instance inds
    for fn in tqdm(os.listdir(pth_split_dir)):
        if not fn.endswith(".pth"):
            continue
        sample_token = fn.replace(".pth", "")
        try:
            data = torch.load(os.path.join(pth_split_dir, fn), map_location="cpu")
            track_dict = (
                data.get("result_track", {})
                    .get("track_gt_inds_to_embed_idx", {})
            )
            pth_track_inds[sample_token] = set(track_dict.keys())
        except Exception:
            pth_track_inds[sample_token] = set()

    entries = []
    skipped_no_map    = 0
    skipped_no_pth    = 0
    skipped_not_tracked = 0

    with open(AGENT_DATA_PATH) as f:
        for line in f:
            item = json.loads(line.strip())
            bbox_token = item["sample_id"]

            if bbox_token not in bbox_map:
                skipped_no_map += 1
                continue

            meta = bbox_map[bbox_token]
            sample_token   = meta["sample_token"]
            instance_token = meta["instance_token"]

            if sample_token not in pth_track_inds:
                skipped_no_pth += 1
                continue

            # Compute instance_ind the same way the training dataset does:
            # nusc.getind('instance', instance_token) + 1  (ins_inds_add_1=True)
            instance_ind = nusc.getind("instance", instance_token)
            if INS_INDS_ADD_1:
                instance_ind += 1

            if instance_ind not in pth_track_inds[sample_token]:
                skipped_not_tracked += 1
                continue

            entries.append({
                "id": bbox_token,
                "uniad_pth": os.path.join(pth_split_dir, f"{sample_token}.pth"),
                "instance_token": instance_token,
                "conversations": _to_from_value(item["conversation"]),
            })

    print(
        f"[agent] {split}: {len(entries)} entries kept\n"
        f"         skipped – no bbox map:    {skipped_no_map}\n"
        f"         skipped – no .pth:        {skipped_no_pth}\n"
        f"         skipped – not tracked:    {skipped_not_tracked}"
    )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output_dir", default="data/stage1")
    parser.add_argument("--pth_dir", default=FUSIONAD_PTH_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading NuScenes...")
    nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

    scene_entries = convert_scene_data(args.split, args.pth_dir)
    agent_entries = convert_agent_data(args.split, args.pth_dir, nusc)

    scene_path = os.path.join(args.output_dir, f"stage1_scene_{args.split}.json")
    agent_path = os.path.join(args.output_dir, f"stage1_agent_{args.split}.json")
    yaml_path  = os.path.join(args.output_dir, f"stage1_combined_{args.split}.yaml")

    with open(scene_path, "w") as f:
        json.dump(scene_entries, f, indent=2)
    print(f"Saved scene data → {scene_path}")

    with open(agent_path, "w") as f:
        json.dump(agent_entries, f, indent=2)
    print(f"Saved agent data → {agent_path}")

    yaml_content = {
        "datasets": [
            {"json_path": scene_path, "sampling_strategy": "all"},
            {"json_path": agent_path, "sampling_strategy": "all"},
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Saved combined YAML  → {yaml_path}")

    print(f"Total: {len(scene_entries)} scene + {len(agent_entries)} agent = {len(scene_entries)+len(agent_entries)}")


if __name__ == "__main__":
    main()
