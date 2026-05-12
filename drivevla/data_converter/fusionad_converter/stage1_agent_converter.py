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
        "conversation": [
            {"role": "user", "content": "Please provide a caption and the BEV coordinate for the following object:<track start><OBJECT><track end>"},
            {"role": "assistant", "content": "A barrier is moving slowly. It is in the back of ego car and in the back of a barrier. The BEV coordinate is (-20.15,41.87)."}
        ]
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
import re
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

FUSIONAD_PTH_DIR = "/home/s56cai/OpenDriveVLA/data/fusionad_results_for_vlm"
SCENE_DATA_PATH  = "/home/s56cai/OpenDriveVLA/data/nuCaption/stage1_scene_data.json"
AGENT_DATA_PATH  = "/home/s56cai/OpenDriveVLA/data/TOD3DCap/stage1_agent_data.jsonl"
BBOX_TOKEN_MAP   = "/home/s56cai/OpenDriveVLA/data/TOD3DCap/final_caption_bbox_token.json"
NUSCENES_ROOT    = "/home/s56cai/OpenDriveVLA/data/nuscenes"
NUSCENES_VER     = "v1.0-trainval"

# ins_inds_add_1=True in FusionAD pipeline config → keys in track_gt_inds_to_embed_idx
# are nusc.getind('instance', token) + 1
INS_INDS_ADD_1 = True

# Role name mapping: role/content → from/value
_ROLE_MAP = {"user": "human", "assistant": "gpt"}


def get_lidar_transforms(nusc, sample_token):
    """
    Return (ego_t, ego_R, lidar_t, lidar_R) for LIDAR_TOP at sample_token.
    Mirrors check_cached_agent_hist.py's transform setup.
    """
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_t = np.array(ego_pose["translation"])
    ego_R = Quaternion(ego_pose["rotation"])
    lidar_t = np.array(cs["translation"])
    lidar_R = Quaternion(cs["rotation"])
    return ego_t, ego_R, lidar_t, lidar_R


def world_to_lidar(world_pos, ego_t, ego_R, lidar_t, lidar_R):
    """Global → ego → lidar frame (x=fwd, y=left). Mirrors check_cached_agent_hist.py."""
    p = ego_R.inverse.rotate(np.array(world_pos) - ego_t)
    p = lidar_R.inverse.rotate(p - lidar_t)
    return p[:2]


def get_current_bev_coord(nusc, instance_token, sample_token):
    """
    Compute the current ego-relative BEV coordinate for an instance at sample_token.

    Uses global → ego → lidar transform (Method 2 from check_cached_agent_hist.py),
    matching the coordinate frame of cached_nuscenes_info.pkl.
    Returns (x, y) in lidar frame (x=fwd, y=left), or None if not annotated.
    """
    ego_t, ego_R, lidar_t, lidar_R = get_lidar_transforms(nusc, sample_token)

    sample = nusc.get("sample", sample_token)
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["instance_token"] == instance_token:
            xy = world_to_lidar(ann["translation"], ego_t, ego_R, lidar_t, lidar_R)
            return float(xy[0]), float(xy[1])
    return None


def replace_bev_in_conversation(conversations, bev_coord):
    """
    Replace 'The BEV coordinate is (...)' in the gpt turn with the computed coordinate.
    Operates on from/value formatted conversation list.
    """
    if bev_coord is None:
        return conversations
    x, y = bev_coord
    new_coord_str = f"({x:.2f},{y:.2f})"
    result = []
    for turn in conversations:
        if turn.get("from") == "gpt":
            turn = dict(turn)
            turn["value"] = re.sub(
                r"The BEV coordinate is \([^)]*\)",
                f"The BEV coordinate is {new_coord_str}",
                turn["value"],
            )
        result.append(turn)
    return result


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

            conversations = _to_from_value(item["conversation"])
            bev_coord = get_current_bev_coord(nusc, instance_token, sample_token)
            conversations = replace_bev_in_conversation(conversations, bev_coord)

            entries.append({
                "id": bbox_token,
                "uniad_pth": os.path.join(pth_split_dir, f"{sample_token}.pth"),
                "instance_token": instance_token,
                "conversations": conversations,
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

    agent_entries = convert_agent_data(args.split, args.pth_dir, nusc)

    agent_path = os.path.join(args.output_dir, f"stage1_agent_{args.split}.json")
    with open(agent_path, "w") as f:
        json.dump(agent_entries, f, indent=2)
    print(f"Saved agent data → {agent_path}")
    print(f"Total: {len(agent_entries)} agent entries")


if __name__ == "__main__":
    main()