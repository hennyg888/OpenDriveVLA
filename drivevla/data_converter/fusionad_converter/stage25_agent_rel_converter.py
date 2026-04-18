"""
Convert Stage 2.5 QA data (object future trajectory) to training format
compatible with LLaVANuScenesDataset (use_uniad_pth=True, skip_build_conversation=True).

For each sample .pth file, tracked object instances are extracted from
track_gt_inds_to_embed_idx (same filtering logic as stage1_fusionad_convertor.py).
The conversation GPT response is the object's future trajectory: up to 6 waypoints in the
agent's own frame at the current sample (position (0,0), x=right, y=forward relative to
the agent's heading). Missing future frames are padded with (UN, UN).

Output format per entry:
    {
        "id": "<sample_token>_<instance_token>",
        "uniad_pth": "data/fusionad_results_for_vlm/<split>/<sample_token>.pth",
        "instance_token": "<instance_token>",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt",   "value": "(x1, y1), (x2, y2), ..., (UN, UN)"}
        ]
    }

Usage:
    python drivevla/data_converter/fusionad_converter/stage25_converter.py \
        --split train \
        --output_dir data/stage2.5
"""

import json
import os
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from llava.constants import (
    DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_MAP_END_TOKEN,
    DEFAULT_OBJECT_TOKEN,
)

FUSIONAD_PTH_DIR    = "/home/s56cai/OpenDriveVLA/data/fusionad_results_for_vlm"
NUSCENES_ROOT       = "/home/s56cai/OpenDriveVLA/data/nuscenes"
NUSCENES_VER        = "v1.0-trainval"
CACHED_NUSCENES_PKL = "/home/s56cai/OpenDriveVLA/data/nuscenes/cached_nuscenes_info.pkl"

# ins_inds_add_1=True in FusionAD pipeline config → keys in track_gt_inds_to_embed_idx
# are nusc.getind('instance', token) + 1
INS_INDS_ADD_1 = True

N_FUTURE = 6
DEBUG = False  # Toggle: set True (or pass --debug) to print ego past-waypoint diagnostics


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
    """Global → ego → lidar (mirrors check_cached_agent_hist.py method 2)."""
    p = ego_R.inverse.rotate(np.array(world_pos) - ego_t)
    p = lidar_R.inverse.rotate(p - lidar_t)
    return p[:2]


def world_to_agent_frame(world_pos, agent_world_pos, agent_world_R):
    """
    Transform a world position into the agent's local frame.

    NuScenes annotation rotation: local x=forward, y=left in world frame.
    Output convention: x=right, y=forward.

    Returns (x_right, y_fwd).
    """
    delta = np.array(world_pos) - agent_world_pos
    local = agent_world_R.inverse.rotate(delta)
    # local[0]=forward, local[1]=left → convert to x=right, y=forward
    return float(-local[1]), float(local[0])


def get_ego_past_waypoints(nusc, sample_token, n_past=4):
    """
    Compute up to n_past past positions of the ego vehicle in the current lidar frame.

    Mirrors check_cached_agent_hist.py method 2: applies the full global→ego→lidar
    transform chain using both ego pose and calibrated sensor (LIDAR_TOP) transforms.

    Returns a list of length n_past ordered oldest-first, where each element is either:
        (float x, float y)  - past ego position in lidar frame (x=fwd, y=left)
        None                - no previous sample exists (padded)
    """
    ego_t, ego_R, lidar_t, lidar_R = get_lidar_transforms(nusc, sample_token)

    waypoints = []
    cur = nusc.get("sample", sample_token)
    for _ in range(n_past):
        if cur["prev"] == "":
            waypoints.append(None)
            # No further prev frames will exist either, but keep padding loop
        else:
            cur = nusc.get("sample", cur["prev"])
            past_pose = get_ego_pose(nusc, cur["token"])
            past_global = np.array(past_pose["translation"])
            xy = world_to_lidar(past_global, ego_t, ego_R, lidar_t, lidar_R)
            waypoints.append((float(xy[0]), float(xy[1])))

    # Collected newest-first; reverse to oldest-first (same convention as future list)
    return list(reversed(waypoints))


def get_his_trajectory(data_dict: dict) -> str:
    """Extract historical ego trajectory string from a cached_nuscenes_info entry."""
    pts = data_dict["gt_ego_his_trajs"][:4]  # [4, 2], last 2 s at 0.5 s intervals
    return "[" + ",".join(f"({p[0]:.2f},{p[1]:.2f})" for p in pts) + "]"


def build_human_turn(his_msg: str) -> str:
    return (
        f"{DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
        f"Object-wise tracking information: "
        f"{DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        f"Ego Vehicle Token: {his_msg}\n"
        f"Please predict relative motion trajectory for the following object: "
        f"{DEFAULT_TRACK_START_TOKEN}{DEFAULT_OBJECT_TOKEN}{DEFAULT_TRACK_END_TOKEN}"
    )


def get_ego_pose(nusc, sample_token):
    """Return the ego_pose record for a sample (using LIDAR_TOP, same as object_relative_pose.py)."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    return nusc.get("ego_pose", sd["ego_pose_token"])


def get_future_waypoints(nusc, instance_token, sample_token, n_future=N_FUTURE):
    """
    Compute up to n_future future positions of an instance in the agent's own frame.

    The agent's position at sample_token is (0, 0). Output coordinate system:
        x = right, y = forward  (relative to the agent's heading at sample_token)

    Returns a list of length n_future where each element is either:
        (float x, float y)  - position in agent frame (x=right, y=forward)
        None                - no annotation exists (will be formatted as (UN, UN))

    Returns None if the instance has no annotation in sample_token (entry should be skipped).
    """
    # Build instance_token → annotation token map for this sample
    sample = nusc.get("sample", sample_token)
    ann_map = {
        nusc.get("sample_annotation", tok)["instance_token"]: tok
        for tok in sample["anns"]
    }

    if instance_token not in ann_map:
        return None  # instance not annotated in this sample – skip entry

    cur_ann = nusc.get("sample_annotation", ann_map[instance_token])
    agent_world_pos = np.array(cur_ann["translation"])
    agent_world_R = Quaternion(cur_ann["rotation"])

    waypoints = []
    next_tok = cur_ann["next"]

    for _ in range(n_future):
        if next_tok == "":
            waypoints.append(None)
        else:
            ann = nusc.get("sample_annotation", next_tok)
            xy = world_to_agent_frame(ann["translation"], agent_world_pos, agent_world_R)
            waypoints.append(xy)
            next_tok = ann["next"]

    return waypoints


def format_waypoints(waypoints):
    """Format waypoint list as a comma-separated string of (x, y) or (UN, UN)."""
    parts = []
    for wp in waypoints:
        if wp is None:
            parts.append("(UN, UN)")
        else:
            parts.append(f"({wp[0]:.2f}, {wp[1]:.2f})")
    return "[" + ", ".join(parts) + "]"


def convert(split, pth_dir, nusc, cached, debug=False):
    pth_split_dir = os.path.join(pth_dir, split)

    pth_files = sorted(fn for fn in os.listdir(pth_split_dir) if fn.endswith(".pth"))

    entries = []
    skipped_load_err  = 0
    skipped_no_cache  = 0
    skipped_no_ann    = 0
    skipped_all_pad   = 0

    for fn in tqdm(pth_files, desc=f"[stage2.5] {split}"):
        sample_token = fn.replace(".pth", "")
        pth_path = os.path.join(pth_split_dir, fn)

        try:
            data = torch.load(pth_path, map_location="cpu")
            track_dict = (
                data.get("result_track", {})
                    .get("track_gt_inds_to_embed_idx", {})
            )
        except Exception as e:
            print(f"Warning: failed to load {fn}: {e}")
            skipped_load_err += 1
            continue

        if not track_dict:
            continue

        if sample_token not in cached:
            skipped_no_cache += len(track_dict)
            continue

        his_msg = get_his_trajectory(cached[sample_token])

        if debug or DEBUG:
            ego_past = get_ego_past_waypoints(nusc, sample_token)
            print(
                f"[DEBUG] sample={sample_token}\n"
                f"        ego past waypoints (x=right, y=fwd, oldest→newest): "
                f"{format_waypoints(ego_past)}"
            )

        for raw_ind in track_dict.keys():
            instance_ind = int(raw_ind)
            actual_ind = (instance_ind - 1) if INS_INDS_ADD_1 else instance_ind

            instance_token = nusc.instance[actual_ind]["token"]

            waypoints = get_future_waypoints(nusc, instance_token, sample_token)

            if waypoints is None:
                # Instance not annotated in this sample – should not happen for tracked objects
                skipped_no_ann += 1
                continue

            if all(wp is None for wp in waypoints):
                # All future frames are missing – no useful trajectory signal
                skipped_all_pad += 1
                continue

            entries.append({
                #"id": f"{sample_token}_{instance_token}",
                "uniad_pth": os.path.join(pth_split_dir, f"{sample_token}.pth"),
                "instance_token": instance_token,
                "conversations": [
                    {"from": "human", "value": build_human_turn(his_msg)},
                    {"from": "gpt",   "value": format_waypoints(waypoints)},
                ],
            })
        
        if debug or DEBUG:
            break  # Process only one file in debug mode


    print(
        f"[stage2.5] {split}: {len(entries)} entries kept\n"
        f"           skipped - load error:    {skipped_load_err}\n"
        f"           skipped - no cache:      {skipped_no_cache}\n"
        f"           skipped - no annotation: {skipped_no_ann}\n"
        f"           skipped - all padded:    {skipped_all_pad}"
    )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output_dir", default="data/stage25")
    parser.add_argument("--pth_dir", default=FUSIONAD_PTH_DIR)
    parser.add_argument("--cached_pkl", default=CACHED_NUSCENES_PKL)
    parser.add_argument("--debug", action="store_true", help="Print ego past-waypoint diagnostics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading cached NuScenes info...")
    with open(args.cached_pkl, "rb") as f:
        cached = pickle.load(f)

    print("Loading NuScenes...")
    nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

    entries = convert(args.split, args.pth_dir, nusc, cached, debug=args.debug)

    out_path = os.path.join(args.output_dir, f"stage25_agent_rel_trajectory_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved → {out_path}  ({len(entries)} entries)")


if __name__ == "__main__":
    main()
