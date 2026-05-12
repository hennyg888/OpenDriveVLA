"""
FusionAD E2E converter for Stage 3 training data.

Mirrors uniad_results_converter.py but gets planning_gt and img_metas
from NuScenes directly, because FusionAD pth files only contain
['result_track', 'result_seg', 'sample_token', 'scene_token'].

Outputs:
    data/fusionad_results_for_vlm/train.json
    data/fusionad_results_for_vlm/val.json

Usage:
    cd /home/s56cai/OpenDriveVLA
    python drivevla/data_converter/fusionad_converter/fusionad_e2e_converter.py --split train
    python drivevla/data_converter/fusionad_converter/fusionad_e2e_converter.py --split val
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

FUSIONAD_PTH_DIR = "data/fusionad_results_for_vlm"
NUSCENES_ROOT    = "data/nuscenes"
NUSCENES_VER     = "v1.0-trainval"
PKL_PATHS = {
    "train": "data/nuscenes/nuscenes_infos_temporal_train.pkl",
    "val":   "data/nuscenes/nuscenes_infos_temporal_val.pkl",
}
CACHED_NUSCENES_PKL = "data/nuscenes/cached_nuscenes_info.pkl"
PLANNING_STEPS = 6
COMMAND_LIST   = ["turn right", "turn left", "keep forward"]


# ---------------------------------------------------------------------------
# NuScenes helpers
# ---------------------------------------------------------------------------

def load_pkl_as_dict(pkl_path):
    """Load temporal pkl and return {sample_token: info}."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    return {info["token"]: info for info in infos}


def _get_transforms(lidar2ego_t, lidar2ego_r, ego2global_t, ego2global_r):
    """Return (l2e_t, l2e_R, e2g_t, e2g_R) as numpy arrays."""
    return (
        np.array(lidar2ego_t),
        Quaternion(lidar2ego_r).rotation_matrix,
        np.array(ego2global_t),
        Quaternion(ego2global_r).rotation_matrix,
    )


def compute_planning(nusc, info):
    """
    Compute ego future trajectory in initial lidar frame.

    Mirrors NuScenesTraj.get_sdc_planning_label() from
    projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py
    without requiring LiDARInstance3DBoxes.

    The SDC center is the lidar origin (0,0,0) in its own frame.
    We chain lidar_k → ego_k → global → init_ego → init_lidar to get
    the relative displacement at each future timestep.

    Returns
    -------
    planning_xy : np.ndarray, shape (PLANNING_STEPS, 2)
        (x, y) waypoints in initial lidar frame; zero-padded for missing steps.
    command : int
        0 = turn right, 1 = turn left, 2 = keep forward
    """
    l2e_t_init, l2e_R_init, e2g_t_init, e2g_R_init = _get_transforms(
        info["lidar2ego_translation"],
        info["lidar2ego_rotation"],
        info["ego2global_translation"],
        info["ego2global_rotation"],
    )

    waypoints = []
    sample = nusc.get("sample", info["token"])

    for _ in range(PLANNING_STEPS):
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])

        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])

        l2e_t_curr, l2e_R_curr, e2g_t_curr, e2g_R_curr = _get_transforms(
            cs["translation"], cs["rotation"],
            ep["translation"], ep["rotation"],
        )

        # SDC center = (0,0,0) in lidar_curr frame
        p = np.zeros(3)
        p = l2e_R_curr @ p + l2e_t_curr                    # → ego_curr
        p = e2g_R_curr @ p + e2g_t_curr                    # → global
        p = np.linalg.inv(e2g_R_init) @ (p - e2g_t_init)  # → init_ego
        p = np.linalg.inv(l2e_R_init) @ (p - l2e_t_init)  # → init_lidar
        waypoints.append(p[:2])

    # Zero-pad to PLANNING_STEPS
    planning_xy = np.zeros((PLANNING_STEPS, 2))
    for i, wp in enumerate(waypoints):
        planning_xy[i] = wp

    # Command: based on final valid x displacement (same logic as trajectory_api.py)
    if not waypoints:
        command = 2  # FORWARD
    elif waypoints[-1][0] >= 2:
        command = 0  # RIGHT
    elif waypoints[-1][0] <= -2:
        command = 1  # LEFT
    else:
        command = 2  # FORWARD

    return planning_xy, command


def format_ego_info(can_bus_data):
    """Mirror uniad_results_converter.py can_bus formatting."""
    if np.all(can_bus_data == 0):
        return "No CAN bus data available."
    return (
        f"acceleration: ({can_bus_data[7]:.4f},{can_bus_data[8]:.4f},{can_bus_data[9]:.4f}) m/s^2, "
        f"rotation_rate: ({can_bus_data[10]:.4f},{can_bus_data[11]:.4f},{can_bus_data[12]:.4f}) rad/s, "
        f"velocity: ({can_bus_data[13]:.4f},{can_bus_data[14]:.4f},{can_bus_data[15]:.4f}) m/s"
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(split, keep_end_of_scene=False):
    print(f"Loading NuScenes ({NUSCENES_VER}) ...")
    nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

    print(f"Loading pkl: {PKL_PATHS[split]}")
    token_to_info = load_pkl_as_dict(PKL_PATHS[split])
    print(f"  {len(token_to_info)} samples in pkl")

    print(f"Loading cached ego masks: {CACHED_NUSCENES_PKL}")
    with open(CACHED_NUSCENES_PKL, "rb") as f:
        cached_nuscenes_data = pickle.load(f)
    print(f"  {len(cached_nuscenes_data)} tokens in cached info")

    # Only keep samples that actually have a pth file
    pth_dir = Path(FUSIONAD_PTH_DIR) / split
    available_tokens = {f.stem for f in pth_dir.glob("*.pth")}
    print(f"  {len(available_tokens)} pth files found in {pth_dir}")
    print(f"  keep_end_of_scene: {keep_end_of_scene}")

    results = []
    skipped_no_pth          = 0
    skipped_invalid_future  = 0

    for sample_token, info in tqdm(token_to_info.items(), desc=f"Converting {split}"):
        if sample_token not in available_tokens:
            skipped_no_pth += 1
            continue

        # Skip samples whose 6-step future walks past end of scene.
        # gt_ego_fut_masks is shape (6,); any 0 means that future step is
        # zero-padded (scene ended early) and would poison training.
        # If keep_end_of_scene=True, these samples are kept (zero-padded future).
        if not keep_end_of_scene:
            cached = cached_nuscenes_data.get(sample_token)
            if cached is None or not np.all(cached["gt_ego_fut_masks"] == 1):
                skipped_invalid_future += 1
                continue

        # ---- planning trajectory + command ----
        planning_xy, command = compute_planning(nusc, info)
        planning_str = "[" + ",".join(
            f"({x[0]:.4f}, {x[1]:.4f})" for x in planning_xy
        ) + "]"
        high_level_command = COMMAND_LIST[command]

        # ---- ego info from can_bus ----
        ego_info = format_ego_info(np.array(info["can_bus"]))

        # ---- build conversation (same format as uniad_results_converter.py) ----
        human_prompt = (
            f"\nFollowing is scene information: <scene_start><SCENE><scene_end>\n"
            f"Following is agent-wise tracking information: <track_start><TRACK><track_end>\n"
            f"Following is map information: <map_start><MAP><map_end>\n"
            f"Following is the ego information: {ego_info}\n"
            f"Following is the planning command: {high_level_command}\n"
            f"Predict the ego vehicle's planning trajectory.\n"
        )

        results.append({
            "id": sample_token,
            "uniad_pth": str(pth_dir / f"{sample_token}.pth"),
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt",   "value": f"The ego vehicle's planning trajectory is {planning_str}."},
            ],
        })

    output_path = Path(FUSIONAD_PTH_DIR) / f"{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] {len(results)} entries written → {output_path}")
    print(f"       skipped (no pth):         {skipped_no_pth}")
    print(f"       skipped (invalid future): {skipped_invalid_future}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument(
        "--keep-end-of-scene",
        action="store_true",
        help=(
            "Keep samples whose 6-step ego future walks past the end of scene "
            "(future steps zero-padded). Default: skip them."
        ),
    )
    args = parser.parse_args()
    convert(args.split, keep_end_of_scene=args.keep_end_of_scene)
