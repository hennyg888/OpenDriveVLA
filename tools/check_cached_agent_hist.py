"""
Check gt_agent_fut_trajs in cached_nuscenes_info.pkl and compare against
NuScenes API waypoints using the same global→ego→lidar transform as vis_object.py.

Both methods use lidar frame (x=fwd, y=left) for direct comparison.

Usage:
    python tools/check_cached_agent_hist.py
"""

import pickle
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

CACHED_PKL    = "/home/hhguo/OpenDriveVLA/data/nuscenes/cached_nuscenes_info.pkl"
NUSCENES_ROOT = "/home/hhguo/OpenDriveVLA/data/nuscenes"
NUSCENES_VER  = "v1.0-trainval"

# Fixed sample / object from vis_object.py
SAMPLE_TOKEN          = "54ff52cfb17742d3b393cd9e35293109"
OBJECT_INSTANCE_TOKEN = "f3bae5757789414dbc4607d2ce09d4b9"

N_FUTURE = 6  # max future waypoints to print


def get_lidar_transforms(nusc, sample_token):
    """
    Return (ego_t, ego_R, lidar_t, lidar_R) for LIDAR_TOP at sample_token.
    Mirrors vis_object.py's transform setup.
    """
    sample = nusc.get("sample", sample_token)
    sd     = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    cs       = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_t   = np.array(ego_pose["translation"])
    ego_R   = Quaternion(ego_pose["rotation"])
    lidar_t = np.array(cs["translation"])
    lidar_R = Quaternion(cs["rotation"])
    return ego_t, ego_R, lidar_t, lidar_R


def world_to_lidar(world_pos, ego_t, ego_R, lidar_t, lidar_R):
    """Global → ego → lidar (mirrors vis_object.py world_to_bev)."""
    p = ego_R.inverse.rotate(np.array(world_pos) - ego_t)
    p = lidar_R.inverse.rotate(p - lidar_t)
    return p[:2]


def get_future_waypoints_nusc(nusc, instance_token, sample_token, n_future=N_FUTURE):
    """
    Absolute future positions in lidar frame via NuScenes API.
    Returns list of length n_future: (x, y) in lidar frame or None.
    Returns None if instance not annotated in sample_token.
    """
    ego_t, ego_R, lidar_t, lidar_R = get_lidar_transforms(nusc, sample_token)

    sample = nusc.get("sample", sample_token)
    ann_map = {
        nusc.get("sample_annotation", tok)["instance_token"]: tok
        for tok in sample["anns"]
    }
    if instance_token not in ann_map:
        return None

    cur_ann  = nusc.get("sample_annotation", ann_map[instance_token])
    waypoints = []
    next_tok  = cur_ann["next"]
    for _ in range(n_future):
        if next_tok == "":
            waypoints.append(None)
        else:
            ann = nusc.get("sample_annotation", next_tok)
            xy  = world_to_lidar(ann["translation"], ego_t, ego_R, lidar_t, lidar_R)
            waypoints.append((float(xy[0]), float(xy[1])))
            next_tok = ann["next"]
    return waypoints


def format_waypoints(waypoints):
    parts = []
    for wp in waypoints:
        parts.append("(UN, UN)" if wp is None else f"({wp[0]:.2f}, {wp[1]:.2f})")
    return "[" + ", ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("Loading cached pkl …")
with open(CACHED_PKL, "rb") as f:
    cached = pickle.load(f)

print("Loading NuScenes …")
nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

entry = cached[SAMPLE_TOKEN]

# annotation tokens ordered same as gt_agent_fut_trajs rows (= sample['anns'])
gt_tokens        = nusc.get("sample", SAMPLE_TOKEN)["anns"]
fut_trajs_flat   = entry["gt_agent_fut_trajs"]  # [num_agents, fut_ts*2] per-step offsets, lidar frame
fut_masks        = entry["gt_agent_fut_masks"]  # [num_agents, fut_ts]
agent_lcf_feat   = entry["gt_agent_lcf_feat"]   # [num_agents, 9]; [:, :2] = agent pos in lidar frame at t=0
num_agents, flat = fut_trajs_flat.shape
fut_ts           = flat // 2

print(f"\nSample : {SAMPLE_TOKEN}")
print(f"Agents in sample: {num_agents}  |  fut_ts={fut_ts}")

# ── Find the agent index for our object ──────────────────────────────────────
target_ann_token = None
for tok in gt_tokens:
    ann = nusc.get("sample_annotation", tok)
    if ann["instance_token"] == OBJECT_INSTANCE_TOKEN:
        target_ann_token = tok
        break

if target_ann_token is None:
    raise ValueError(f"Instance {OBJECT_INSTANCE_TOKEN} not found in sample annotations.")

agent_idx = list(gt_tokens).index(target_ann_token)
print(f"Object instance : {OBJECT_INSTANCE_TOKEN}")
print(f"Annotation token: {target_ann_token}")
print(f"Agent index     : {agent_idx}")

# ── METHOD 1: cached gt_agent_fut_trajs (per-step offsets, lidar frame) ──────
print("\n" + "="*60)
print("METHOD 1 — cached gt_agent_fut_trajs")
print("  Frame : lidar (x=fwd, y=left)")
print("  Format: per-step offsets → cumsum gives absolute displacement")
print("="*60)

offsets = fut_trajs_flat[agent_idx].reshape(fut_ts, 2)  # [fut_ts, 2]
masks   = fut_masks[agent_idx]                           # [fut_ts]

print(f"\n  Per-step offsets (Δx, Δy) with mask:")
for j in range(min(N_FUTURE, fut_ts)):
    m = masks[j]
    print(f"    t+{j+1}: ({offsets[j,0]:7.3f}, {offsets[j,1]:7.3f})  mask={m:.0f}")

# agent's t=0 position in lidar frame; adding it shifts cumsum to ego-relative coords
agent_t0 = agent_lcf_feat[agent_idx, :2]  # (x, y) in lidar frame
abs_positions = agent_t0 + np.cumsum(offsets, axis=0)
print(f"\n  Agent t=0 lidar pos: ({agent_t0[0]:.3f}, {agent_t0[1]:.3f})")
print(f"\n  Ego-relative absolute positions (lidar frame, x=fwd, y=left):")
waypoints_cached = []
for j in range(min(N_FUTURE, fut_ts)):
    m = masks[j]
    if m > 0:
        wp = (abs_positions[j, 0], abs_positions[j, 1])
        waypoints_cached.append(wp)
        print(f"    t+{j+1}: ({wp[0]:7.3f}, {wp[1]:7.3f})")
    else:
        waypoints_cached.append(None)
        print(f"    t+{j+1}: (UN, UN)  [masked]")

print(f"\n  Formatted: {format_waypoints(waypoints_cached)}")

# ── METHOD 2: NuScenes API, global→ego→lidar (same as vis_object.py) ─────────
print("\n" + "="*60)
print("METHOD 2 — NuScenes API (global → ego → lidar, x=fwd, y=left)")
print("  Frame : lidar (x=fwd, y=left) — absolute positions")
print("="*60)

waypoints_nusc = get_future_waypoints_nusc(nusc, OBJECT_INSTANCE_TOKEN, SAMPLE_TOKEN)
if waypoints_nusc is None:
    print("  Instance not annotated in this sample — no waypoints.")
else:
    for j, wp in enumerate(waypoints_nusc[:N_FUTURE]):
        if wp is None:
            print(f"    t+{j+1}: (UN, UN)")
        else:
            print(f"    t+{j+1}: ({wp[0]:7.3f}, {wp[1]:7.3f})")
    print(f"\n  Formatted: {format_waypoints(waypoints_nusc[:N_FUTURE])}")

# ── All agents summary ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL AGENTS — cached gt_agent_fut_trajs (absolute displacement)")
print("="*60)
for i, ann_tok in enumerate(gt_tokens):
    ann = nusc.get("sample_annotation", ann_tok)
    offs = fut_trajs_flat[i].reshape(fut_ts, 2)
    msk  = fut_masks[i]
    abs_pos = agent_lcf_feat[i, :2] + np.cumsum(offs, axis=0)
    wps = []
    for j in range(min(N_FUTURE, fut_ts)):
        wps.append((abs_pos[j,0], abs_pos[j,1]) if msk[j] > 0 else None)
    marker = " ◄ TARGET" if ann_tok == target_ann_token else ""
    print(f"  [{i:2d}] {ann['category_name']:30s}  {format_waypoints(wps)}{marker}")
