#!/usr/bin/env python3
import json
import re
import pickle
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from llava.constants import (
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRACK_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_EGO_START_TOKEN,
    DEFAULT_EGO_END_TOKEN,
    DEFAULT_COMMAND_START_TOKEN,
    DEFAULT_COMMAND_END_TOKEN,
    DEFAULT_QUESTION_START,
    DEFAULT_QUESTION_END,
    DEFAULT_ANSWER_START,
    DEFAULT_ANSWER_END,
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_TRAJ_TOKEN
)

def get_sample_split(nusc: NuScenes, sample_token: str) -> str:
    """
    Determine the split of a sample given its token.
    
    Parameters:
        nusc: An instance of the NuScenes dataset.
        sample_token: The token of the sample (string).
    
    Returns:
        The split the sample belongs to, e.g., 'train', 'val', or 'test'.
    """
    sample_record = nusc.get("sample", sample_token)
    scene_token = sample_record["scene_token"]
    scene_record = nusc.get("scene", scene_token)
    scene_name = scene_record["name"]
    splits = create_splits_scenes()
    for split_name, scene_tokens in splits.items():
        if scene_name in scene_tokens:
            return split_name
    return "unknown"

def generate_user_message(data_dict):
        
    """
    Ego-States:
        gt_ego_lcf_feat: [vx, vy, ?, ?, v_yaw (rad/s), ego_length, ego_width, v0 (vy from canbus), Kappa (steering)]
    """
    ego_message = ""
    vx = data_dict['gt_ego_lcf_feat'][0]*0.5
    vy = data_dict['gt_ego_lcf_feat'][1]*0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    cx = data_dict['gt_ego_lcf_feat'][2]
    cy = data_dict['gt_ego_lcf_feat'][3]
    vhead = data_dict['gt_ego_lcf_feat'][7]*0.5
    steeling = data_dict['gt_ego_lcf_feat'][8]
    # ego_message += f"Ego states:"
    ego_message += f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
    ego_message += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
    ego_message += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
    ego_message += f" - Can Bus: ({cx:.2f},{cy:.2f})"
    ego_message += f" - Heading Speed: ({vhead:.2f})"
    ego_message += f" - Steering: ({steeling:.2f})"

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    # his_message = ""
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    # his_message += f"Historical trajectory (last 2 seconds):"
    his_message = f"[({xh1:.2f},{yh1:.2f}),({xh2:.2f},{yh2:.2f}),({xh3:.2f},{yh3:.2f}),({xh4:.2f},{yh4:.2f})]"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    # cmd_message = ""
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "turn right"
    elif left > 0:
        mission_goal = "turn left"
    else:
        assert forward > 0
        mission_goal = "keep forward"
    # cmd_message += f"Mission Goal: "
    cmd_message = f"{mission_goal}"
    
    """
    Planning trajectory:
        gt_ego_fut_trajs: [6, 2] last 6 seconds
    """
    # x1 = data_dict['gt_ego_fut_trajs'][1][0]
    # x2 = data_dict['gt_ego_fut_trajs'][2][0]
    # x3 = data_dict['gt_ego_fut_trajs'][3][0]
    # x4 = data_dict['gt_ego_fut_trajs'][4][0]
    # x5 = data_dict['gt_ego_fut_trajs'][5][0]
    # x6 = data_dict['gt_ego_fut_trajs'][6][0]
    # y1 = data_dict['gt_ego_fut_trajs'][1][1]
    # y2 = data_dict['gt_ego_fut_trajs'][2][1]
    # y3 = data_dict['gt_ego_fut_trajs'][3][1]
    # y4 = data_dict['gt_ego_fut_trajs'][4][1]
    # y5 = data_dict['gt_ego_fut_trajs'][5][1]
    # y6 = data_dict['gt_ego_fut_trajs'][6][1]
    
    x1 = data_dict['gt_ego_fut_diff'][0][0]
    y1 = data_dict['gt_ego_fut_diff'][0][1]
    x2 = data_dict['gt_ego_fut_diff'][1][0]
    y2 = data_dict['gt_ego_fut_diff'][1][1]
    x3 = data_dict['gt_ego_fut_diff'][2][0]
    y3 = data_dict['gt_ego_fut_diff'][2][1]
    x4 = data_dict['gt_ego_fut_diff'][3][0]
    y4 = data_dict['gt_ego_fut_diff'][3][1]
    x5 = data_dict['gt_ego_fut_diff'][4][0]
    y5 = data_dict['gt_ego_fut_diff'][4][1]
    x6 = data_dict['gt_ego_fut_diff'][5][0]
    y6 = data_dict['gt_ego_fut_diff'][5][1]

    traj_message = f"[({x1:.2f},{y1:.2f}),({x2:.2f},{y2:.2f}),({x3:.2f},{y3:.2f}),({x4:.2f},{y4:.2f}),({x5:.2f},{y5:.2f}),({x6:.2f},{y6:.2f})]"

    return ego_message, his_message, cmd_message, traj_message


def process_traj_data(data, split, nusc):

    # id_to_info = process_train_data(input_path)
    converted_entries = []
    
    for sample_token, value in data.items():
        
        if get_sample_split(nusc, sample_token) != split:
            continue

        ego_message, his_message, cmd_message, traj_message = generate_user_message(value)
        # command = info["planning_command"]
        # acceleration = info["acceleration"]
        # rotation_rate = info["rotation_rate"]
        # velocity = info["velocity"]
        # coords_string = info["trajectory"]

        unidata_path = f"data/uniad_results_for_vlm/{split}/{sample_token}.pth"
        
        converted_entry = {
            "qa_id": sample_token+'_trajectory',
            "sample_id": sample_token,
            "uniad_pth": unidata_path,
            "conversations": [
                {
                    "from": "human",
                    # "value": (
                    #     f"\n{DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}"
                    #     f"\n{DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}"
                    #     f"\n{DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}"
                    #     f"\n{DEFAULT_EGO_START_TOKEN}"
                    #     # f"acceleration:({acceleration[0]:.1f},{acceleration[1]:.1f})m/s^2, "
                    #     # f"rotation_rate:({rotation_rate[0]:.1f},{rotation_rate[1]:.1f})rad/s, "
                    #     # f"velocity:({velocity[0]:.1f},{velocity[1]:.1f})m/s"
                    #     # f"acceleration:({acceleration[0]:.2f},{acceleration[1]:.2f})m/s^2, "
                    #     # f"rotation_rate:({rotation_rate[0]:.2f},{rotation_rate[1]:.2f})rad/s, "
                    #     # f"velocity:({velocity[0]:.2f},{velocity[1]:.2f})m/s"
                    #     f"{DEFAULT_EGO_END_TOKEN}"
                    #     f"\n{DEFAULT_COMMAND_START_TOKEN}{command}{DEFAULT_COMMAND_END_TOKEN}"
                    #     f"\n{DEFAULT_TRAJ_TOKEN}"
                    # )
                    "value": (
                        f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
                        f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
                        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
                        f"Ego states: {ego_message}\n"
                        f"Historical trajectory (last 2 seconds): {his_message}\n"
                        f"Mission goal: {cmd_message}\n"
                        # f"Plan trajectory: {DEFAULT_TRAJ_TOKEN}"
                        f"Please plan relative motion trajectory for ego vehicle: <trajectory>"

                        
                        # f"\nFollowing is scene information: {DEFAULT_SCENE_TOKEN}"
                        # f"\nFollowing is object-wise tracking information: {DEFAULT_TRACK_TOKEN}"
                        # f"\nFollowing is map information: {DEFAULT_MAP_TOKEN}"
                        # f"\nFollowing is ego states:"
                        # f"acceleration:({acceleration[0]:.1f},{acceleration[1]:.1f})m/s^2, "
                        # f"rotation_rate:({rotation_rate[0]:.1f},{rotation_rate[1]:.1f})rad/s, "
                        # f"velocity:({velocity[0]:.1f},{velocity[1]:.1f})m/s"
                        # f"acceleration:({acceleration[0]:.2f},{acceleration[1]:.2f})m/s^2, "
                        # f"rotation_rate:({rotation_rate[0]:.2f},{rotation_rate[1]:.2f})rad/s, "
                        # f"velocity:({velocity[0]:.2f},{velocity[1]:.2f})m/s"
                        # f"{DEFAULT_EGO_END_TOKEN}"
                        # f"\nFollowing is the planning command: {command}"
                        # f"\n{DEFAULT_COMMAND_START_TOKEN}{command}{DEFAULT_COMMAND_END_TOKEN}"
                        # f"\n{DEFAULT_TRAJ_TOKEN}"
                    )
                },
                {
                    "from": "gpt",
                    "value": f"{DEFAULT_TRAJ_START_TOKEN}{traj_message}{DEFAULT_TRAJ_END_TOKEN}"
                }
            ]
        }
        
        converted_entries.append(converted_entry)

    return converted_entries


data = pickle.load(open('/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/data/nuscenes_full/cached_nuscenes_info.pkl', 'rb'))
nusc = NuScenes(version='v1.0-trainval', dataroot='/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/data/nuscenes_full', verbose=True)
    
# OUTPUT_DIR = 'data/trajectory_experiments_25Feb27/Exp1_scene_tracking_map_ego_history_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp2_tracking_map_ego_history_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp3_scene_map_ego_history_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp4_scene_tracking_ego_history_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp5_scene_tracking_map_history_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp6_scene_tracking_map_ego_command'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp7_scene_tracking_map_ego_history'
# OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp8_ego_history_command'

OUTPUT_DIR = 'data/DriveVLA/experiments/trajectory_experiments_diff_25mar01/Exp1_scene_tracking_map_ego_history_command'

# OUTPUT_DIR = 'data/DriveVLA/Exp2_scene_tracking_map_nospace_withbracket_2decimals'
# OUTPUT_DIR = 'data/DriveVLA/Exp3_scene_map_command_nospace_withbracket_2decimals'
# OUTPUT_DIR = 'data/DriveVLA/Exp4_tracking_map_command_nospace_withbracket_2decimals'
# OUTPUT_DIR = 'data/DriveVLA/Exp5_scene_tracking_map_noego_command_nospace_withbracket_2decimals'

# OUTPUT_DIR = 'data/DriveVLA/Exp6_scene_tracking_map_command_nospace_withoutbracket_2decimals'
# OUTPUT_DIR = 'data/DriveVLA/Exp7_scene_tracking_map_command_nospace_withbracket_1decimals'  
# OUTPUT_DIR = 'data/DriveVLA/Exp8_scene_tracking_map_command_nospace_withoutbracket_1decimals'  

os.makedirs(OUTPUT_DIR, exist_ok=True)

split = "train"  # Change to "val" if processing validation data
converted_entries = process_traj_data(data, split, nusc)

output_file = f'{OUTPUT_DIR}/traj_converted_data_{split}.json'
with open(output_file, "w") as json_file:
    json_file.write("[\n")
    # Use the converted_entries as the buffer
    buffer = converted_entries  
    total_entries = len(buffer)
    
    for i, item in enumerate(buffer):
        if i > 0:
            json_file.write(",\n")
        json.dump(item, json_file, indent=2)
    json_file.write("\n]")

print(f"Converted entries have been saved to {output_file}")

# ------------------------------------------------------------------------------------------------ 

split = "val"  # Change to "val" if processing validation data


converted_entries = process_traj_data(data, split, nusc)

output_file = f'{OUTPUT_DIR}/traj_converted_data_{split}.json'
with open(output_file, "w") as json_file:
    json_file.write("[\n")
    # Use the converted_entries as the buffer
    buffer = converted_entries  
    total_entries = len(buffer)
    
    for i, item in enumerate(buffer):
        if i > 0:
            json_file.write(",\n")
        json.dump(item, json_file, indent=2)
    json_file.write("\n]")

print(f"Converted entries have been saved to {output_file}")


