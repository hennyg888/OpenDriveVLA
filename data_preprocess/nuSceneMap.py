import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple
import time

from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.nuscenes import NuScenes

STAGE1_PATH = "/home/s56cai/OpenDriveVLA/data/nuCaption/stage1_scene_data.json"
LAYER_COUNT_PATH = "/home/hhguo/OpenDriveVLA/data/nuScenesMap/layer_count_data.jsonl"
OUT_PATH = "/home/hhguo/OpenDriveVLA/data/nuScenesMap/stage1_map_data.jsonl"
RADIUS = 54  # meters, from /home/hhguo/OpenDriveVLA/projects/configs/bevfusion_track_map/bevfusion.py
BOUNDARY_MODE = 'within' # 'intersect' or 'within'
USER_PROMPT = "Please provide a caption for the following map:<map_start><MAP><map_end>"

def collect_stage1_sample_tokens(stage1_data) -> set:
    tokens = set()

    def add_from_obj(obj):
        if isinstance(obj, dict):
            if "sample_token" in obj and isinstance(obj["sample_token"], str):
                tokens.add(obj["sample_token"])
            for v in obj.values():
                add_from_obj(v)
        elif isinstance(obj, list):
            for v in obj:
                add_from_obj(v)

    add_from_obj(stage1_data)
    return tokens

def read_json_or_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in ("[", "{"):
        return json.loads(text)

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def main():
    print(f"Loading stage1: {STAGE1_PATH}")
    stage1 = read_json_or_jsonl(STAGE1_PATH)
    stage1_tokens = collect_stage1_sample_tokens(stage1)

    out_p = Path(OUT_PATH)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    layer_p = Path(LAYER_COUNT_PATH)
    layer_p.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_bad_token = 0
    skipped_no_caption = 0

    t_start = time.perf_counter()
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/s56cai/OpenDriveVLA/data/nuscenes', verbose=False)
    nusc_maps = {}
    locations = sorted({log['location'] for log in nusc.log})
    #['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    for loc in locations:
        nusc_maps[loc] = NuScenesMap(dataroot='/home/s56cai/OpenDriveVLA/data/nuscenes', map_name=loc)
    t_end = time.perf_counter()
    print(f"Loaded NuScenes and NuScenesMap in {t_end - t_start:.2f} seconds.")
    # records_intersect_patch_time = 0
    # sample_get_time = 0
    # captioning_time = 0
    with (
        out_p.open("w", encoding="utf-8") as f,
        layer_p.open("w", encoding="utf-8") as f_layer,
    ):
        counter = 0
        for sample_token in stage1_tokens:
            counter += 1
            if counter > 500:
                break

            #start_time = time.perf_counter()
            try:
                sample = nusc.get('sample', sample_token)
            except Exception:
                skipped_bad_token += 1
                continue
            #end_time = time.perf_counter()
            #sample_get_time += end_time - start_time

            scene = nusc.get('scene', sample['scene_token'])
            log = nusc.get('log', scene['log_token'])
            map_name = log['location']   # e.g. 'boston-seaport'
            this_map = nusc_maps[map_name]

            sd_token = sample['data']['LIDAR_TOP']
            sd = nusc.get('sample_data', sd_token)

            ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
            x, y, z = ego_pose['translation']
            #print(f"Ego pose translation: x={x}, y={y}, z={z}")

            patch = (
                x - RADIUS,
                y - RADIUS,
                x + RADIUS,
                y + RADIUS
            )

            query_layers = [#'drivable_area',   #too vague, ignored
                            'road_segment',     #only captioning intersections
                            #'road_block',      #ignored, lanes more precise
                            'lane',
                            'ped_crossing',
                            'walkway',
                            'stop_line',
                            'carpark_area',
                            'road_divider',
                            'lane_divider'
                            #'traffic_light'    #ignored, not counting traffic lights
                            ]

            #start_time = time.perf_counter()
            records_intersect_patch = this_map.get_records_in_patch(patch, query_layers, mode=BOUNDARY_MODE)
            #end_time = time.perf_counter()
            #records_intersect_patch_time += end_time - start_time
            
            map_caption = ""
            #start_time = time.perf_counter()
            intersection_count = 0
            for seg_tok in records_intersect_patch['road_segment']:
                segment = this_map.get('road_segment', seg_tok)
                if segment['is_intersection']:
                    intersection_count += 1
            if intersection_count:
                map_caption += f"There {'are' if intersection_count > 1 else 'is'} {intersection_count} road intersection{'s' if intersection_count > 1 else ''}.\n"
            
            car_lane_count = len(records_intersect_patch['lane'])
            if car_lane_count:
                map_caption += f"There {'are' if car_lane_count > 1 else 'is'} {car_lane_count} car lane{'s' if car_lane_count > 1 else ''}.\n"
            
            ped_crossing_count = len(records_intersect_patch['ped_crossing'])
            if ped_crossing_count:
                map_caption += f"There {'are' if ped_crossing_count > 1 else 'is'} {ped_crossing_count} pedestrian crosswalk{'s' if ped_crossing_count > 1 else ''} marked with white stripes.\n"
            
            walkway_count = len(records_intersect_patch['walkway'])
            if walkway_count:
                map_caption += f"There {'are' if walkway_count > 1 else 'is'} {walkway_count} walkway{'s' if walkway_count > 1 else ''} for pedestrians.\n"
            
            stop_line_count = len(records_intersect_patch['stop_line'])
            if stop_line_count:
                map_caption += f"There {'are' if stop_line_count > 1 else 'is'} {stop_line_count} stop line{'s' if stop_line_count > 1 else ''} indicating where vehicles must stop.\n"
            
            carpark_area_count = len(records_intersect_patch['carpark_area'])
            if carpark_area_count:
                map_caption += f"There {'are' if carpark_area_count > 1 else 'is'} {carpark_area_count} carpark area{'s' if carpark_area_count > 1 else ''} for vehicle parking.\n"
            
            road_divider_count = len(records_intersect_patch['road_divider'])
            if road_divider_count:
                map_caption += f"There {'are' if road_divider_count > 1 else 'is'} {road_divider_count} physical road divider{'s' if road_divider_count > 1 else ''} separating lanes of traffic.\n"
            
            lane_divider_count = len(records_intersect_patch['lane_divider'])
            if lane_divider_count:
                map_caption += f"There {'are' if lane_divider_count > 1 else 'is'} {lane_divider_count} lane marker{'s' if lane_divider_count > 1 else ''} separating lanes of traffic.\n"
            
            map_elem_count = (
                intersection_count +
                car_lane_count +
                ped_crossing_count +
                walkway_count +
                stop_line_count +
                carpark_area_count +
                road_divider_count +
                lane_divider_count
            )

            if map_elem_count == 0:
                skipped_no_caption += 1
                continue

            map_caption = f"There {'are' if map_elem_count > 1 else 'is'} {map_elem_count} map element{'s' if map_elem_count > 1 else ''} in total.\n" + map_caption
            #end_time = time.perf_counter()
            #captioning_time += end_time - start_time

            out_item = {
                "sample_id": sample_token,
                "conversation": [
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": map_caption},
                ],
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1

            layer_count_item = {"sample_id": sample_token,
                                "intersection_count": intersection_count,
                                "car_lane_count": car_lane_count,
                                "ped_crossing_count": ped_crossing_count,
                                "walkway_count": walkway_count,
                                "stop_line_count": stop_line_count,
                                "carpark_area_count": carpark_area_count,
                                "road_divider_count": road_divider_count,
                                "lane_divider_count": lane_divider_count}
            f_layer.write(json.dumps(layer_count_item, ensure_ascii=False) + "\n")
    
    print("Done.")
    print(f"Output: {OUT_PATH}")
    print(f"kept={kept}")
    print(f"skipped_bad_token={skipped_bad_token}")
    print(f"skipped_no_caption={skipped_no_caption}")
    # print(f"Total time in get_records_in_patch: {records_intersect_patch_time:.2f} seconds.")
    # print(f"Total time in sample get: {sample_get_time:.2f} seconds.")
    # print(f"Total time in captioning: {captioning_time:.2f} seconds.")
    print(f"Total time: {time.perf_counter() - t_start:.2f} seconds.")

if __name__ == "__main__":
    main()