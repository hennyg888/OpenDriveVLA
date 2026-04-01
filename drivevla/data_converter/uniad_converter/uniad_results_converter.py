import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib
matplotlib.use('Agg')

class UniadResultsConverter:
    def __init__(self, args):
        """Initialize the converter with input and output paths.
        
        Args:
            input_dir: Directory containing .pth files
            output_path: Path to save the output json file
        """
        self.input_dir = Path(f"data/uniad_results_for_vlm/{args.nuscenes_split}")
        self.output_path = Path(f"data/uniad_results_for_vlm/{args.nuscenes_split}.json")
        # self.nusc = NuScenes(version=args.nuscenes_version, dataroot='data/nuscenes', verbose=True)
        
    def get_pth_files(self) -> List[Path]:
        """Get list of all .pth files from input directory."""
        return list(self.input_dir.glob("*.pth"))
    
    def convert_to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data to json format."""
        # 1. Extract relevant information from data
        scene_token = data['scene_token']
        sample_token = data['sample_token']
        img_metas = data['img_metas']
        result_track = data['result_track']
        result_seg = data['result_seg']
        planning_gt = data['planning_gt']

        # nuscenes
        # sample = self.nusc.get('sample', sample_token)
        # scene_token = sample['scene_token']
        # scene = self.nusc.get('scene', scene_token)

        # 2. Retrieve image filenames
        img_filenames = img_metas['filename']
        # Fix the image path difference between the nuscenes v1.0-trainval and nuscenes v1.0-mini
        img_filenames = [os.path.join('./data/nuscenes/', img_filename) if img_filename.startswith('samples') else img_filename for img_filename in img_filenames]

        # 3. Convert planning_gt_2d to string
        planning_gt_2d = planning_gt['sdc_planning'][0,:, :6, :2].squeeze().tolist()
        # convert planning_gt_2d to string
        planning_gt_2d_str = "[" + ",".join(f"({x[0]:.4f}, {x[1]:.4f})" for x in planning_gt_2d) + "]"
        
        # 4. Convert ego_info to string
        """
        img_metas['can_bus']: (18,)
        [0] - x: x coordinate in global coordinate system meters
        [1] - y: y coordinate in global coordinate system meters
        [2] - z: z coordinate in global coordinate system meters
        [3:7] - quaternion: rotation represented by quaternion (w,x,y,z)
        [7] - acceleration x accel (ego vehicle frame) m/s^2
        [8] - acceleration y accel (ego vehicle frame) m/s^2
        [9] - acceleration z accel (ego vehicle frame) m/s^2
        [10] - angular velocity x rotation_rate (ego vehicle frame) rad/s
        [11] - angular velocity y rotation_rate (ego vehicle frame) rad/s
        [12] - angular velocity z rotation_rate (ego vehicle frame) rad/s
        [13] - velocity x vel (ego vehicle frame) m/s
        [14] - velocity y vel (ego vehicle frame) m/s
        [15] - velocity z vel (ego vehicle frame) m/s
        [16] - yaw angle in radians
        [17] - yaw angle in degrees

        nuscenes ego vehicle frame:
        x-axis: Points forward (direction of vehicle movement)
        y-axis: Points to the right of the vehicle
        z-axis: Points downward

        uniad bev frame:
        x-axis: Points to the right of the vehicle
        y-axis: Points forward (direction of vehicle movement)
        z-axis: Points upward

        yaw angle:

            North/X-axis (0°)
                ^
                |
                |
        West    |    East
        ←-------+-------→
                |
                |
                v
            South (180°)
            
        0 degrees: Vehicle facing north
        90 degrees: Vehicle facing east
        180 degrees: Vehicle facing south
        270 degrees: Vehicle facing west
        """
        can_bus_data = img_metas['can_bus']
        if np.all(can_bus_data == 0):
            ego_info = f"No CAN bus data available."
        else:
            ego_info = f"acceleration: ({can_bus_data[7]:.4f},{can_bus_data[8]:.4f},{can_bus_data[9]:.4f}) m/s^2, rotation_rate: ({can_bus_data[10]:.4f},{can_bus_data[11]:.4f},{can_bus_data[12]:.4f}) rad/s, velocity: ({can_bus_data[13]:.4f},{can_bus_data[14]:.4f},{can_bus_data[15]:.4f}) m/s"
        
        # 5. Convert high_level_command to string
        # command refer to https://github.com/OpenDriveLab/UniAD/issues/97#issuecomment-1661426627
        command_list = ["turn right", "turn left", "keep forward"]
        high_level_command = command_list[planning_gt["command"].item()]

        # 6. Construct the human prompt
        human_prompt = \
f"""
Following is scene information: <scene_token>
Following is agent-wise tracking information: <track_token>
Following is map information: <map_token>
Following is the ego information: {ego_info}
Following is the planning command: {high_level_command}
Predict the ego vehicle's planning trajectory.
"""
# Following is the vehicle's camera images: <image><image><image><image><image><image>

        # 7. Construct the conversation
        conversations = [
            {
                "from": "human",
                "value": human_prompt
            },
            {
                "from": "gpt", 
                "value": f"The ego vehicle's planning trajectory is {planning_gt_2d_str}."
            }
        ]
        
        # 8. Construct the full json item
        item = {
            "id": sample_token,
            # "image": img_filenames,
            "uniad_pth": str(self.input_dir / f"{sample_token}.pth"),
            "conversations": conversations
        }
        
        return item
    
    def process_single_file(self, pth_file: Path) -> Dict[str, Any]:
        """Process a single pth file and return json item."""
        data = torch.load(pth_file, map_location='cpu')
        return self.convert_to_json(data)
    
    def convert(self):
        """Convert .pth files to json in parallel."""
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get list of all pth files
        pth_files = self.get_pth_files()
        
        # Use number of CPU cores for parallel processing
        # num_processes = cpu_count
        num_processes = 16
        
        # Create a process pool
        with Pool(processes=num_processes) as pool:
            # Process files in parallel with progress bar
            results = list(tqdm(
                pool.imap(self.process_single_file, pth_files),
                total=len(pth_files),
                desc=f"Converting files using {num_processes} processes"
            ))
        
        # Write results to output file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Convert Uniad results with specified split')
    parser.add_argument('--nuscenes_split', type=str, default='train',
                       choices=['train', 'val'],
                       help='nuScenes dataset split (train or val)')
    # parser.add_argument('--nuscenes_version', type=str, default='v1.0-trainval',
    #                    choices=['v1.0-mini', 'v1.0-trainval'],
    #                    help='nuScenes dataset version (v1.0-mini or v1.0-trainval)')
    
    args = parser.parse_args()
    
    converter = UniadResultsConverter(args)
    converter.convert()

if __name__ == "__main__":
    main()
