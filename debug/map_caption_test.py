import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/s56cai/OpenDriveVLA/data/nuscenes', verbose=False)

render_cameras = False
render_map = False
sample_num = 3287
# Pick a sample and render the front camera image.
sample_token = sample_token = nusc.sample[sample_num]['token']
#sample 90: cd908fdce67249f3a8a14b987f00592e
#sample 3287: 5ff472758749447ca27827543e5d50dd
print("sample_token: ", sample_token)

sample = nusc.get('sample', 'f0155b286316424f8058a6a2a3f74721')
#nusc.sample[sample_num]

scene = nusc.get('scene', sample['scene_token'])
log = nusc.get('log', scene['log_token'])
map_name = log['location']   # e.g. 'boston-seaport'
nusc_map = NuScenesMap(dataroot='/home/s56cai/OpenDriveVLA/data/nuscenes', map_name=map_name)
bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')


sd_token = sample['data']['LIDAR_TOP']
sd = nusc.get('sample_data', sd_token)

ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
x, y, z = ego_pose['translation']
print(f"Ego pose translation: x={x}, y={y}, z={z}")

#print('Next road objects:', nusc_map.get_next_roads(x, y))
#print('Road objects on selected point:', nusc_map.layers_on_point(x, y), '\n')

#54m is lidar point cloud radius range
radius = 51.2  # meters

patch = (
    x - radius,
    y - radius,
    x + radius,
    y + radius
)

records_intersect_patch = nusc_map.get_records_in_patch(patch, nusc_map.non_geometric_layers, mode='within')
for layer in nusc_map.non_geometric_layers:
    print(f"layer {layer} with token num {len(records_intersect_patch[layer])}")
for seg_tok in records_intersect_patch['lane']:
    segment = nusc_map.get('lane', seg_tok)
    print(segment['lane_type'])
#nusc_map.non_geometric_layers
if render_map:
    fig, ax = nusc_map.render_map_patch(patch, ['road_segment'], figsize=(20, 20), bitmap=bitmap)
    fig.savefig("map_patch.png", dpi=300, bbox_inches="tight")

#close to desired result for sample 90 at 54m radius mode=within
#stop_line with stop_line_type likely "edge of drivable area"
#road_divider likely "physical divider" + also need cone objects count?
#lane_divider likely "lane marker separating lanes"
#ped_crossing likely "pedestrian crosswalk marked with white stripes", but maybe actually walkway

if render_cameras:
    camera_channels = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, cam in zip(axes, camera_channels):
        sample_data_token = sample['data'][cam]

        # Render image onto the given axis
        nusc.render_sample_data(
            sample_data_token,
            ax=ax,
            with_anns=False,
            verbose=False
        )

        ax.set_title(cam)
        ax.axis('off')

    plt.tight_layout()

    # Save figure
    out_path = f"sample_{sample_num}_cameras.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)