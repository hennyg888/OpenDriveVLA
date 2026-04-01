import sys
sys.path.append('.')

import cv2
import torch
import argparse
import os
import glob
import numpy as np
import mmcv
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.utils import splits
from map_api import NuScenesMap
from utils import AgentPredictionData
from render.bev_render import BEVRender
from render.cam_render import CameraRender
import json
from tqdm import tqdm
from drivevla.utils.trajectory_utils import retrieve_traj
import pickle
from drivevla.utils.tensor_utils import change_tensor_to_float32
from projects.mmdet3d_plugin.uniad.detectors.uniad_e2e import match_bbox

class Visualizer:
    """
    BaseRender class
    """

    def __init__(
            self,
            dataroot='/mnt/petrelfs/yangjiazhi/e2e_proj/data/nus_mini',
            version='v1.0-mini',
            perception_pkl=None,
            planning_json=None,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_gt_boxes=False,
            show_lidar=False,
            show_command=False,
            show_hd_map=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False,
            pth_base_path=None,
            object_agent_traj_file=None,
            scene_tokens=None,
            sample_token=None):
        if os.path.exists('data/nusc.pkl'):
            print(">>> Loading NuScenes from cache: data/nusc.pkl")
            with open('data/nusc.pkl', 'rb') as f:
                self.nusc = pickle.load(f)
            print(">>> Successfully loaded NuScenes from cache")
        else:
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_lidar = show_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.token_set = set()
        self.scene_tokens = scene_tokens
        self.sample_token = sample_token
        self.pth_base_path = pth_base_path
        self.object_agent_traj_file = object_agent_traj_file

        self.predictions, self.n_trajectories = self._parse_predictions_multitask(perception_pkl, planning_json)

        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

        if self.show_hd_map:
            self.nusc_maps = {
                'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
                'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
            }

    def build_uniad_results(self, perception_pkl, planning_json):
        # if perception_pkl does not exist, build it
        if not os.path.exists(perception_pkl):
            print(f">>> Perception pkl file {perception_pkl} does not exist, building it...")
            uniad_e2e_results:dict = mmcv.load('data/uniad_results_for_vlm/uniad_e2e_results.pkl')
            del uniad_e2e_results['occ_results_computed']
            del uniad_e2e_results['planning_results_computed']
            for bbox_result in uniad_e2e_results['bbox_results']:
                del bbox_result['planning_traj']
                # del bbox_result['traj_0']
                # del bbox_result['traj_scores_0']
                # del bbox_result['traj_1']
                # del bbox_result['traj_scores_1']
                # del bbox_result['traj']
                # del bbox_result['traj_scores']

            # save the new perception pkl
            mmcv.dump(uniad_e2e_results, perception_pkl)
            print(f">>> Perception results saved to {perception_pkl}")

        uniad_perception_results = mmcv.load(perception_pkl)

        with open(planning_json, 'r') as f:
            pred_trajs_dict = json.load(f)
        
        for bbox_result in uniad_perception_results['bbox_results']:
            bbox_result['planning_traj'] = torch.tensor(pred_trajs_dict[f"{bbox_result['token']}_trajectory"])
            
        return uniad_perception_results
    
    def rematch_to_get_lookup(self, pth_data):
        gt_inds = pth_data[0]['gt_inds']
        pth_data[0]['track_gt_inds_to_embed_idx'] = {}

        detected_bboxes = pth_data[0]['track_bbox_results'][0][0].tensor
        matched_idx, iou_matrix = match_bbox(detected_bboxes[:-1, :7], pth_data[0]['gt_bboxes_3d'][:,:7])
        for embed_idx, gt_inds_idx in enumerate(matched_idx):
            if len(gt_inds) == 0:
                print("gt_inds is empty")
                continue
            if gt_inds_idx != -1:
                pth_data[0]['track_gt_inds_to_embed_idx'][int(gt_inds[gt_inds_idx])] = embed_idx


    def _parse_predictions_multitask(self, perception_pkl, planning_json):
        outputs = self.build_uniad_results(perception_pkl, planning_json) # outputs is a dict 
        outputs = outputs['bbox_results']
        prediction_dict = dict()

        # scene_token = "fcbccedd61424f1b85dcbf8f897f9754"  # move forward

        # scene_token = "84e056bd8e994362a37cba45c0f75558"  # final turn right
        # scene_token = "cba3ddd5c3664a43b6a08e586e094900"  # turn right, failed to change to turn left
        # scene_token = "ddb615d9bb22484cabc6545b632a1025"  # turn right, no way to foward

        # scene_token = "ed242d80ccb34b139aaf9ab89859332e"  # turn left, all can at 17816, but detection failed

        # scene_token = "d29527ec841045d18d04a933e7a0afd2"  # turn left, fail to move forward
        # scene_token = "a499ee875da34e2b9655afb999edb8a9"  # keep forward, GOOD
        # scene_token = "efa5c96f05594f41a2498eb9f2e7ad99" # turn left, GOOD
        # scene_token = "f5b29a1e09d04355adcd60ab72de006b" # turn left

        # following are all move forward
        # scene_token = "c525507ee2ef4c6d8bb64b0e0cf0dd32" # final lot of cars good
        # scene_token = "3dd2be428534403ba150a0b60abc6a0a" # too fast
        # scene_token = "4962cb207a824e57bd10a2af49354b16" # too fast
        # scene_token = "2ed0fcbfc214478ca3b3ce013e7723ba" # bad
        # scene_token = "c65c4acf86954f8cbd53a3541a3bfa3a" # too fast
        # scene_token = "5a0dd8908a3a459b83ec5eb6ac7d0f82" # fine fast
        # scene_token = "41fde20fedcd4d22ab26811688612870"
        
        # UniAD drive on sidewalk
        # sample_tokens = ["fd8420396768425eabec9bdddf7e64b6"]

        # sidewalk
        # scene_token = "acc29386502047339e1ec6b9c7e512d2"
        # scene_token = "813213458a214a39a1d1fc77fa52fa34"
        
        if self.scene_tokens:
            scene_tokens = self.scene_tokens
            sample_tokens = []
            for scene_token in scene_tokens:
                scene = self.nusc.get('scene', scene_token)
                sample_token = scene['first_sample_token']
                while sample_token != '':
                    sample = self.nusc.get('sample', sample_token)
                    sample_tokens.append(sample_token)
                    sample_token = sample['next']

        elif self.sample_token:
            sample_tokens = [self.sample_token]

        else:
            sample_tokens = []


        pth_file_path_base = self.pth_base_path

        if self.scene_tokens is None and self.sample_token is None:
            for file in os.listdir(pth_file_path_base):
                if file.endswith(".pth"):
                    sample_tokens.append(file.split('.')[0])
            print(f"Loading {len(sample_tokens)} samples")

        for sample_token in tqdm(sample_tokens, desc="Processing samples"):
            # find pth file
            pth_file_path = None
            pth_file = f"{sample_token}.pth"
            potential_pth_path = os.path.join(pth_file_path_base, pth_file)
            
            if os.path.exists(potential_pth_path):
                pth_file_path = potential_pth_path
            else:
                print(f"Can't find pth file for sample {sample_token}")
                continue
            
            # load pth file
            try:
                pth_data = torch.load(pth_file_path, map_location='cpu')
            except Exception as e:
                print(f"Failed to load pth file for sample {sample_token}: {e}")
                continue

            # convert bf16 tensor in pth_data to float32
            pth_data = change_tensor_to_float32(pth_data)

            # self.rematch_to_get_lookup(pth_data)

            matching_entries = []

            # load json file
            object_agent_trajectory_file = self.object_agent_traj_file
            with open(object_agent_trajectory_file, 'r') as f:
                # json_data = json.load(f)
                for line in f:
                    json_data = json.loads(line.strip())
                    id_parts = json_data['id'].split('_')
                    sample_id = id_parts[0]
                    if sample_id != sample_token:
                        continue
                    else:
                        box_token = id_parts[1]
                        instance_token = self.nusc.get('sample_annotation', box_token)['instance_token']
                        instance_id = self.nusc.getind('instance', instance_token) + 1

                        if len(json_data['answer']) == 1:
                            traj_text = np.array(retrieve_traj(json_data['answer'][0]))
                        else:
                            trajs_list = []
                            for traj_idx in range(len(json_data['answer'])):
                                trajs_list.append(np.array(retrieve_traj(json_data['answer'][traj_idx])))
                        
                        traj_text = np.array(retrieve_traj(json_data['answer'][0]))
                        matching_entries.append({
                        'sample_token': sample_token,
                        'box_token': box_token,
                        'instance_token': instance_token,
                        'instance_id':instance_id,
                        'trajs': traj_text if len(json_data['answer']) == 1 else trajs_list,
                        })


            for k in range(len(outputs)):
                token = outputs[k]['token']

                if token != sample_token:
                    continue

                if 'track_ids' in outputs[k]:
                    track_ids_from_pth = pth_data[0]['track_ids'].cpu().detach().numpy()
                else:
                    track_ids_from_pth = None

                for i, entry in enumerate(matching_entries):
                    instance_id = entry['instance_id']
                    if instance_id in pth_data[0]['track_gt_inds_to_embed_idx']:
                        track_id_index = pth_data[0]['track_gt_inds_to_embed_idx'][instance_id]
                        track_id = track_ids_from_pth[track_id_index]
                        entry['track_id_index'] = track_id_index
                        entry['track_id'] = track_id

                self.token_set.add(token)

                # if self.show_sdc_traj: # show_sdc_traj is False
                #     outputs[k]['boxes_3d'].tensor = torch.cat(
                #         [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                #     outputs[k]['scores_3d'] = torch.cat(
                #         [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
                #     outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
                #         (1,), device=outputs[k]['labels_3d'].device)], dim=0)
                    
                # detection
                bboxes = pth_data[0]['boxes_3d']
                scores = pth_data[0]['scores_3d']
                labels = pth_data[0]['labels_3d']

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels.cpu().detach().numpy()
                track_boxes = bboxes.tensor.cpu().detach().numpy()

                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims = bboxes.dims.cpu().detach().numpy()
                track_yaw = bboxes.yaw.cpu().detach().numpy()

                # speed
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                # trajectories
                n_objects = len(track_scores)
                n_trajectories = len(json_data['answer'])
                if matching_entries and len(matching_entries) > 0:
                    if n_trajectories == 1:
                        if 'trajs' in matching_entries[0]:
                            n_timestamps = matching_entries[0]['trajs'].shape[0]
                            n_coords = matching_entries[0]['trajs'].shape[1]
                        else:
                            n_timestamps = 6
                            n_coords = 2
                    else:
                        if 'trajs' in matching_entries[0] and len(matching_entries[0]['trajs']) > 0:
                            n_timestamps = matching_entries[0]['trajs'][0].shape[0]
                            n_coords = matching_entries[0]['trajs'][0].shape[1]
                        else:
                            n_timestamps = 6
                            n_coords = 2
                else:
                    n_timestamps = 6
                    n_coords = 2

                trajs_tensor = torch.zeros((n_objects, n_trajectories, n_timestamps, n_coords))
                trajs_score_tensor = torch.zeros((n_objects, n_trajectories))

                # mask for objects that have predicted trajectories in json
                has_traj_mask = torch.zeros(n_objects, dtype=torch.bool)

                # fill trajs_tensor and trajs_score_tensor
                for entry in matching_entries:
                    if 'track_id_index' in entry and 'trajs' in entry:
                        track_idx = entry['track_id_index']

                        if n_trajectories == 1:
                            trajs_tensor[track_idx, 0, :, :] = torch.tensor(entry['trajs'])
                            # set score to 1.0 if there is only one trajectory
                            trajs_score_tensor[track_idx, 0] = 1.0
                            has_traj_mask[track_idx] = True
                        else:
                            for traj_idx, traj in enumerate(entry['trajs']):
                                trajs_tensor[track_idx, traj_idx, :, :] = torch.tensor(traj)
                                trajs_score_tensor[track_idx, traj_idx] = 1.0
                            has_traj_mask[track_idx] = True
                
                trajs_tensor = trajs_tensor.numpy()
                trajs_score_tensor = trajs_score_tensor.numpy()

                predicted_agent_list = []

                # occflow
                if self.with_occ_map: 
                    if 'topk_query_ins_segs' in outputs[k]['occ']:
                        occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu(
                        ).numpy()
                    else:
                        occ_map = np.zeros((1, 5, 200, 200))
                else:
                    occ_map = None

                occ_idx = 0

                for i in range (len(track_scores)):

                    if track_scores[i] < 0.25:
                        continue

                    pred_traj = trajs_tensor[i] 
                    pred_traj_score = trajs_score_tensor[i] 
                    pred_track_id = track_ids_from_pth[i]
                    
                                
                    if occ_map is not None and track_labels[i] in self.veh_id_list:
                        occ_map_cur = occ_map[occ_idx, :, ::-1]
                        occ_idx += 1
                    else:
                        occ_map_cur = None
                    
                    predicted_agent_list.append(
                        AgentPredictionData(
                            track_scores[i],
                            track_labels[i],
                            track_centers[i],
                            track_dims[i],
                            track_yaw[i],
                            track_velocity[i],
                            pred_traj,
                            pred_traj_score,
                            pred_track_id=pred_track_id,
                            pred_occ_map=occ_map_cur,
                            past_pred_traj=None
                        )
                    )

                if self.with_map: 
                    map_thres = 0.7
                    score_list = pth_data[0]['pts_bbox']['score_list'].cpu().numpy().transpose([
                        1, 2, 0])
                    predicted_map_seg = pth_data[0]['pts_bbox']['lane_score'].cpu().numpy().transpose([
                        1, 2, 0])  # H, W, C
                    predicted_map_seg[..., -1] = score_list[..., -1]
                    predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                    predicted_map_seg = predicted_map_seg[::-1, :, :]
                else:
                    predicted_map_seg = None

                if self.with_planning:
                    # detection
                    bboxes = pth_data[0]['sdc_boxes_3d']
                    scores = pth_data[0]['sdc_scores_3d']
                    labels = 0

                    track_scores = scores.cpu().detach().numpy()
                    track_labels = labels
                    track_boxes = bboxes.tensor.cpu().detach().numpy()

                    track_centers = bboxes.gravity_center.cpu().detach().numpy()
                    track_dims = bboxes.dims.cpu().detach().numpy()
                    track_yaw = bboxes.yaw.cpu().detach().numpy()
                    track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                    if self.show_command:
                        command = outputs[k]['command'][0].cpu().detach().numpy()
                    else:
                        command = None
                    planning_agent = AgentPredictionData(
                        track_scores[0],
                        track_labels,
                        track_centers[0],
                        track_dims[0],
                        track_yaw[0],
                        track_velocity[0],
                        outputs[k]['planning_traj'].cpu().detach().numpy(),
                        1,
                        pred_track_id=-1,
                        pred_occ_map=None,
                        past_pred_traj=None,
                        is_sdc=True,
                        command=command,
                    )
                    predicted_agent_list.append(planning_agent)
                else:
                    planning_agent = None
                prediction_dict[token] = dict(predicted_agent_list=predicted_agent_list,
                                            predicted_map_seg=predicted_map_seg,
                                            predicted_planning=planning_agent)
            del pth_data
        return prediction_dict, n_trajectories

    def visualize_bev(self, sample_token, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.show_lidar: 
            self.bev_render.show_lidar_data(sample_token, self.nusc)
        if self.bev_render.show_gt_boxes: 
            self.bev_render.render_anno_data(
                sample_token, self.nusc, self.predict_helper)
        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'], top_k=self.n_trajectories)
        if self.with_map: 
            self.bev_render.render_pred_map_data(
                self.predictions[sample_token]['predicted_map_seg'])
        if self.with_occ_map: 
            self.bev_render.render_occ_map_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [self.predictions[sample_token]['predicted_planning']])
            self.bev_render.render_planning_data(
                self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
        if self.show_hd_map: 
            self.bev_render.render_hd_map(
                self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        self.cam_render.render_pred_traj(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def to_video(self, folder_path, out_path, fps=4, downsample=1):
        imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

def main(args):
    render_cfg = dict(
        with_occ_map=False,
        with_map=True,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=False,
        show_lidar=False,
        show_command=True,
        show_hd_map=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )

    viser = Visualizer(version=args.nuscenes_version, perception_pkl=args.perception_pkl, planning_json=args.planning_json, pth_base_path = args.pth_base_path, object_agent_traj_file=args.object_agent_traj_file, scene_tokens = args.scene_tokens, sample_token = args.sample_token, dataroot='data/nuscenes', **render_cfg)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    val_splits = splits.val

    scene_token_to_name = dict()
    for i in range(len(viser.nusc.scene)):
        scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']

    for i in tqdm(range(len(viser.nusc.sample)), desc="Visualizing samples"):
        sample_token = viser.nusc.sample[i]['token']
        scene_token = viser.nusc.sample[i]['scene_token']

        if scene_token_to_name[scene_token] not in val_splits:
            # print(i, sample_token, 'not in val set!')
            continue

        if sample_token not in viser.token_set:
            # print(i, sample_token, 'not in prediction pkl!')
            continue

        viser.visualize_bev(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))

        if args.project_to_cam:
            viser.visualize_cam(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))
            viser.combine(os.path.join(args.out_folder, str(i).zfill(3)))

    # viser.to_video(args.out_folder, args.demo_video, fps=4, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_version', default='v1.0-mini', help='Visualize on nuscenes v1.0-mini or v1.0-trainval set')
    parser.add_argument('--perception_pkl', default='data/uniad_results_for_vlm/uniad_perception_results.pkl', help='Path to uniad perception results.pkl')
    parser.add_argument('--planning_json', default='output/uniad-lmms-lab_llava-onevision-qwen2-0.5b-ov-mm_language_model/20250131_222036_1_epoch/pred_trajs_dict.json', help='Path to planning results.json')
    parser.add_argument('--out_folder', default=None, help='Output folder path')
    parser.add_argument('--demo_video', default=None, help='Demo video name')
    parser.add_argument('--project_to_cam', default=True, help='Project to cam (default: True)')
    parser.add_argument('--pth_base_path', default='output/result_pth_for_vis/result_3B_E2E_1GPU', help='Path to directory containing pth files')
    parser.add_argument('--object_agent_traj_file', default='output/agent_traj/3B_stage3_object_agent_trajectory_on_QA_20250307_143246/planning_conversations_val_mini.json', help='Path to agent trajectory file')
    parser.add_argument('--scene_tokens', default=None, nargs='+', help='List of scene tokens (space separated)')
    parser.add_argument('--sample_token', default=None, help='Sample token')

    args = parser.parse_args()

    if args.out_folder is None or args.demo_video is None:
        args.out_folder = os.path.join(os.path.dirname(args.planning_json), 'figures/')
        args.demo_video = os.path.join(os.path.dirname(args.planning_json), 'test_demo.avi')

    main(args)
