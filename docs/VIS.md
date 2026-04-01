# Visualize DriveVLA planning results

The visualization figures and video will be saved in the same folder as the <pred_trajs_dict.json>.

```shell
python drivevla/visualize/run.py \
    --planning_json /PATH/TO/pred_trajs_dict.json \
    --pth_base_path /PATH/TO/pth_data_folder \
    --object_agent_traj_file /PATH/TO/object_agent_trajectories.json \
    --nuscenes_version VERSION

    # If you want to visualize specific scenes/sample, you can use the following arguments
    # --scene_tokens SCENE_TOKEN1 SCENE_TOKEN2 ... \
    # --sample_token SAMPLE_TOKEN \
```

For example

```shell
python drivevla/visualize/run.py \
    --planning_json output/ego_traj/3B_E2E/pred_trajs_dict.json \
    --pth_base_path output/result_pth_for_vis/result_3B_E2E \
    --object_agent_traj_file output/agent_traj/3B_stage3_object_agent_trajectory_on_QA_20250307_143246/planning_conversations_val.json \
    --nuscenes_version v1.0-trainval

    # --scene_tokens fcbccedd61424f1b85dcbf8f897f9754 325cef682f064c55a255f2625c533b75 \
    # or 
    # --sample_token 3e8750f331d7499e9b5123e9eb70f2e2
```
