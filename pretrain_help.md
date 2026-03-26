make a uniquely names pretrain config file e.g.: `OpenDriveVLA/projects/configs/uniad_bevfusion/pretrain_img_only.py`

make sure to give a new wandb run name at the bottom of the config in the wandb logger hook section

edit track_map_former for track and map heads training behavior and unaid_bevfusion for bevfusion training behavior

start tmux sessions with `tmux new -s <session_name>`

enter pixi shell, remember to check if flash-attn and shapely==1.8.5.post1 are installed

start train with `bash tools/uniad_bevfusion.sh <config_path> 8(num of GPUs)` config path is the path to your pretrain config file made in the beginning of this guide

detach from tmux session with Ctrl+B, D

when training done, outputs will be in `OpenDriveVLA/projects/work_dirs/uniad_bevfusion/<pretrain_config_filename`, so will every end of epoch checkpoints

kill tmux session with `tmux kill-session -t <session_name>`, view existing tmux sessions with `tmux ls`