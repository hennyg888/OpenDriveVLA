import torch

ckpt_path = "/home/s56cai/ckpt/uniad_stage1/uniad_base_track_map.pth"
#"/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain_decoder/epoch_3.pth"

# Load checkpoint (map to CPU to be safe)
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Print top-level keys (helps understand structure)
print("Checkpoint keys:", checkpoint.keys())

# Most common cases:
# 1. checkpoint["state_dict"]
# 2. checkpoint["model"]
# 3. checkpoint itself is already the state_dict

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

print("\nParameter names and shapes:\n")

for name, param in state_dict.items():
    if isinstance(param, torch.Tensor):
        print(f"{name}: {tuple(param.shape)}")
    else:
        print(f"{name}: (not a tensor, type={type(param)})")