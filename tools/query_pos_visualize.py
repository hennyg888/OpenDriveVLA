import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualize UNiAD track_map_former query points in BEV")
parser.add_argument("--ckpt", default="/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain_decoder/epoch_6.pth", type=str)
parser.add_argument("--normalize", action="store_true", help="Apply sigmoid+pc_range normalization (default false)")
parser.add_argument("--pc-range", nargs=6, type=float, default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], help="Point cloud range for normalization")
# 'normalize' toggles both sigmoid and pc_range mapping as one boolean flag
parser.add_argument("--output", default="bev_ref_points.png", type=str, help="Output plot filename")
args = parser.parse_args()

ckpt_path = args.ckpt

# Load checkpoint (map to CPU to be safe)
checkpoint = torch.load(ckpt_path, map_location="cpu")

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# key names in UNiAD track_map_former module
qe_key = "track_map_former.query_embedding.weight"
rp_w_key = "track_map_former.reference_points.weight"
rp_b_key = "track_map_former.reference_points.bias"

for k in (qe_key, rp_w_key, rp_b_key):
    if k not in state_dict:
        raise KeyError(f"Required checkpoint key not found: {k}")

query_embedding = state_dict[qe_key].float()  # (num_query+1, embed_dims*2)
ref_w = state_dict[rp_w_key].float()           # (3, embed_dims)
ref_b = state_dict[rp_b_key].float()           # (3,)

# Perform query transformation
num_queries, total_dim = query_embedding.shape
half_dim = total_dim // 2
query_half = query_embedding[:, :half_dim]    # first half corresponds to query[..., :dim//2]

ref_linear = nn.Linear(half_dim, 3)
ref_linear.weight.data.copy_(ref_w)
ref_linear.bias.data.copy_(ref_b)

with torch.no_grad():
    ref_pts = ref_linear(query_half)           # (num_query+1, 3)

# Optional normalization to world coordinate with sigmoid+pc_range (same as track logic)
if args.normalize:
    ref_pts_sig = ref_pts.sigmoid()
    pr = args.pc_range
    x = ref_pts_sig[:, 0:1] * (pr[3] - pr[0]) + pr[0]
    y = ref_pts_sig[:, 1:2] * (pr[4] - pr[1]) + pr[1]
    z = ref_pts_sig[:, 2:3] * (pr[5] - pr[2]) + pr[2]
    ref_pts_world = torch.cat([x, y, z], dim=-1)
else:
    ref_pts_world = ref_pts

print(f"Loaded {num_queries} queries and produced {ref_pts_world.shape[0]} points (normalize={args.normalize})")

# BEV plot (x,y) colored by z
pts_np = ref_pts_world.cpu().numpy()
plt.figure(figsize=(8, 8))
plt.scatter(pts_np[:, 0], pts_np[:, 1], c=pts_np[:, 2], cmap="viridis", s=20, alpha=0.8)
plt.colorbar(label="z (meters)")
plt.title("BEV of reference points from track_map_former")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal", "box")
plt.grid(True)
output_path = "bev_ref_points_ours.png"
plt.savefig(output_path, bbox_inches="tight", dpi=200)
print(f"Saved BEV plot to {output_path}")
