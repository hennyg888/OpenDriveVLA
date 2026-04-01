import os
import torch
import shutil
from transformers import AutoModelForVision2Seq, AutoTokenizer

# Define paths
model_path = "/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/checkpoints/Qwen2.5-VL-3B"
output_path = "/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/checkpoints/Qwen2.5-VL-3B-mm-projector"

# Create the target directory if it does not exist
os.makedirs(output_path, exist_ok=True)

# Load the model
model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save the Vision-Language Connector (merger)
vl_connector = model.visual.merger
torch.save(vl_connector.state_dict(), f"{output_path}/vl_connector.pth")
print("Vision-Language Connector (VLM) saved successfully!")

# Save the Multi-Modal Projector (mm_projector) if it exists
if hasattr(model, "mm_projector"):
    torch.save(model.mm_projector.state_dict(), f"{output_path}/mm_projector.pth")
    print("Multi-modal projector (mm_projector) saved successfully!")

# Copy config.json
shutil.copy(f"{model_path}/config.json", f"{output_path}/config.json")
print("Config file saved successfully!")

# Copy generation_config.json if it exists
gen_config_path = f"{model_path}/generation_config.json"
if os.path.exists(gen_config_path):
    shutil.copy(gen_config_path, f"{output_path}/generation_config.json")
    print("Generation config file saved successfully!")

print("All necessary components saved successfully!")