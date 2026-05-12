#!/bin/bash
# Stage 1: Hierarchical Vision-Language Alignment
# Only mm_mlp_adapter (3 projectors: scene, track, map) is trained.
# LLM and vision tower are frozen.
# Base model: Qwen/Qwen2.5-0.5B-Instruct

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

TUNABLE_PARTS="mm_mlp_adapter"
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
VISION_MODEL_VERSION="fusionad_track_map"
FUSIONAD_CKPT="/home/s56cai/ckpt/fusionad/fusion_latest.pth"
PROMPT_VERSION="qwen_planning_oriented_vlm"

TRAIN_DATA_PATH="data/stage1/stage1_combined_train.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="stage1_fusionad_0.5b"
OUTPUT_DIR="output/stage1/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}/log

BASE_RUN_NAME="fusionad-Qwen2.5-0.5B-${EXPERIMENT_NAME}"

echo ">>> Stage 1 Training Configuration:"
echo "- BASE_MODEL:      ${BASE_MODEL}"
echo "- TUNABLE_PARTS:   ${TUNABLE_PARTS}"
echo "- VISION_TOWER:    ${VISION_MODEL_VERSION}"
echo "- FUSIONAD_CKPT:   ${FUSIONAD_CKPT}"
echo "- TRAIN_DATA:      ${TRAIN_DATA_PATH}"
echo "- OUTPUT_DIR:      ${OUTPUT_DIR}"

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=8 \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${BASE_MODEL} \
    --version ${PROMPT_VERSION} \
    --data_path ${TRAIN_DATA_PATH} \
    --image_folder . \
    --mm_tunable_parts=${TUNABLE_PARTS} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_tower_lr=0 \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 256 \
    --vision_tower_pretrained ${FUSIONAD_CKPT} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${BASE_RUN_NAME} \
    --output_dir ${OUTPUT_DIR}/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --in_nuscenes_order False \
    --use_uniad_pth True \
    --skip_build_conversation True \
    2>&1 | tee ${OUTPUT_DIR}/log/train.log
