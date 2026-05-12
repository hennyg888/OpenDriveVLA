#!/bin/bash
# Stage 2.5: Conditional Agent Trajectory Forecasting
# Projectors + LLM continue to be trained; vision encoder is frozen.
# Data: Agent trajectory prediction (scene + track + map + ego state → future waypoints)
# Base model: Stage 2 checkpoint (fusionad-Qwen2.5-0.5B)

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

TUNABLE_PARTS="mm_mlp_adapter,mm_language_model"
VISION_MODEL_VERSION="fusionad_track_map"
FUSIONAD_CKPT="/home/s56cai/ckpt/fusionad/fusion_latest.pth"
PROMPT_VERSION="qwen_planning_oriented_vlm"

# Stage 2 checkpoint (projector + LLM weights already embedded)
STAGE2_CKPT_DIR="/home/s56cai/OpenDriveVLA/output/stage2/stage2_fusionad_0.5b_20260413_164921/fusionad-Qwen2.5-0.5B-stage2_fusionad_0.5b"

TRAIN_DATA_PATH="data/stage25/stage25_train.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="stage25_fusionad_0.5b"
OUTPUT_DIR="output/stage25/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}/log

BASE_RUN_NAME="fusionad-Qwen2.5-0.5B-${EXPERIMENT_NAME}"

echo ">>> Stage 2.5 Training Configuration:"
echo "- STAGE2_CKPT:     ${STAGE2_CKPT_DIR}"
echo "- TUNABLE_PARTS:   ${TUNABLE_PARTS}"
echo "- TRAIN_DATA:      ${TRAIN_DATA_PATH}"
echo "- OUTPUT_DIR:      ${OUTPUT_DIR}"

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${STAGE2_CKPT_DIR} \
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
    --learning_rate 1e-5 \
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
