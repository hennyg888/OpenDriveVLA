#!/bin/bash
# Full pipeline for the map-filter fix debug run.
# Mirrors drivevla/finetune/new_train/full_pipeline_fusionad_0.5b.sh but:
#   - all stages (Stage 1, Stage 2, Stage 3, Eval) use 9 GPUs
#   - outputs land under output/map_debug/<TIMESTAMP>/{stage1,stage2,e2e,eval}
#   - skips Stage 2.5 (Stage 3 inits directly from Stage 2 ckpt)
#
# Pickle + Stage 1/2/3 data JSONs are assumed already regenerated after the
# panseg chosen_output_query_things fix.

set -e
set -o pipefail

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

cd /home/s56cai/OpenDriveVLA

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
VISION_MODEL_VERSION="fusionad_track_map"
FUSIONAD_CKPT="/home/s56cai/ckpt/fusionad/fusion_latest.pth"
PROMPT_VERSION="qwen_planning_oriented_vlm"
NUSCENES_CFG="projects/configs/fusionad/fusion_base_track_map.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT_OUTPUT_DIR="/home/s56cai/OpenDriveVLA/output/map_debug/${TIMESTAMP}"
mkdir -p ${ROOT_OUTPUT_DIR}
cp "$0" "${ROOT_OUTPUT_DIR}/"

echo "========================================================"
echo "Pipeline start: ${TIMESTAMP}"
echo "Output root:    ${ROOT_OUTPUT_DIR}"
echo "Base model:     ${BASE_MODEL}"
echo "========================================================"

# ============================================================================
# Stage 1 — Hierarchical Vision-Language Alignment (projectors only)
# ============================================================================
STAGE1_TUNABLE_PARTS="mm_mlp_adapter"
STAGE1_TRAIN_DATA="data/stage1/stage1_combined_train.yaml"
STAGE1_EXPERIMENT_NAME="map_debug_stage1_fusionad_0.5b"
STAGE1_RUN_NAME="fusionad-Qwen2.5-0.5B-${STAGE1_EXPERIMENT_NAME}"
STAGE1_OUT_DIR="${ROOT_OUTPUT_DIR}/stage1"
STAGE1_CKPT_DIR="${STAGE1_OUT_DIR}/${STAGE1_RUN_NAME}"
mkdir -p ${STAGE1_OUT_DIR}/log

MASTER_PORT_S1=$(shuf -i 20000-60000 -n 1)

echo ""
echo ">>> [Stage 1] launching — projectors only (mm_mlp_adapter)"
echo "    tunable:   ${STAGE1_TUNABLE_PARTS}"
echo "    data:      ${STAGE1_TRAIN_DATA}"
echo "    ckpt out:  ${STAGE1_CKPT_DIR}"
echo "    port:      ${MASTER_PORT_S1}"
echo ""

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_S1} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${BASE_MODEL} \
    --version ${PROMPT_VERSION} \
    --data_path ${STAGE1_TRAIN_DATA} \
    --image_folder . \
    --mm_tunable_parts=${STAGE1_TUNABLE_PARTS} \
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
    --run_name ${STAGE1_RUN_NAME} \
    --output_dir ${STAGE1_CKPT_DIR} \
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
    2>&1 | tee ${STAGE1_OUT_DIR}/log/train.log

echo ""
echo ">>> [Stage 1] done."

# Sanity-check that Stage 1 produced the projectors Stage 2 needs.
for p in mm_projector_track.bin mm_projector_scene.bin mm_projector_map.bin; do
    if [ ! -f "${STAGE1_CKPT_DIR}/${p}" ]; then
        echo "ERROR: Stage 1 did not produce ${STAGE1_CKPT_DIR}/${p}; aborting before Stage 2."
        exit 1
    fi
done

# ============================================================================
# Stage 2 — Driving Knowledge Instruction Tuning (projectors + LLM)
# Stage 2 projectors initialized from Stage 1 weights.
# ============================================================================
STAGE2_TUNABLE_PARTS="mm_mlp_adapter,mm_language_model"
STAGE2_TRAIN_DATA="data/stage2/stage2_train.yaml"
STAGE2_EXPERIMENT_NAME="map_debug_stage2_fusionad_0.5b"
STAGE2_RUN_NAME="fusionad-Qwen2.5-0.5B-${STAGE2_EXPERIMENT_NAME}"
STAGE2_OUT_DIR="${ROOT_OUTPUT_DIR}/stage2"
STAGE2_CKPT_DIR="${STAGE2_OUT_DIR}/${STAGE2_RUN_NAME}"
mkdir -p ${STAGE2_OUT_DIR}/log

PRETRAIN_PROJECTOR_TRACK="${STAGE1_CKPT_DIR}/mm_projector_track.bin"
PRETRAIN_PROJECTOR_SCENE="${STAGE1_CKPT_DIR}/mm_projector_scene.bin"
PRETRAIN_PROJECTOR_MAP="${STAGE1_CKPT_DIR}/mm_projector_map.bin"

MASTER_PORT_S2=$(shuf -i 20000-60000 -n 1)

echo ""
echo ">>> [Stage 2] launching — projectors (from Stage 1) + LLM"
echo "    tunable:       ${STAGE2_TUNABLE_PARTS}"
echo "    data:          ${STAGE2_TRAIN_DATA}"
echo "    stage1 ckpt:   ${STAGE1_CKPT_DIR}"
echo "    ckpt out:      ${STAGE2_CKPT_DIR}"
echo "    port:          ${MASTER_PORT_S2}"
echo ""

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_S2} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${BASE_MODEL} \
    --version ${PROMPT_VERSION} \
    --data_path ${STAGE2_TRAIN_DATA} \
    --image_folder . \
    --mm_tunable_parts=${STAGE2_TUNABLE_PARTS} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_tower_lr=0 \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 256 \
    --vision_tower_pretrained ${FUSIONAD_CKPT} \
    --pretrain_mm_mlp_adapter_track ${PRETRAIN_PROJECTOR_TRACK} \
    --pretrain_mm_mlp_adapter_scene ${PRETRAIN_PROJECTOR_SCENE} \
    --pretrain_mm_mlp_adapter_map   ${PRETRAIN_PROJECTOR_MAP} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${STAGE2_RUN_NAME} \
    --output_dir ${STAGE2_CKPT_DIR} \
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
    2>&1 | tee ${STAGE2_OUT_DIR}/log/train.log

echo ""
echo ">>> [Stage 2] done."

if [ ! -d "${STAGE2_CKPT_DIR}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT_DIR}"
    exit 1
fi

# ============================================================================
# Stage 3 — E2E training (skip Stage 2.5; init from Stage 2 ckpt directly).
# Tunable: vision tower + projectors + LLM.
# ============================================================================
E2E_DATA="data/fusionad_results_for_vlm/train.json"
E2E_EXP_NAME="e2e"
E2E_OUT_DIR="${ROOT_OUTPUT_DIR}/e2e"
E2E_RUN_NAME="fusionad-Qwen2.5-0.5B-${E2E_EXP_NAME}"
E2E_CKPT_DIR="${E2E_OUT_DIR}/${E2E_RUN_NAME}"
mkdir -p ${E2E_OUT_DIR}/log

MASTER_PORT_S3=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Stage 3 / E2E] launching — init from Stage 2 (skipping Stage 2.5)"
echo "    base ckpt:  ${STAGE2_CKPT_DIR}"
echo "    data:       ${E2E_DATA}"
echo "    ckpt out:   ${E2E_CKPT_DIR}"
echo "    port:       ${MASTER_PORT_S3}"
echo "=========================================================="

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_S3} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${STAGE2_CKPT_DIR} \
    --version ${PROMPT_VERSION} \
    --data_path ${E2E_DATA} \
    --image_folder . \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 256 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${E2E_RUN_NAME} \
    --output_dir ${E2E_CKPT_DIR} \
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
    --in_nuscenes_order True \
    --use_uniad_pth False \
    --nuscenes_cfg_file ${NUSCENES_CFG} \
    2>&1 | tee ${E2E_OUT_DIR}/log/train.log

echo ""
echo ">>> [Stage 3 / E2E] done."

if [ ! -d "${E2E_CKPT_DIR}" ]; then
    echo "ERROR: E2E checkpoint not found at ${E2E_CKPT_DIR}"
    exit 1
fi

# ============================================================================
# Eval — inference on val split + eval_drivevla metrics.
# ============================================================================
EVAL_DIR="${ROOT_OUTPUT_DIR}/eval"
mkdir -p ${EVAL_DIR}/log ${EVAL_DIR}/results
PLAN_CONV_PATH="${EVAL_DIR}/results/planning_conversations_val.json"
EVAL_LOG="${EVAL_DIR}/log/eval.log"

MASTER_PORT_EVAL=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Eval] inference + eval_drivevla.py"
echo "    final ckpt: ${E2E_CKPT_DIR}"
echo "    output:     ${PLAN_CONV_PATH}"
echo "    port:       ${MASTER_PORT_EVAL}"
echo "=========================================================="

torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_EVAL} \
    drivevla/inference_drivevla.py \
    --bf16 \
    --model-path ${E2E_CKPT_DIR} \
    --output ${PLAN_CONV_PATH} \
    2>&1 | tee ${EVAL_LOG}

python drivevla/eval_drivevla.py \
    --output ${PLAN_CONV_PATH} \
    --include-end-of-scene \
    2>&1 | tee -a ${EVAL_LOG}

echo ""
echo "========================================================"
echo "Pipeline done."
echo "  Run root:     ${ROOT_OUTPUT_DIR}"
echo "  Stage 1 ckpt: ${STAGE1_CKPT_DIR}"
echo "  Stage 2 ckpt: ${STAGE2_CKPT_DIR}"
echo "  E2E ckpt:     ${E2E_CKPT_DIR}"
echo "  Eval log:     ${EVAL_LOG}"
echo "========================================================"
