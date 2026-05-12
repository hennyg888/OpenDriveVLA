#!/bin/bash
# Sequential Stage 3 curriculum, resumed from the map_debug Stage 2 checkpoint.
#
#   Phase A — Stage 3 WITHOUT "Ego states:" and "Historical trajectory:"
#             lines in the training prompt (--include_ego_history False).
#             Init from Stage 2 ckpt. Forces the model to solve the
#             planning task from scene/track/map/mission_goal alone.
#   Phase B — Stage 3 WITH those two lines (--include_ego_history True).
#             Init from Phase A's Stage 3 ckpt (not Stage 2). Reintroduces
#             the ego shortcut on top of a representation that already
#             learned to lean on perception.
#
# Eval runs once, on the Phase B ckpt. inference_drivevla.py is unchanged
# — eval prompts always include ego + history (matches Phase B training).

set -e
set -o pipefail

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

cd /home/s56cai/OpenDriveVLA

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
VISION_MODEL_VERSION="fusionad_track_map"
PROMPT_VERSION="qwen_planning_oriented_vlm"
NUSCENES_CFG="projects/configs/fusionad/fusion_base_track_map.py"

STAGE2_CKPT_DIR="/home/s56cai/OpenDriveVLA/output/map_debug/20260422_012832/stage2/fusionad-Qwen2.5-0.5B-map_debug_stage2_fusionad_0.5b"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT_OUTPUT_DIR="/home/s56cai/OpenDriveVLA/output/ego_ablation/${TIMESTAMP}_curriculum"
mkdir -p ${ROOT_OUTPUT_DIR}
cp "$0" "${ROOT_OUTPUT_DIR}/"

echo "========================================================"
echo "ego_ablation curriculum  (Phase A no-ego  →  Phase B with-ego  →  eval)"
echo "  timestamp:       ${TIMESTAMP}"
echo "  output root:     ${ROOT_OUTPUT_DIR}"
echo "  base (Stage 2):  ${STAGE2_CKPT_DIR}"
echo "========================================================"

if [ ! -d "${STAGE2_CKPT_DIR}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT_DIR}"
    exit 1
fi

E2E_DATA="data/fusionad_results_for_vlm/train.json"

# ============================================================================
# Phase A — Stage 3 without ego + history (init from Stage 2).
# ============================================================================
PHASE_A_EXP_NAME="e2e_phase_a_noego"
PHASE_A_OUT_DIR="${ROOT_OUTPUT_DIR}/phase_a_noego"
PHASE_A_RUN_NAME="fusionad-Qwen2.5-0.5B-${PHASE_A_EXP_NAME}"
PHASE_A_CKPT_DIR="${PHASE_A_OUT_DIR}/${PHASE_A_RUN_NAME}"
mkdir -p ${PHASE_A_OUT_DIR}/log

MASTER_PORT_A=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Phase A] Stage 3 NO ego+history — init from Stage 2"
echo "    base ckpt:  ${STAGE2_CKPT_DIR}"
echo "    data:       ${E2E_DATA}"
echo "    ckpt out:   ${PHASE_A_CKPT_DIR}"
echo "    port:       ${MASTER_PORT_A}"
echo "=========================================================="

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_A} \
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
    --run_name ${PHASE_A_RUN_NAME} \
    --output_dir ${PHASE_A_CKPT_DIR} \
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
    --include_ego_history False \
    --nuscenes_cfg_file ${NUSCENES_CFG} \
    2>&1 | tee ${PHASE_A_OUT_DIR}/log/train.log

echo ""
echo ">>> [Phase A] done."

if [ ! -d "${PHASE_A_CKPT_DIR}" ]; then
    echo "ERROR: Phase A checkpoint not found at ${PHASE_A_CKPT_DIR}"
    exit 1
fi

# ============================================================================
# Phase B — Stage 3 with ego + history (init from Phase A ckpt).
# ============================================================================
PHASE_B_EXP_NAME="e2e_phase_b_withego"
PHASE_B_OUT_DIR="${ROOT_OUTPUT_DIR}/phase_b_withego"
PHASE_B_RUN_NAME="fusionad-Qwen2.5-0.5B-${PHASE_B_EXP_NAME}"
PHASE_B_CKPT_DIR="${PHASE_B_OUT_DIR}/${PHASE_B_RUN_NAME}"
mkdir -p ${PHASE_B_OUT_DIR}/log

MASTER_PORT_B=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Phase B] Stage 3 WITH ego+history — init from Phase A"
echo "    base ckpt:  ${PHASE_A_CKPT_DIR}"
echo "    data:       ${E2E_DATA}"
echo "    ckpt out:   ${PHASE_B_CKPT_DIR}"
echo "    port:       ${MASTER_PORT_B}"
echo "=========================================================="

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_B} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${PHASE_A_CKPT_DIR} \
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
    --run_name ${PHASE_B_RUN_NAME} \
    --output_dir ${PHASE_B_CKPT_DIR} \
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
    --include_ego_history True \
    --nuscenes_cfg_file ${NUSCENES_CFG} \
    2>&1 | tee ${PHASE_B_OUT_DIR}/log/train.log

echo ""
echo ">>> [Phase B] done."

if [ ! -d "${PHASE_B_CKPT_DIR}" ]; then
    echo "ERROR: Phase B checkpoint not found at ${PHASE_B_CKPT_DIR}"
    exit 1
fi

# ============================================================================
# Eval — inference on val split + eval_drivevla metrics (Phase B ckpt only).
# ============================================================================
EVAL_DIR="${ROOT_OUTPUT_DIR}/eval"
mkdir -p ${EVAL_DIR}/log ${EVAL_DIR}/results
PLAN_CONV_PATH="${EVAL_DIR}/results/planning_conversations_val.json"
EVAL_LOG="${EVAL_DIR}/log/eval.log"

MASTER_PORT_EVAL=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Eval] inference + eval_drivevla.py (Phase B ckpt)"
echo "    final ckpt: ${PHASE_B_CKPT_DIR}"
echo "    output:     ${PLAN_CONV_PATH}"
echo "    port:       ${MASTER_PORT_EVAL}"
echo "=========================================================="

torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_EVAL} \
    drivevla/inference_drivevla.py \
    --bf16 \
    --model-path ${PHASE_B_CKPT_DIR} \
    --output ${PLAN_CONV_PATH} \
    2>&1 | tee ${EVAL_LOG}

python drivevla/eval_drivevla.py \
    --output ${PLAN_CONV_PATH} \
    --include-end-of-scene \
    2>&1 | tee -a ${EVAL_LOG}

echo ""
echo "========================================================"
echo "Curriculum pipeline done."
echo "  Run root:        ${ROOT_OUTPUT_DIR}"
echo "  Stage 2 ckpt:    ${STAGE2_CKPT_DIR}"
echo "  Phase A ckpt:    ${PHASE_A_CKPT_DIR}"
echo "  Phase B ckpt:    ${PHASE_B_CKPT_DIR}"
echo "  Eval log:        ${EVAL_LOG}"
echo "========================================================"
