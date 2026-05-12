#!/bin/bash
# Resume pipeline starting from an existing Stage 2 checkpoint:
#   [1/3] Stage 2.5  (agent-relative trajectory data, mm_mlp_adapter + LLM)
#   [2/3] Stage 3 E2E training
#   [3/3] Inference (9 GPUs) + eval_drivevla.py on full val set
#          (eval uses --include-end-of-scene, so all tokens are scored)
#
# Outputs land under a NEW timestamped folder inside
#   /home/s56cai/OpenDriveVLA/output/new_train/<TIMESTAMP>/
#   {stage25,e2e,eval}/
#
# Launch:
#   bash drivevla/finetune/new_train/resume_stage25_e2e_eval_fusionad_0.5b.sh
set -e
set -o pipefail

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

cd /home/s56cai/OpenDriveVLA

# ---------- Shared ----------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_ROOT="/home/s56cai/OpenDriveVLA/output/new_train"
RUN_ROOT="${PIPELINE_ROOT}/${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"
cp "$0" "${RUN_ROOT}/"

PROMPT_VERSION="qwen_planning_oriented_vlm"
VISION_MODEL_VERSION="fusionad_track_map"
FUSIONAD_CKPT="/home/s56cai/ckpt/fusionad/fusion_latest.pth"
NUSCENES_CFG="projects/configs/fusionad/fusion_base_track_map.py"

STAGE2_CKPT_DIR="/home/s56cai/OpenDriveVLA/output/new_train/20260419_180613/stage2/fusionad-Qwen2.5-0.5B-new_train_stage2_fusionad_0.5b"

if [ ! -d "${STAGE2_CKPT_DIR}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT_DIR}"
    exit 1
fi

echo "========================================================"
echo "Pipeline start: ${TIMESTAMP}"
echo "Run root:       ${RUN_ROOT}"
echo "Stage 2 ckpt:   ${STAGE2_CKPT_DIR}"
echo "========================================================"

# ---------- [1/3] Stage 2.5 ----------
STAGE25_DATA="/home/hhguo/OpenDriveVLA/data/stage25/stage25_agent_rel_trajectory_train.json"
STAGE25_EXP_NAME="stage25_agent_rel"
STAGE25_OUTPUT_DIR="${RUN_ROOT}/stage25"
STAGE25_RUN_NAME="fusionad-Qwen2.5-0.5B-${STAGE25_EXP_NAME}"
STAGE25_CKPT_DIR="${STAGE25_OUTPUT_DIR}/${STAGE25_RUN_NAME}"
mkdir -p "${STAGE25_OUTPUT_DIR}/log"

MASTER_PORT_S25=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [1/3] Stage 2.5 training"
echo "    base ckpt:  ${STAGE2_CKPT_DIR}"
echo "    data:       ${STAGE25_DATA}"
echo "    output:     ${STAGE25_CKPT_DIR}"
echo "    port:       ${MASTER_PORT_S25}"
echo "=========================================================="

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_S25} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${STAGE2_CKPT_DIR} \
    --version ${PROMPT_VERSION} \
    --data_path ${STAGE25_DATA} \
    --image_folder . \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
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
    --run_name ${STAGE25_RUN_NAME} \
    --output_dir ${STAGE25_CKPT_DIR} \
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
    2>&1 | tee ${STAGE25_OUTPUT_DIR}/log/train.log

if [ ! -d "${STAGE25_CKPT_DIR}" ]; then
    echo "ERROR: Stage 2.5 checkpoint not found at ${STAGE25_CKPT_DIR}"
    exit 1
fi

# ---------- [2/3] Stage 3 E2E ----------
E2E_DATA="data/fusionad_results_for_vlm/train.json"
E2E_EXP_NAME="e2e"
E2E_OUTPUT_DIR="${RUN_ROOT}/e2e"
E2E_RUN_NAME="fusionad-Qwen2.5-0.5B-${E2E_EXP_NAME}"
E2E_CKPT_DIR="${E2E_OUTPUT_DIR}/${E2E_RUN_NAME}"
mkdir -p "${E2E_OUTPUT_DIR}/log"

MASTER_PORT_S3=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [2/3] Stage 3 E2E training"
echo "    base ckpt:  ${STAGE25_CKPT_DIR}"
echo "    data:       ${E2E_DATA}"
echo "    output:     ${E2E_CKPT_DIR}"
echo "    port:       ${MASTER_PORT_S3}"
echo "=========================================================="

ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=9 --master_port=${MASTER_PORT_S3} \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${STAGE25_CKPT_DIR} \
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
    2>&1 | tee ${E2E_OUTPUT_DIR}/log/train.log

if [ ! -d "${E2E_CKPT_DIR}" ]; then
    echo "ERROR: E2E checkpoint not found at ${E2E_CKPT_DIR}"
    exit 1
fi

# ---------- [3/3] Inference + eval ----------
EVAL_DIR="${RUN_ROOT}/eval"
mkdir -p "${EVAL_DIR}/log" "${EVAL_DIR}/results"
PLAN_CONV_PATH="${EVAL_DIR}/results/planning_conversations_val.json"
EVAL_LOG="${EVAL_DIR}/log/eval.log"

MASTER_PORT_EVAL=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [3/3] Inference (9 GPUs) + eval_drivevla.py"
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
echo "  Run root:    ${RUN_ROOT}"
echo "  Stage 2.5:   ${STAGE25_CKPT_DIR}"
echo "  Stage 3 E2E: ${E2E_CKPT_DIR}"
echo "  Eval log:    ${EVAL_LOG}"
echo "========================================================"
