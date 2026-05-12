#!/bin/bash
# Resume an interrupted full_pipeline run from Stage 3 onward (E2E + Eval).
# Reuses the existing Stage 2 checkpoint already inside a timestamped
# run folder; Stage 3 and Eval outputs are written into the SAME folder
# (under e2e/ and eval/), so the resumed run is indistinguishable from
# a completed one.
#
# Any partial e2e/ or eval/ contents from the failed run are moved aside
# to e2e.failed_<ts>/ and eval.failed_<ts>/ rather than deleted.
#
# Launch (default picks up the user's target run):
#   bash drivevla/finetune/new_train/resume_from_stage3_fusionad_0.5b.sh
# Override run folder:
#   bash drivevla/finetune/new_train/resume_from_stage3_fusionad_0.5b.sh /path/to/run_root

set -e
set -o pipefail

export WANDB_PROJECT="OpenDriveVLA"
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

cd /home/s56cai/OpenDriveVLA

RUN_ROOT="${1:-/home/s56cai/OpenDriveVLA/output/new_train/20260419_180613}"

# ---------- shared (mirror full_pipeline_fusionad_0.5b.sh) ----------
VISION_MODEL_VERSION="fusionad_track_map"
PROMPT_VERSION="qwen_planning_oriented_vlm"
NUSCENES_CFG="projects/configs/fusionad/fusion_base_track_map.py"

STAGE2_RUN_NAME="fusionad-Qwen2.5-0.5B-new_train_stage2_fusionad_0.5b"
STAGE2_CKPT_DIR="${RUN_ROOT}/stage2/${STAGE2_RUN_NAME}"

if [ ! -d "${STAGE2_CKPT_DIR}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT_DIR}"
    exit 1
fi

SAFE_TS=$(date +%Y%m%d_%H%M%S)
echo "========================================================"
echo "Resuming at:    ${SAFE_TS}"
echo "Run root:       ${RUN_ROOT}"
echo "Stage 2 ckpt:   ${STAGE2_CKPT_DIR}"
echo "========================================================"

# Move aside any existing e2e/eval from the failed run so the new run starts clean.
for sub in e2e eval; do
    if [ -d "${RUN_ROOT}/${sub}" ]; then
        BACKUP="${RUN_ROOT}/${sub}.failed_${SAFE_TS}"
        echo "Moving old ${sub}/ aside -> ${BACKUP}"
        mv "${RUN_ROOT}/${sub}" "${BACKUP}"
    fi
done

# Keep a copy of this resume script alongside the original pipeline script.
cp "$0" "${RUN_ROOT}/"

# ============================================================================
# Stage 3 — E2E training, starting from the existing Stage 2 checkpoint.
# (Skipping Stage 2.5, same as the original pipeline.)
# ============================================================================
E2E_DATA="data/fusionad_results_for_vlm/train.json"
E2E_EXP_NAME="e2e"
E2E_OUT_DIR="${RUN_ROOT}/e2e"
E2E_RUN_NAME="fusionad-Qwen2.5-0.5B-${E2E_EXP_NAME}"
E2E_CKPT_DIR="${E2E_OUT_DIR}/${E2E_RUN_NAME}"
mkdir -p ${E2E_OUT_DIR}/log

MASTER_PORT_S3=$(shuf -i 20000-60000 -n 1)

echo ""
echo "=========================================================="
echo ">>> [Stage 3 / E2E] resuming from Stage 2 ckpt"
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
# Eval — inference on val + eval_drivevla metrics.
# ============================================================================
EVAL_DIR="${RUN_ROOT}/eval"
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

torchrun --nproc_per_node=4 --master_port=${MASTER_PORT_EVAL} \
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
echo "Resume pipeline done."
echo "  Run root:  ${RUN_ROOT}"
echo "  E2E ckpt:  ${E2E_CKPT_DIR}"
echo "  Eval log:  ${EVAL_LOG}"
echo "========================================================"
