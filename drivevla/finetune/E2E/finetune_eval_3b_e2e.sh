# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
TUNABLE_PARTS="mm_vision_tower,mm_mlp_adapter,mm_language_model"
TUNABLE_PARTS_CLEAN="${TUNABLE_PARTS//\//_}"
VISION_MODEL_VERSION="uniad_track_map"

CKPT_PATH="output/0_pretrain/3B_2/stage3_object_agent_trajectory_on_QA_20250307_143246/uniad-Qwen_Qwen2.5-3B-Instruct-mm_mlp_adapter,mm_language_model-stage3_object_agent_trajectory_on_QA"
CKPT_PATH_CLEAN="Qwen_Qwen2.5-3B-Instruct"

PROMPT_VERSION="qwen_planning_oriented_vlm"

EXPERIMENT_NAME="E2E"
TRAIN_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp1_scene_tracking_map_ego_history_command/traj_converted_data_train.json"

BASE_RUN_NAME="uniad-${CKPT_PATH_CLEAN}-${TUNABLE_PARTS_CLEAN}-${EXPERIMENT_NAME}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

TIMESTAMP_BASE=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="output/0_ablation/3B_E2E/${TIMESTAMP_BASE}"
LOG_RESULT_DIR="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP_BASE}"
mkdir -p ${LOG_RESULT_DIR}/log
mkdir -p ${LOG_RESULT_DIR}/results

cp $0 ${LOG_RESULT_DIR}

TRAIN_LOG_FILE="${LOG_RESULT_DIR}/log/train.log"
EVAL_LOG_FILE="${LOG_RESULT_DIR}/log/eval.log"

SAVE_CKPT_DIR="${LOG_RESULT_DIR}/${BASE_RUN_NAME}"

echo ">>> Experiment Configuration:" | tee -a ${TRAIN_LOG_FILE}
echo "- LOG_RESULT_DIR: ${LOG_RESULT_DIR}" | tee -a ${TRAIN_LOG_FILE}
echo "- EXPERIMENT_NAME: ${EXPERIMENT_NAME}" | tee -a ${TRAIN_LOG_FILE}
echo "- TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}" | tee -a ${TRAIN_LOG_FILE}
echo "- CKPT_PATH: ${CKPT_PATH}" | tee -a ${TRAIN_LOG_FILE}
echo "- BASE_RUN_NAME: ${BASE_RUN_NAME}" | tee -a ${TRAIN_LOG_FILE}
echo "- SAVE_CKPT_DIR: ${SAVE_CKPT_DIR}" | tee -a ${TRAIN_LOG_FILE}
echo "----------------------------------------" | tee -a ${TRAIN_LOG_FILE}

    # -m debugpy --listen 5678 --wait-for-client \
ACCELERATE_CPU_AFFINITY=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 torchrun --nproc_per_node=4 \
    llava/train/train_DriveVLA.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${TRAIN_DATA_PATH} \
    --image_folder . \
    --mm_tunable_parts=${TUNABLE_PARTS_CLEAN} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 256 \
    --vision_tower_pretrained checkpoints/uniad_track_map \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir ${SAVE_CKPT_DIR} \
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
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --in_nuscenes_order True \
    --use_uniad_pth False \
    2>&1 | tee -a ${TRAIN_LOG_FILE}

    # --resume_from_checkpoint "checkpoints/${BASE_RUN_NAME}/checkpoint-24612" \

# You can delete the sdpa attn_implementation if you want to use flash attn

# ----------------------------------------------------------------
# Run ablation evaluation script

EXPERIMENT_NAME="Exp1_scene_tracking_map_ego_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp1_scene_tracking_map_ego_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp2_tracking_map_ego_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp2_tracking_map_ego_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp3_scene_map_ego_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp3_scene_map_ego_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp4_scene_tracking_ego_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp4_scene_tracking_ego_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp5_scene_tracking_map_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp5_scene_tracking_map_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp6_scene_tracking_map_ego_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp6_scene_tracking_map_ego_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp7_scene_tracking_map_ego_history"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp7_scene_tracking_map_ego_history/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}

# ----------------------------------------------------------------

EXPERIMENT_NAME="Exp8_ego_history_command"
EVAL_DATA_PATH="data/DriveVLA/experiments/trajectory_experiments_25Feb27/Exp8_ego_history_command/traj_converted_data_val.json"

bash $(dirname $0)/ablation_eval_e2e.sh \
    ${EXPERIMENT_DIR} \
    ${SAVE_CKPT_DIR} \
    ${EXPERIMENT_NAME} \
    ${EVAL_DATA_PATH}
