EXPERIMENT_DIR=$1
SAVE_CKPT_DIR=$2
EXPERIMENT_NAME=$3
EVAL_DATA_PATH=$4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_RESULT_DIR="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p ${LOG_RESULT_DIR}/log
mkdir -p ${LOG_RESULT_DIR}/results
cp $0 ${LOG_RESULT_DIR}
EVAL_LOG_FILE="${LOG_RESULT_DIR}/log/eval.log"

echo ">>> Experiment Configuration:" | tee -a ${EVAL_LOG_FILE}
echo "- LOG_RESULT_DIR: ${LOG_RESULT_DIR}" | tee -a ${EVAL_LOG_FILE}
echo "- EXPERIMENT_NAME: ${EXPERIMENT_NAME}" | tee -a ${EVAL_LOG_FILE}
echo "- EVAL_DATA_PATH: ${EVAL_DATA_PATH}" | tee -a ${EVAL_LOG_FILE}
echo "- SAVE_CKPT_DIR: ${SAVE_CKPT_DIR}" | tee -a ${EVAL_LOG_FILE}
echo "----------------------------------------" | tee -a ${EVAL_LOG_FILE}

# ----------------------------------------------------------------

echo ">>> Inference ${SAVE_CKPT_DIR} (epoch-1)..." | tee -a ${EVAL_LOG_FILE}

mkdir -p ${LOG_RESULT_DIR}/results/epoch-1
PLAN_CONV_VAL_FILE_PATH="${LOG_RESULT_DIR}/results/epoch-1/planning_conversations_val.json"

torchrun --nproc_per_node=4 \
    drivevla/inference_drivevla.py \
    --bf16 \
    --model-path ${SAVE_CKPT_DIR} \
    --output ${PLAN_CONV_VAL_FILE_PATH} \
    --data ${EVAL_DATA_PATH} \
    2>&1 | tee -a ${EVAL_LOG_FILE}

echo ">>> Evaluating ${PLAN_CONV_VAL_FILE_PATH}..." | tee -a ${EVAL_LOG_FILE}

python drivevla/eval_drivevla.py \
    --output ${PLAN_CONV_VAL_FILE_PATH} \
    2>&1 | tee -a ${EVAL_LOG_FILE}
