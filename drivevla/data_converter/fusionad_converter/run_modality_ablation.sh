#!/bin/bash
# Modality ablation for FusionAD:
#   Pass 1 — camera-only  (lidar features zeroed, --zero-lidar)
#   Pass 2 — lidar-only   (camera features zeroed, --zero-camera)
# Then eval both result sets with eval_from_pth.py.
#
# Features are zeroed post-backbone inside FusionAD.get_bevs() — see
# projects/fusionad_plugin_new/fusionad/detectors/fusionad_track.py.
# Flags set on the model via attributes from pickle_fusionad_eval.py.
#
# Baseline (un-ablated) inference is NOT run here — assumes
# data/fusionad_eval_results/${SPLIT}/ already exists, or is produced
# separately by running pickle_fusionad_eval.py without --zero-* flags.
#
# Env-var overrides:
#   NPROC          (default 8)
#   SPLIT          (default val)
#   CKPT           (default /home/s56cai/ckpt/fusionad/fusion_latest.pth)
#   BASE_OUT       (default data/fusionad_eval_results; suffix _no_lidar /
#                   _no_camera is appended automatically by the py script)
#   EVAL_OUT_ROOT  (default output/fusionad_modality_ablation/<timestamp>)

set -e
set -o pipefail

cd /home/s56cai/OpenDriveVLA
export PYTHONPATH=/home/s56cai/OpenDriveVLA:$PYTHONPATH

NPROC=${NPROC:-8}
SPLIT=${SPLIT:-val}
CKPT=${CKPT:-/home/s56cai/ckpt/fusionad/fusion_latest.pth}
BASE_OUT=${BASE_OUT:-data/fusionad_eval_results}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-output/fusionad_modality_ablation/${TIMESTAMP}}

mkdir -p ${EVAL_OUT_ROOT}/log
cp "$0" "${EVAL_OUT_ROOT}/"

echo "=========================================================="
echo "FusionAD modality ablation"
echo "  timestamp:     ${TIMESTAMP}"
echo "  nproc:         ${NPROC}"
echo "  split:         ${SPLIT}"
echo "  ckpt:          ${CKPT}"
echo "  base out:      ${BASE_OUT}"
echo "  eval out root: ${EVAL_OUT_ROOT}"
echo "=========================================================="

# --- Pass 1: camera-only (lidar blanked) ---------------------------------
MASTER_PORT_1=$(shuf -i 20000-60000 -n 1)
echo ""
echo ">>> [Pass 1] camera-only inference  (--zero-lidar)"
echo "    port: ${MASTER_PORT_1}"
torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT_1} \
    drivevla/data_converter/fusionad_converter/pickle_fusionad_eval.py \
    --nuscenes_set ${SPLIT} \
    --ckpt ${CKPT} \
    --out_dir ${BASE_OUT} \
    --zero-lidar \
    2>&1 | tee ${EVAL_OUT_ROOT}/log/infer_camera_only.log

# --- Pass 2: lidar-only (camera blanked) ---------------------------------
MASTER_PORT_2=$(shuf -i 20000-60000 -n 1)
echo ""
echo ">>> [Pass 2] lidar-only inference  (--zero-camera)"
echo "    port: ${MASTER_PORT_2}"
torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT_2} \
    drivevla/data_converter/fusionad_converter/pickle_fusionad_eval.py \
    --nuscenes_set ${SPLIT} \
    --ckpt ${CKPT} \
    --out_dir ${BASE_OUT} \
    --zero-camera \
    2>&1 | tee ${EVAL_OUT_ROOT}/log/infer_lidar_only.log

# --- Eval: camera-only ----------------------------------------------------
#   eval_from_pth.py joins ${pth_dir}/${split} internally, so pass the
#   suffix base (no trailing /${SPLIT}).
echo ""
echo ">>> [Eval] camera-only metrics"
mkdir -p ${EVAL_OUT_ROOT}/camera_only
python drivevla/data_converter/fusionad_converter/eval_from_pth.py \
    --nuscenes_set ${SPLIT} \
    --pth_dir ${BASE_OUT}_no_lidar \
    --jsonfile_prefix ${EVAL_OUT_ROOT}/camera_only \
    --metric bbox \
    2>&1 | tee ${EVAL_OUT_ROOT}/log/eval_camera_only.log

# --- Eval: lidar-only -----------------------------------------------------
echo ""
echo ">>> [Eval] lidar-only metrics"
mkdir -p ${EVAL_OUT_ROOT}/lidar_only
python drivevla/data_converter/fusionad_converter/eval_from_pth.py \
    --nuscenes_set ${SPLIT} \
    --pth_dir ${BASE_OUT}_no_camera \
    --jsonfile_prefix ${EVAL_OUT_ROOT}/lidar_only \
    --metric bbox \
    2>&1 | tee ${EVAL_OUT_ROOT}/log/eval_lidar_only.log

echo ""
echo "=========================================================="
echo "Modality ablation done."
echo "  camera-only pths:    ${BASE_OUT}_no_lidar/${SPLIT}/"
echo "  lidar-only  pths:    ${BASE_OUT}_no_camera/${SPLIT}/"
echo "  camera-only metrics: ${EVAL_OUT_ROOT}/camera_only/  (log: ${EVAL_OUT_ROOT}/log/eval_camera_only.log)"
echo "  lidar-only  metrics: ${EVAL_OUT_ROOT}/lidar_only/   (log: ${EVAL_OUT_ROOT}/log/eval_lidar_only.log)"
echo "=========================================================="
