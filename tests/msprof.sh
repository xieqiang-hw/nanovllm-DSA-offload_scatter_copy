#!/usr/bin/env bash

set -euo pipefail

unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-9.0.1
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:${PYTHONPATH:-}

RESULTS_DIR=${RESULTS_DIR:-results/msprof_copy200_1000iters}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
BATCH_SIZE=${BATCH_SIZE:-24}
SOURCE_LEN=${SOURCE_LEN:-65536}
HBM_SLOTS=${HBM_SLOTS:-8192}
COPY_COUNT=${COPY_COUNT:-200}
COPY_CAP=${COPY_CAP:-2048}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-1000}
SEED=${SEED:-7}

mkdir -p "${RESULTS_DIR}"

CARD_COUNT=$(awk -F, '{print NF}' <<<"${DEVICES}")
CASE_NAME="cards${CARD_COUNT}_bs${BATCH_SIZE}_copy${COPY_COUNT}"
JSON_OUTPUT="${RESULTS_DIR}/${CASE_NAME}.json"
LOG_OUTPUT="${RESULTS_DIR}/${CASE_NAME}.log"
PROFILE_DIR="${RESULTS_DIR}/msprof_${CASE_NAME}"

echo "Running ${CASE_NAME}"
echo "Devices: ${DEVICES}"
echo "Profile: ${PROFILE_DIR}"

msprof --output="${PROFILE_DIR}" \
  python3 tests/test_scatter_copy_multiprocess.py \
    --devices "${DEVICES}" \
    --batch-size "${BATCH_SIZE}" \
    --source-len "${SOURCE_LEN}" \
    --hbm-slots "${HBM_SLOTS}" \
    --copy-min "${COPY_COUNT}" \
    --copy-max "${COPY_COUNT}" \
    --copy-cap "${COPY_CAP}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --seed "${SEED}" \
    --same-seed \
    --output "${JSON_OUTPUT}" \
    2>&1 | tee "${LOG_OUTPUT}"

echo "Result: ${JSON_OUTPUT}"
echo "Profile: ${PROFILE_DIR}"
