#!/usr/bin/env bash

set -euo pipefail

unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export CANN_INSTALL_PATH=${ASCEND_HOME_PATH}
source "${ASCEND_HOME_PATH}/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=${A5_TEST_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/torch_extension${PYTHONPATH:+:${PYTHONPATH}}"
export SOC_VERSION=ascend950
export ASCEND_CUSTOM_OPP_PATH="${PWD}/_custom_opp_bf16/vendors/customize"
export NANOVLLM_A5_INSTALL_OPP_PATH="${PWD}/_custom_opp_bf16"
export NANOVLLM_CUST_OPAPI_LIB="${PWD}/_custom_opp_bf16/vendors/customize/op_api/lib/libcust_opapi.so"

RESULTS_DIR=${RESULTS_DIR:-results/kvcache_scatter_copy_multiprocess}
CARD_COUNTS=${CARD_COUNTS:-"8 4 1"}
BATCH_SIZES=${BATCH_SIZES:-"8 32"}
COPY_COUNTS=${COPY_COUNTS:-"100 200 300 500 2048"}
DTYPES=${DTYPES:-"bf16"}
SOURCE_LEN=${SOURCE_LEN:-65536}
HBM_SLOTS=${HBM_SLOTS:-8192}
COPY_CAP=${COPY_CAP:-2048}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-1000}
SEED=${SEED:-7}

mkdir -p "${RESULTS_DIR}"

for card_count in ${CARD_COUNTS}; do
  devices=$(seq -s, 0 $((card_count - 1)))
  for dtype in ${DTYPES}; do
    for batch_size in ${BATCH_SIZES}; do
      for copy_count in ${COPY_COUNTS}; do
        case_name="cards${card_count}_${dtype}_bs${batch_size}_copy${copy_count}"
        echo "Running ${case_name} on logical devices ${devices}"
        python3 tests/test_kvcache_scatter_copy_multiprocess.py \
          --devices "${devices}" \
          --dtype "${dtype}" \
          --batch-size "${batch_size}" \
          --source-len "${SOURCE_LEN}" \
          --hbm-slots "${HBM_SLOTS}" \
          --copy-min "${copy_count}" \
          --copy-max "${copy_count}" \
          --copy-cap "${COPY_CAP}" \
          --warmup "${WARMUP}" \
          --iters "${ITERS}" \
          --seed "${SEED}" \
          --same-seed \
          --output "${RESULTS_DIR}/${case_name}.json" \
          2>&1 | tee "${RESULTS_DIR}/${case_name}.log"
      done
    done
  done
done

python3 tests/analyze_kvcache_scatter_copy_results.py \
  --results-dir "${RESULTS_DIR}"

python3 tests/format_kvcache_scatter_copy_csv.py \
  --input "${RESULTS_DIR}/kvcache_scatter_copy_timing_summary.csv"
