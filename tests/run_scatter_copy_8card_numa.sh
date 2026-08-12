#!/usr/bin/env bash

# Temporary fixed-topology runner for the 8-card Ascend 950 host:
# NPU 0,1,6,7 -> NUMA node 3 (CPUs 288-383)
# NPU 2,3,4,5 -> NUMA node 1 (CPUs 96-191)

unset ASCEND_CUSTOM_OPP_PATH
unset OPS_DSA_OFFLOAD_A5_INSTALL_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
unset OPS_OVERLAP_BUILD_JOBS
unset OPS_OVERLAP_PYTHON
unset SOC_VERSION
unset CANN_INSTALL_PATH
unset IGNORE_INFER_ERROR

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:${PYTHONPATH:-}

set -euo pipefail

command -v numactl >/dev/null || {
  echo "ERROR: numactl is required for CPU and memory NUMA binding." >&2
  exit 1
}

RESULTS_DIR=${RESULTS_DIR:-results/a5_8card_numa}
BATCH_SIZES=${BATCH_SIZES:-"8 32"}
COPY_COUNTS=${COPY_COUNTS:-"0 100 200 300 500 2048"}
DTYPES=${DTYPES:-"bf16"}
SOURCE_LEN=${SOURCE_LEN:-65536}
HBM_SLOTS=${HBM_SLOTS:-8192}
COPY_CAP=${COPY_CAP:-2048}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-1000}
SEED=${SEED:-7}

DEVICES=0,1,2,3,4,5,6,7
NUMA_MAP=0:3,1:3,2:1,3:1,4:1,5:1,6:3,7:3

mkdir -p "${RESULTS_DIR}"

for dtype in ${DTYPES}; do
  for batch_size in ${BATCH_SIZES}; do
    for copy_count in ${COPY_COUNTS}; do
      case_name="cards8_numa_${dtype}_bs${batch_size}_copy${copy_count}"
      output_path="${RESULTS_DIR}/${case_name}.json"
      log_path="${RESULTS_DIR}/${case_name}.log"
      echo "Running A5 ${case_name}; NUMA map: ${NUMA_MAP}"

      python3 tests/test_scatter_copy_multiprocess.py \
        --devices "${DEVICES}" \
        --numa-map "${NUMA_MAP}" \
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
        --output "${output_path}" 2>&1 | tee "${log_path}"
    done
  done
done

python3 tests/analyze_scatter_copy_results.py --results-dir "${RESULTS_DIR}"
