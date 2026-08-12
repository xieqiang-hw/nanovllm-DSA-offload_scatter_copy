unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-9.0.1
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-results/multiprocess}
SOURCE_LEN=${SOURCE_LEN:-65536}
HBM_SLOTS=${HBM_SLOTS:-8192}
COPY_CAP=${COPY_CAP:-2048}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-1000}
SEED=${SEED:-7}

mkdir -p "${RESULTS_DIR}"

for card_count in 1 2 4 8 16; do
  devices=$(seq -s, 0 $((card_count - 1)))
  for batch_size in 8 24; do
    for copy_count in 0 100 200 300 500 2048; do
      case_name="cards${card_count}_bs${batch_size}_copy${copy_count}"
      output_path="${RESULTS_DIR}/${case_name}.json"
      log_path="${RESULTS_DIR}/${case_name}.log"
      echo "Running multiprocess ${case_name} on devices ${devices}"

      python3 tests/test_scatter_copy_multiprocess.py \
        --devices "${devices}" \
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
