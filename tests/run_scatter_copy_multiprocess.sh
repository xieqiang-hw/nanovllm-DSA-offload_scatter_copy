#!/usr/bin/env bash
set -euo pipefail
SCATTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCATTER_ROOT}"
SCATTER_PYTHON=${PYTHON_BIN:-${NANOVLLM_A5_OPS_PYTHON:-python3}}
export TEST_VISIBLE_DEVICES=${TEST_VISIBLE_DEVICES:-${A3_TEST_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}}
# Help and validation work on development hosts without CANN or Torch.
SCATTER_PREVIEW=${DRY_RUN:-0}
for SCATTER_ARG in "$@"; do
  case "${SCATTER_ARG}" in --dry-run|--help|-h) SCATTER_PREVIEW=1 ;; esac
done
if [[ "${SCATTER_PREVIEW}" != 1 ]]; then
  export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}
  export CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-${ASCEND_HOME_PATH}}
  if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    set +u
    source "${ASCEND_HOME_PATH}/set_env.sh"
    set -u
  fi
  unset ASCEND_CUSTOM_OPP_PATH OPS_OVERLAP_INSTALL_OPP_PATH
  export SOC_VERSION=ascend910_9391
  export ASCEND_LAUNCH_BLOCKING=0
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
fi
export ASCEND_RT_VISIBLE_DEVICES=${TEST_VISIBLE_DEVICES}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCATTER_ROOT}/torch_extension${PYTHONPATH:+:${PYTHONPATH}}"
exec "${SCATTER_PYTHON}" "${SCATTER_ROOT}/tests/run_scatter_copy_sweep.py" "$@"
