#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCENDC_DIR="${ROOT}/csrc"
TORCH_DIR="${ROOT}/torch_extension"
LOCAL_OPP="${ROOT}/_custom_opp"
PYTHON_BIN="${OPS_OVERLAP_PYTHON:-python3}"
RAW_SOC="${SOC_VERSION:-ascend910_9391}"
CANN_INSTALL_PATH="${CANN_INSTALL_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}}"
BUILD_JOBS="${OPS_OVERLAP_BUILD_JOBS:-$(nproc)}"
OP_LIST="kvcache_scatter_copy"

case "${RAW_SOC}" in
  ascend910_9391)
    SOC="ascend910_93"
    OPS_OVERLAP_OPC_SOC_VERSION="${OPS_OVERLAP_OPC_SOC_VERSION:-}"
    ;;
  ascend910b1)
    SOC="ascend910b"
    OPS_OVERLAP_OPC_SOC_VERSION="${OPS_OVERLAP_OPC_SOC_VERSION:-Ascend910B1}"
    ;;
  ascend910b3)
    SOC="ascend910b"
    OPS_OVERLAP_OPC_SOC_VERSION="${OPS_OVERLAP_OPC_SOC_VERSION:-Ascend910B3}"
    ;;
  *)
    SOC="${RAW_SOC}"
    OPS_OVERLAP_OPC_SOC_VERSION="${OPS_OVERLAP_OPC_SOC_VERSION:-}"
    ;;
esac

if [[ ! -d "${CANN_INSTALL_PATH}" ]]; then
  echo "[ops_overlap] ERROR: CANN path does not exist: ${CANN_INSTALL_PATH}" >&2
  exit 2
fi

export CANN_INSTALL_PATH
export ASCEND_HOME_PATH="${CANN_INSTALL_PATH}"
export OPS_CPU_NUMBER="${BUILD_JOBS}"
export OPS_OVERLAP_OPC_SOC_VERSION

echo "[ops_overlap] root: ${ROOT}"
echo "[ops_overlap] python: ${PYTHON_BIN}"
echo "[ops_overlap] CANN: ${CANN_INSTALL_PATH}"
echo "[ops_overlap] SoC: raw=${RAW_SOC}, build=${SOC}"
echo "[ops_overlap] jobs: ${BUILD_JOBS}"
echo "[ops_overlap] operator: kvcache_scatter_copy"

pushd "${ASCENDC_DIR}" >/dev/null
bash build.sh -n "${OP_LIST}" -c "${SOC}" -p "${CANN_INSTALL_PATH}"
popd >/dev/null

RUN_PKG="$(find "${ASCENDC_DIR}/output" -maxdepth 1 -name 'CANN-custom_ops-*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
  echo "[ops_overlap] ERROR: custom-op .run package was not generated." >&2
  exit 1
fi

case "${LOCAL_OPP}" in
  "${ROOT}"/*) ;;
  *)
    echo "[ops_overlap] ERROR: refusing to replace unsafe local OPP path: ${LOCAL_OPP}" >&2
    exit 1
    ;;
esac
rm -rf "${LOCAL_OPP}"
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

VENDOR="${LOCAL_OPP}/vendors/ops-overlap"
OPAPI="${VENDOR}/op_api/lib/libcust_opapi.so"
BINARY_INFO="${VENDOR}/op_impl/ai_core/tbe/kernel/config/${SOC}/binary_info_config.json"

if [[ ! -f "${OPAPI}" ]]; then
  echo "[ops_overlap] ERROR: missing ${OPAPI}" >&2
  exit 1
fi
if [[ ! -f "${BINARY_INFO}" ]]; then
  echo "[ops_overlap] ERROR: missing ${BINARY_INFO}" >&2
  exit 1
fi

for symbol in aclnnNanovllmKvcacheScatterCopy; do
  if ! nm -D "${OPAPI}" | grep -q "${symbol}"; then
    echo "[ops_overlap] ERROR: ${symbol} is absent from ${OPAPI}" >&2
    exit 1
  fi
  echo "[ops_overlap] verified opapi symbol: ${symbol}"
done

for op_type in NanovllmKvcacheScatterCopy; do
  if ! grep -q "${op_type}" "${BINARY_INFO}"; then
    echo "[ops_overlap] ERROR: ${op_type} is absent from ${BINARY_INFO}" >&2
    exit 1
  fi
  echo "[ops_overlap] verified kernel registration: ${op_type}"
done

pushd "${TORCH_DIR}" >/dev/null
rm -rf build
rm -f ops_overlap/_C*.so
MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

export ASCEND_CUSTOM_OPP_PATH="${VENDOR}"
export OPS_OVERLAP_INSTALL_OPP_PATH="${LOCAL_OPP}"
PYTHONPATH="${TORCH_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
  "import ops_overlap; print('[ops_overlap] torch extension import: OK')"

echo "[ops_overlap] build complete"
