#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DEV="${ROOT}/op_dev"
BUILD_ROOT="${ROOT}/build"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${ROOT}/_custom_opp"
TORCH_EXTENSION="${ROOT}/torch_extension"

PYTHON_BIN="${OPS_DSA_A5_PYTHON:-python3}"
BUILD_JOBS="${OPS_DSA_A5_BUILD_JOBS:-$(nproc)}"
# The CANN 9.1 AscendC toolchain exposes one generic A5 compute unit:
# ascend950. Normalize 950PR/950DT product names so an old environment value
# cannot turn into the unsupported CMake target ascend950pr_* / ascend950dt_*.
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-ascend950}"
case "${A5_SOC_VERSION_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*)
        A5_SOC_VERSION="ascend950"
        ;;
    *)
        echo "[ops_dsa_offload_a5] ERROR: this experiment only supports Ascend 950; got A5_SOC_VERSION=${A5_SOC_VERSION_RAW}." >&2
        exit 2
        ;;
esac

if ! command -v msopgen >/dev/null 2>&1; then
    echo "[ops_dsa_offload_a5] ERROR: msopgen is unavailable; source the CANN environment first." >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ops_dsa_offload_a5] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

case "${GENERATED}" in
    "${ROOT}"/*) ;;
    *)
        echo "[ops_dsa_offload_a5] ERROR: unsafe generated-project path: ${GENERATED}" >&2
        exit 2
        ;;
esac
case "${LOCAL_OPP}" in
    "${ROOT}"/*) ;;
    *)
        echo "[ops_dsa_offload_a5] ERROR: unsafe local OPP path: ${LOCAL_OPP}" >&2
        exit 2
        ;;
esac

echo "[ops_dsa_offload_a5] root: ${ROOT}"
echo "[ops_dsa_offload_a5] Python: ${PYTHON_BIN}"
echo "[ops_dsa_offload_a5] A5 target: ${A5_SOC_VERSION}"
if [[ "${A5_SOC_VERSION_RAW,,}" != "${A5_SOC_VERSION}" ]]; then
    echo "[ops_dsa_offload_a5] normalized product name ${A5_SOC_VERSION_RAW} -> ${A5_SOC_VERSION}"
fi
echo "[ops_dsa_offload_a5] jobs: ${BUILD_JOBS}"

OP_NAMES=(
    A5KvcacheScatterCopy
)

rm -rf "${GENERATED}"
mkdir -p "${BUILD_ROOT}"
for index in "${!OP_NAMES[@]}"; do
    op_name="${OP_NAMES[index]}"
    echo "[ops_dsa_offload_a5] generate operator: ${op_name}"
    msopgen_args=(
        gen
        -i "${OP_DEV}/ops.json"
        -f aclnn
        -c "ai_core-${A5_SOC_VERSION}"
        -lan cpp
        -op "${op_name}"
        -out "${GENERATED}"
    )
    if (( index > 0 )); then
        msopgen_args+=(-m 1)
    fi
    msopgen "${msopgen_args[@]}"
done

# The generated project owns only its build scaffold. The checked-in host and
# kernel sources below are the authoritative implementation.
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"

pushd "${GENERATED}" >/dev/null
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
popd >/dev/null

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
    echo "[ops_dsa_offload_a5] ERROR: msopgen project did not produce a .run package." >&2
    exit 1
fi

rm -rf "${LOCAL_OPP}"
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

mapfile -t OPAPI_LIBS < <(find "${LOCAL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so')
if [[ "${#OPAPI_LIBS[@]}" -ne 1 ]]; then
    echo "[ops_dsa_offload_a5] ERROR: expected one local libcust_opapi.so, found ${#OPAPI_LIBS[@]}." >&2
    exit 1
fi
VENDOR_DIR="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd)"

for op_name in "${OP_NAMES[@]}"; do
    if ! find "${VENDOR_DIR}" -type f -name 'binary_info_config.json' -print0 |
        xargs -0 -r grep -q "${op_name}"; then
        echo "[ops_dsa_offload_a5] ERROR: ${op_name} is absent from generated kernel metadata." >&2
        exit 1
    fi
done

pushd "${TORCH_EXTENSION}" >/dev/null
rm -rf build
rm -f ops_dsa_offload_a5/_C*.so
MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export OPS_DSA_OFFLOAD_A5_INSTALL_OPP_PATH="${LOCAL_OPP}"
PYTHONPATH="${TORCH_EXTENSION}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
    "import ops_dsa_offload_a5; print('[ops_dsa_offload_a5] torch extension import: OK')"

echo "[ops_dsa_offload_a5] build complete"
echo "[ops_dsa_offload_a5] local vendor: ${VENDOR_DIR}"
