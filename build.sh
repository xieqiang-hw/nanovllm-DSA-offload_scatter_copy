#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
JOBS="${MAX_JOBS:-$(nproc)}"
SOC="$("${PYTHON}" "${ROOT}/torch_extension/kvcache_ops/_soc.py")"
BUILD="${ROOT}/build/${SOC}"
GENERATED="${BUILD}/custom_op"
OPP="${BUILD}/opp"
"${PYTHON}" "${ROOT}/tests/check_build.py" --definition "${ROOT}/csrc/ops.json"
command -v msopgen >/dev/null || { echo "Source the CANN development environment first." >&2; exit 2; }
echo "Building KvcacheScatterCopy (BF16 + C8), target=${SOC}, jobs=${JOBS}"
mkdir -p "${BUILD}"
rm -rf "${GENERATED}"
msopgen gen -i "${ROOT}/csrc/ops.json" -f aclnn -c "ai_core-${SOC}" \
    -lan cpp -op KvcacheScatterCopy -out "${GENERATED}"
for part in op_host op_kernel; do
    cp -a "${ROOT}/csrc/src/kvcache_scatter_copy/${part}/." "${GENERATED}/${part}/"
    {
        printf '#pragma once\n#define SCATTER_SOC "%s"\n' "${SOC}"
        if [[ "${SOC}" == ascend950 ]]; then
            printf '#define SCATTER_A5 1\n'
        else
            printf '#define SCATTER_A5 0\n'
        fi
    } > "${GENERATED}/${part}/scatter_target.h"
done
(cd "${GENERATED}" && OPS_CPU_NUMBER="${JOBS}" bash build.sh)
mapfile -t PACKAGES < <(find "${GENERATED}/build_out" -maxdepth 1 -name '*.run' -type f)
[[ ${#PACKAGES[@]} -eq 1 ]] || { echo "Expected one operator package." >&2; exit 1; }
rm -rf "${OPP}"
mkdir -p "${OPP}"
bash "${PACKAGES[0]}" --quiet --install-path="${OPP}"
mapfile -t LIBS < <(find "${OPP}/vendors" -path '*/op_api/lib/libcust_opapi.so' -type f)
[[ ${#LIBS[@]} -eq 1 ]] || { echo "Expected one local libcust_opapi.so." >&2; exit 1; }
# Verify the generated reference ABI rather than guessing an output-argument count.
"${PYTHON}" "${ROOT}/tests/check_build.py" "${OPP}"
(cd "${ROOT}/torch_extension" &&
    rm -f kvcache_ops/_C*.so &&
    MAX_JOBS="${JOBS}" "${PYTHON}" setup.py build_ext --inplace --build-temp "${BUILD}/torch_extension")
echo "Build complete: ${OPP}"
