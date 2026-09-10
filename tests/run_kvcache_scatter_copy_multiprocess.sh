#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_scatter_copy_multiprocess.sh" "$@"
