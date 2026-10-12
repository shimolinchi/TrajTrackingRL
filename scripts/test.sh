#!/usr/bin/env bash
# User configuration
python_bin="python"
# End user configuration
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit the user block and run without arguments." >&2; exit 2; }
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
mkdir -p tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ -f configs/local.env ]]; then set -a; source configs/local.env; set +a; fi
exec "$python_bin" -m pytest
