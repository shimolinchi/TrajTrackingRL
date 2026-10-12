#!/usr/bin/env bash
# User configuration
python_bin="python"
workers=4
certify=false
physical_gpu=0
# End user configuration
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit the user block and run without arguments." >&2; exit 2; }
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
mkdir -p tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export TRAJTRACKING_WORKSPACE="$root"
if [[ -f configs/local.env ]]; then set -a; source configs/local.env; set +a; fi
flags=()
[[ "$certify" == false ]] || flags+=(--certify --physical-gpu "$physical_gpu")
exec "$python_bin" -m trajectory_tracking.cli.prepare \
  --dataset "$TRAJTRACKING_DATASET_ROOT" --asset-root "$TRAJTRACKING_OBJECT_ASSET_ROOT" \
  --cache-root logs/trajectory_data/automatic --report tmp/preparation.json \
  --workers "$workers" --invalid error "${flags[@]}"
