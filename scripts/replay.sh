#!/usr/bin/env bash
# User configuration
python_bin="python"
case_directory="logs/your_run/evaluation_videos/html_latest_cases"
port=8481
# End user configuration
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit the user block and run without arguments." >&2; exit 2; }
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
mkdir -p tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "$python_bin" -m trajectory_tracking.visualization.state_replay.failure_html_server \
  --data "$case_directory" --port "$port"
