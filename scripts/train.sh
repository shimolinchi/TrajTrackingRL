#!/usr/bin/env bash
# User configuration
python_bin="python"
recipe="src/trajectory_tracking/resources/baseline.yaml"
contract="tmp/prepared/contract.json"
gpu_ids=(0 1 2 3 4 5 6 7)
envs_per_gpu=8192
max_updates=20000
interval=100
browser_port=8480
html_port=8481
wandb="disabled"
resume_checkpoint=""
checkpoint=""
# End user configuration
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit the user block and run without arguments." >&2; exit 2; }
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
mkdir -p tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export TRAJTRACKING_WORKSPACE="$root"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [[ -f configs/local.env ]]; then set -a; source configs/local.env; set +a; fi
gpu_csv="$(IFS=,; echo "${gpu_ids[*]}")"
flags=()
[[ -z "$resume_checkpoint" ]] || flags+=(--resume-checkpoint "$resume_checkpoint")
[[ -z "$checkpoint" ]] || flags+=(--checkpoint "$checkpoint")
exec "$python_bin" -m trajectory_tracking.cli.launch --mode train --config "$recipe" --contract "$contract" \
  --gpus "$gpu_csv" --per-gpu-envs "$envs_per_gpu" --max-updates "$max_updates" --interval "$interval" \
  --browser-port "$browser_port" --html-port "$html_port" --wandb "$wandb" "${flags[@]}"
