"""Small public launcher; all algorithm settings come from the baseline recipe."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml

from .preflight import CONFIG, ROOT, verify, gpu_plan
from ..configuration import load_trajectory_direct_config, resolve_trajectory_direct_config

def validate_baseline(config):
    """Permit a new dataset, while fixing the released algorithm."""
    expected = resolve_trajectory_direct_config(yaml.safe_load(CONFIG.read_text()))
    actual, baseline = asdict(config), asdict(expected)
    actual.pop("data"); baseline.pop("data")
    if actual != baseline or config.data.adapter != "tro_success_continuous":
        raise ValueError("this release supports only the frozen delta-action baseline")

def exclusive_gpus(indices):
    root = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15))
    by_uuid = {g.findtext("uuid"): g for g in root.findall("gpu")}
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
        "--format=csv,noheader"], text=True, timeout=15)
    gpus = {int(row.split(",")[0]): by_uuid[row.split(",")[1].strip()]
            for row in rows.splitlines()}
    selected = []
    for index in indices:
        if index not in gpus:
            raise ValueError(f"unknown physical GPU {index}")
        gpu = gpus[index]
        if gpu.findall("processes/process_info"):
            raise ValueError(f"physical GPU {index} has active compute/graphics processes")
        used = int(gpu.findtext("fb_memory_usage/used").split()[0])
        utilization = int(gpu.findtext("utilization/gpu_util").split()[0])
        if used > 128 or utilization != 0:
            raise ValueError(f"physical GPU {index} is not idle")
        selected.append(dict(index=index, uuid=gpu.findtext("uuid")))
    return selected

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "evaluate", "preflight"), default="train")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--per-gpu-envs", type=int, default=8192)
    p.add_argument("--max-updates", type=int, default=20000)
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--browser-port", type=int, default=8480)
    p.add_argument("--html-port", type=int, default=8481)
    p.add_argument("--name", default="delta_baseline")
    p.add_argument("--wandb", choices=("disabled", "online", "offline"), default="disabled")
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    plan = gpu_plan(a.gpus, a.per_gpu_envs)
    config = load_trajectory_direct_config(a.config, project_root=ROOT)
    validate_baseline(config)
    proof = verify(root=ROOT, config_path=a.config, contract_path=a.contract,
        report_path=Path(os.environ["TRAJTRACKING_ALIGNMENT_REPORT"]),
        scene_asset_root=Path(os.environ["TRAJTRACKING_SCENE_ASSET_ROOT"]))
    print(json.dumps(proof, indent=2), flush=True)
    if a.mode == "preflight":
        return 0
    if a.mode == "evaluate" and a.checkpoint is None:
        p.error("evaluate requires --checkpoint")
    if a.mode != "train" and a.resume_checkpoint is not None:
        p.error("only train supports exact resume")
    bindings = exclusive_gpus(plan["gpu_indices"])
    env = dict(os.environ, TRAJTRACKING_WORKSPACE=str(ROOT),
        TRAJTRACKING_PHYSICAL_GPUS=a.gpus,
        CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in bindings),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMNI_KIT_ACCEPT_EULA="YES")
    tmp = ROOT / "tmp"; tmp.mkdir(exist_ok=True)
    env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    (tmp/"launch_plan.json").write_text(json.dumps(dict(plan, bindings=bindings, mode=a.mode), indent=2))
    if env.get("TRAJTRACKING_NCCL_LIBRARY"):
        env["LD_PRELOAD"] = env["TRAJTRACKING_NCCL_LIBRARY"]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={plan['world_size']}", "--max-restarts=0",
        "-m", "trajectory_tracking.cli.distributed_train",
        "--expected-world-size", str(plan["world_size"]),
        "--cuda-device-indices", ",".join(str(i) for i in range(plan["world_size"])),
        "--config", str(a.config.resolve()), "--device", "cuda:0",
        "--num-envs", str(plan["global_envs"]), "--max-updates", str(a.max_updates),
        "--checkpoint-interval", str(a.interval), "--evaluation-interval", str(a.interval),
        "--heldout-test-interval", str(a.interval), "--evaluation-data-split", "test",
        "--target-success-rate", "0.95", "--success-patience", "3", "--disable-success-early-stop",
        "--gate-no-offset-mean-error-m", "0.03", "--gate-no-offset-success-rate", "0.96",
        "--source-label-admission", "--scene-asset-root", env["TRAJTRACKING_SCENE_ASSET_ROOT"],
        "--output-root", str(ROOT/"logs/trajectory_tracking"), "--experiment-name", a.name,
        "--reference-storage", "cpu", "--headless", "--scene-build-mode", "legacy",
        "--browser-enabled", "--no-browser-object-flow-enabled",
        "--browser-port", str(a.browser_port)]
    if a.record:cmd += ["--record-evaluation-video", "--evaluation-html-port", str(a.html_port)]
    if a.wandb != "disabled":
        cmd += ["--wandb-enabled", "--wandb-mode", a.wandb, "--wandb-project", "TrajTrackingRL",
                "--wandb-group", "delta-baseline", "--wandb-run-name", a.name]
    if a.resume_checkpoint:cmd += ["--resume-checkpoint", str(a.resume_checkpoint.resolve())]
    if a.mode == "evaluate":
        checkpoint=a.checkpoint/"policy.pt" if a.checkpoint.is_dir() else a.checkpoint
        cmd += ["--evaluate-checkpoint", str(checkpoint.resolve())]
    os.execvpe(sys.executable, cmd, env)

if __name__ == "__main__":
    raise SystemExit(main())
