"""End-to-end torchrun resume, including distinct worker RNG and environment state."""

import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
import pytest
from parallel_fixtures import synthetic_cli_config

ROOT = Path(__file__).resolve().parents[1]


def test_periodic_training_recording_is_rejected(monkeypatch):
    from trajectory_tracking.cli import distributed_train as cli
    base = dict(num_envs=8192, max_updates=2, checkpoint_interval=1,
                evaluation_interval=2, heldout_test_interval=0, disable_success_early_stop=True,
                browser_enabled=True, browser_env_index=0, wandb_enabled=False)
    monkeypatch.setattr(cli, "_parse_args", lambda remaining: SimpleNamespace(**base))
    monkeypatch.setattr(sys, "argv", ["distributed_train", "--record-video"])
    with pytest.raises(ValueError, match="periodic training video is disabled"):
        cli._arguments()


def _run(output, name, updates, resume=None, *, world_size=2):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc-per-node={world_size}", "--max-restarts=0", "-m", "trajectory_tracking.cli.distributed_train",
               "--synthetic-smoke", "--expected-world-size", str(world_size), "--config",
               str(synthetic_cli_config(output)),
               "--device", "cpu", "--num-envs", str(4 * world_size), "--max-updates", str(updates),
               "--checkpoint-interval", "1", "--evaluation-interval", "1", "--heldout-test-interval", "0",
               "--target-success-rate", "0.95", "--success-patience", "3", "--disable-success-early-stop",
               "--gate-no-offset-mean-error-m", "0.03", "--gate-no-offset-success-rate", "0.96",
               "--scene-asset-root", ".cache/scene_assets/nvidia/isaac-5.1", "--output-root", str(output),
               "--experiment-name", name, "--no-wandb-enabled", "--no-browser-enabled"]
    if resume is not None:
        command += ["--resume-checkpoint", str(resume)]
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Cold imports on a shared SDK filesystem can take minutes. This is a
    # startup budget; worker failure-cleanup deadlines remain independently tested.
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            raise
        result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    matches = list(output.glob(f"*_{name}"))
    assert len(matches) == 1  # Not one W&B/artifact directory per rank.
    return matches[0]


@pytest.mark.parametrize("world_size", [2, 7])
def test_torchrun_resume_matches_uninterrupted_policy(tmp_path, world_size):
    first = _run(tmp_path, "first", 2, world_size=world_size)
    resumed = _run(tmp_path, "resumed", 3, first / "checkpoints/update_000002", world_size=world_size)
    full = _run(tmp_path, "full", 3, world_size=world_size)
    def policy(run):
        return torch.load(run / "checkpoints/update_000003/policy.pt", weights_only=False, map_location="cpu")
    actual, expected = policy(resumed), policy(full)
    for key, value in expected["trainer"]["model"].items():
        torch.testing.assert_close(value, actual["trainer"]["model"][key], atol=0, rtol=0)
    metadata = json.loads((resumed / "checkpoints/update_000003/complete.json").read_text())
    assert {row["file"] for row in metadata["files"]} == {"policy.pt"} | {
        f"rank_{rank:03d}.pt" for rank in range(world_size)}
    for rank in range(world_size):
        filename = f"checkpoints/update_000003/rank_{rank:03d}.pt"
        resumed_state = torch.load(resumed / filename, weights_only=False, map_location="cpu")
        full_state = torch.load(full / filename, weights_only=False, map_location="cpu")
        assert torch.equal(resumed_state["sample_generator"], full_state["sample_generator"])
    logs = [json.loads(row) for row in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in logs if row["event"] == "update"] == [3]


@pytest.mark.parametrize("exit_code,close_raises", [(0, False), (1, True)])
def test_simulator_worker_exit_preserves_status_and_skips_stage_callbacks(monkeypatch, exit_code, close_raises):
    from trajectory_tracking.cli import distributed_train as cli
    events = []
    def close(**kwargs):
        events.append(kwargs)
        if close_raises:
            raise RuntimeError("backend close failed")
    def process_exit(code):
        events.append(code)
        raise SystemExit(code)
    monkeypatch.setattr(cli.os, "_exit", process_exit)
    with pytest.raises(SystemExit) as result:
        cli._close_simulator_process(SimpleNamespace(close=close), exit_code)
    assert result.value.code == exit_code
    expected_close = [] if exit_code else [{"wait_for_replicator": False, "skip_cleanup": True}]
    assert events == expected_close + [exit_code]


@pytest.mark.parametrize("exit_code", [0, 1, 17, 130])
def test_simulator_process_exit_subprocess_preserves_requested_status(exit_code):
    # A C++ shutdown can exit the process itself, bypassing Python finally.
    # Use a real child process: monkeypatching os._exit into an exception misses it.
    code = """
import os
import signal
from types import SimpleNamespace
from trajectory_tracking.cli.distributed_train import _close_simulator_process

def close(**kwargs):
    assert kwargs == {"wait_for_replicator": False, "skip_cleanup": True}
    print("APP_CLOSE_INTERNAL_EXIT_ZERO", flush=True)
    os._exit(0)

app = SimpleNamespace(close=close)
requested = REQUESTED_STATUS
if requested:
    try:
        raise RuntimeError("original worker failure must remain visible")
    except RuntimeError:
        _close_simulator_process(app, requested)
else:
    _close_simulator_process(app, requested)
""".replace("REQUESTED_STATUS", str(exit_code))
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == exit_code, (result.stdout, result.stderr)
    assert f"distributed worker terminating with code {exit_code}" in result.stdout
    if exit_code:
        assert "APP_CLOSE_INTERNAL_EXIT_ZERO" not in result.stdout
        assert "Traceback (most recent call last)" in result.stderr
        assert "RuntimeError: original worker failure must remain visible" in result.stderr
    else:
        assert "APP_CLOSE_INTERNAL_EXIT_ZERO" in result.stdout
        assert not result.stderr
