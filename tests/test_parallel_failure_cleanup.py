"""Real CPU subprocesses exercise production main/cleanup; no Isaac or GPU claim."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", [
    "success", "training_error", "tracker_error", "browser_error",
    "both_errors", "tracker_hangs", "normal_cleanup_error", "startup_error",
])
def test_worker_cleanup_subprocess(tmp_path, scenario):
    env = dict(os.environ, TMPDIR=str(ROOT / "tmp"), TMP=str(ROOT / "tmp"),
               TEMP=str(ROOT / "tmp"), WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
               RANK="0", LOCAL_RANK="0", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", scenario, str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    out, err = result.stdout, result.stderr
    (tmp_path / "stdout.log").write_text(out)
    (tmp_path / "stderr.log").write_text(err)
    assert "PRODUCTION_MAIN" in out, out + err
    started = next(float(line.split("=", 1)[1]) for line in out.splitlines()
                   if line.startswith("CLEANUP_STARTED_AT="))
    cleanup_seconds = time.monotonic() - started
    (tmp_path / "result.json").write_text(json.dumps(dict(
        scenario=scenario, returncode=result.returncode, cleanup_seconds=cleanup_seconds)))
    assert cleanup_seconds < 8, out + err
    if scenario == "success":
        assert result.returncode == 0, out + err
        markers = ["TRAIN_COMPLETE", "TRACKER_FINISH_0", "CONTEXT_CLOSE",
                   "OWNED_BROWSER_REAPED", "APP_CLOSE"]
        assert [out.index(x) for x in markers] == sorted(out.index(x) for x in markers)
        assert "Traceback" not in err
        return
    assert result.returncode == 1, out + err
    assert "Traceback (most recent call last)" in err
    assert "APP_CLOSE" not in out  # Fake Kit close exits0, just like native fast shutdown.
    if scenario == "startup_error":
        assert "RuntimeError: ORIGINAL_STARTUP_FAILURE" in err
        assert "CONTEXT_CLOSE" not in out
        return
    if scenario == "normal_cleanup_error":
        assert "RuntimeError: ORIGINAL_NORMAL_CLEANUP_FAILURE" in err
    else:
        assert "RuntimeError: ORIGINAL_TRAINING_FAILURE" in err
        assert "CONTEXT_CLOSE" not in out  # Calling it would block forever.
    assert "OWNED_BROWSER_REAPED" in out
    assert "TRACKER_FINISH_1" in out
    if scenario in ("browser_error", "both_errors"):
        assert err.index("ORIGINAL_TRAINING_FAILURE") < err.index("SECONDARY_BROWSER_FAILURE")
    if scenario in ("tracker_error", "both_errors"):
        assert err.index("ORIGINAL_TRAINING_FAILURE") < err.index("SECONDARY_TRACKER_FAILURE")


def _worker(scenario, output):
    # The shared interpreter may have a foreign editable project in sitecustomize.
    sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]
    from types import SimpleNamespace
    from threading import Event
    import torch
    from parallel_fixtures import synthetic_cli_config
    from trajectory_tracking.cli import distributed_train as cli

    assert Path(cli.__file__).resolve().is_relative_to(ROOT)
    config = synthetic_cli_config(output)
    sys.argv = ["distributed_train", "--synthetic-smoke", "--expected-world-size", "1",
                "--config", str(config), "--device", "cpu", "--num-envs", "4",
                "--max-updates", "1", "--checkpoint-interval", "1",
                "--evaluation-interval", "1", "--heldout-test-interval", "0",
                "--target-success-rate", "0.95", "--success-patience", "3",
                "--gate-no-offset-mean-error-m", "0.03", "--gate-no-offset-success-rate", "0.96",
                "--scene-asset-root", ".cache/scene_assets/nvidia/isaac-5.1",
                "--disable-success-early-stop", "--output-root", str(output),
                "--experiment-name", scenario, "--no-wandb-enabled", "--no-browser-enabled"]
    context = cli.ParallelContext(torch.device("cpu"))

    def close_context():
        print("CONTEXT_CLOSE", flush=True)
        if scenario == "normal_cleanup_error":
            raise RuntimeError("ORIGINAL_NORMAL_CLEANUP_FAILURE")
        if scenario != "success":
            Event().wait()  # Models NCCL destruction waiting for a peer forever.

    context.close = close_context
    cli.ParallelContext.from_torchrun = lambda **kwargs: context

    class Tracker:
        run_id = run_url = None

        def __init__(self, **kwargs):
            pass

        def bind_manifest(self, manifest):
            pass

        def finish(self, *, exit_code):
            print(f"TRACKER_FINISH_{exit_code}", flush=True)
            if scenario == "tracker_hangs":
                Event().wait()  # Exercise the failure-only five-second deadline.
            if scenario in ("tracker_error", "both_errors"):
                raise RuntimeError("SECONDARY_TRACKER_FAILURE")

    class OwnedBrowser:
        def __init__(self):
            # Own an actual local child; reap precisely it, never a process group.
            self.child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        def observe(self, env):
            pass

        def close(self):
            self.child.terminate()
            self.child.wait(timeout=2)
            print("OWNED_BROWSER_REAPED", flush=True)
            if scenario in ("browser_error", "both_errors"):
                raise RuntimeError("SECONDARY_BROWSER_FAILURE")

    def app_close(**kwargs):
        print("APP_CLOSE", flush=True)
        assert kwargs == {"wait_for_replicator": False, "skip_cleanup": True}
        os._exit(0)

    original_runtime = cli.create_runtime

    def runtime(args, config, context, data):
        if scenario == "startup_error":
            # Exercise the real create_runtime exception branch, without Isaac.
            from trajectory_tracking.environment import scene
            scene.launch_simulation_app = lambda **kwargs: SimpleNamespace(
                app=SimpleNamespace(close=app_close))

            def failed_scene(**kwargs):
                print(f"CLEANUP_STARTED_AT={time.monotonic()}", flush=True)
                raise RuntimeError("ORIGINAL_STARTUP_FAILURE")

            scene.create_simulation_context = failed_scene
            data.source = SimpleNamespace(spec=SimpleNamespace(physics=None))
            args.synthetic_smoke = False
            return original_runtime(args, config, context, data)
        result = original_runtime(args, config, context, data)
        result.browser = OwnedBrowser()
        result.app = SimpleNamespace(close=app_close)
        return result

    def train(self, **kwargs):
        print(f"CLEANUP_STARTED_AT={time.monotonic()}", flush=True)
        if scenario not in ("success", "normal_cleanup_error"):
            raise RuntimeError("ORIGINAL_TRAINING_FAILURE")
        print("TRAIN_COMPLETE", flush=True)
        return 1

    cli.create_runtime = runtime
    cli.TrajectoryWandbMirror = Tracker
    cli.SynchronousCoordinator.run = train
    cli._run_nominal_gate = lambda *args, **kwargs: None
    print("PRODUCTION_MAIN", flush=True)
    raise SystemExit(cli.main())


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    _worker(sys.argv[2], Path(sys.argv[3]))
