"""Lifecycle/retention/failure contracts; encoding is exercised with real ffmpeg."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from trajectory_tracking.visualization.evaluation_browser_recorder import EvaluationBrowserRecorder, OWNER, rotate_owned
from trajectory_tracking.training.coordinator import SynchronousCoordinator


@pytest.mark.parametrize("fails", [False, True])
def test_evaluation_scope_finishes_before_training_state_restore(fails):
    events = []
    @contextmanager
    def scope(**metadata):
        assert metadata == dict(update=7, split="test", policy="learned")
        events.append("scope_start")
        try:
            yield
        finally:
            events.append("recording_finished")
    def evaluate():
        events.append("actual_evaluation")
        if fails:
            raise ValueError("evaluation failed")
        return dict(event="test_evaluation")
    env = SimpleNamespace(rollout_state_dict=lambda: {}, load_rollout_state_dict=lambda state: events.append("restore"))
    runner = SimpleNamespace(env=env, trainer=SimpleNamespace(model=SimpleNamespace(training=True, train=lambda mode: None)),
                             observation_callback=lambda env: events.append("training_frame"))
    coordinator = SynchronousCoordinator(runner=runner,
        context=SimpleNamespace(barrier=lambda: None, reduce=lambda t, op: t),
        artifacts=None, tracker=None, contract={}, checkpoint_kwargs={},
        evaluate_train=None, evaluate_test=evaluate, evaluation_scope=scope)
    coordinator.emit = lambda payload: None
    if fails:
        with pytest.raises(ValueError, match="evaluation failed"):
            coordinator.evaluate(7, heldout=True)
    else:
        coordinator.evaluate(7, heldout=True)
    assert events == ["scope_start", "actual_evaluation", "recording_finished", "restore", "training_frame"]


def test_four_real_encoded_clips_retain_latest_three_and_foreign_files(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    original = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10",
        "-t", "0.2", "-c:v", "libx264", "-threads", "1", str(original)], check=True, timeout=30)
    foreign = tmp_path / "eval_foreign.mp4"
    shutil.copyfile(original, foreign)
    (tmp_path / "eval_foreign.json").write_text(json.dumps(dict(owner=OWNER, run_id="someone_else")))
    (tmp_path / "eval_unrelated.json").write_text("malformed foreign metadata")
    for index in range(4):
        rotate_owned(tmp_path, "my_run", reserve=1)
        shutil.copyfile(original, tmp_path / f"eval_{index}.mp4")
        (tmp_path / f"eval_{index}.json").write_text(json.dumps(dict(owner=OWNER, run_id="my_run")))
        assert len(list(tmp_path.glob("eval_[0-9].mp4"))) <= 3
    assert sorted(p.stem for p in tmp_path.glob("eval_[0-9].mp4")) == ["eval_1", "eval_2", "eval_3"]
    assert foreign.read_bytes() == original.read_bytes()


def test_start_failure_is_logged_reaped_and_does_not_interrupt_evaluation(tmp_path, monkeypatch):
    import trajectory_tracking.visualization.evaluation_browser_recorder as module
    recorder = EvaluationBrowserRecorder(url="http://unused", run_dir=tmp_path / "run", commit="abc", tmp_root=tmp_path)
    def fail(*args, **kwargs):
        raise OSError("browser unavailable")
    monkeypatch.setattr(module.subprocess, "Popen", fail)
    with recorder.scope(update=0, split="test", policy="learned"):
        recorder.frame_ready()
        recorder.frame_ready()  # No unbounded retry for the same evaluation.
    rows = [json.loads(x) for x in (recorder.directory / "recorder.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "recording_error" for row in rows) == 1
    assert recorder.active is None
    assert not list(tmp_path.glob("evaluation-video-*"))


def test_short_evaluation_waits_for_capture_and_records_hold(tmp_path, monkeypatch):
    import trajectory_tracking.visualization.evaluation_browser_recorder as module
    recorder = EvaluationBrowserRecorder(url="http://unused", run_dir=tmp_path / "run", commit="abc", tmp_root=tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Unit test only: lifecycle bookkeeping, not evidence of a rendered video.
    (scratch / "clip.mp4").write_bytes(b"bookkeeping-test-only")
    (scratch / "result.json").write_text(json.dumps(dict(recording_ended_monotonic=80.0)))
    waits = []
    state = dict(process=SimpleNamespace(wait=lambda timeout: waits.append(timeout) or 0), scratch=scratch,
                 name="eval_unit", metadata=dict(update=2, split="test"))
    recorder._finish(state, ended=30.)
    assert waits == [240]
    metadata = json.loads((recorder.directory / "eval_unit.json").read_text())
    assert metadata["hold_seconds"] == 50
    assert metadata["training_waited_for_recording"] is True


def test_cli_video_modes_are_exclusive_and_old_default_is_preserved(monkeypatch):
    import sys
    from trajectory_tracking.cli import distributed_train as cli
    base = dict(num_envs=8192, max_updates=2, checkpoint_interval=1, evaluation_interval=2,
                heldout_test_interval=0, disable_success_early_stop=True, browser_enabled=True,
                browser_env_index=0, wandb_enabled=False)
    monkeypatch.setattr(cli, "_parse_args", lambda remaining: SimpleNamespace(**base))
    monkeypatch.setattr(sys, "argv", ["distributed_train"])
    assert not cli._arguments().record_evaluation_video
    monkeypatch.setattr(sys, "argv", ["distributed_train", "--record-evaluation-video"])
    assert cli._arguments().record_evaluation_video
    monkeypatch.setattr(sys, "argv", ["distributed_train", "--record-video", "--record-evaluation-video"])
    with pytest.raises(ValueError, match="not both"):
        cli._arguments()


def test_worker_timeout_cleans_owned_process_group_and_logs(tmp_path, monkeypatch):
    import trajectory_tracking.visualization.evaluation_browser_recorder as module
    recorder = EvaluationBrowserRecorder(url="http://unused", run_dir=tmp_path / "run", commit="abc", tmp_root=tmp_path)
    waits, signals = [], []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("worker", timeout)
        return -15
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with recorder.scope(update=3, split="test"):
        recorder.active["process"] = SimpleNamespace(pid=123456, wait=wait)
    assert waits == [240, 5]
    assert [pid for pid, sig in signals] == [123456, 123456]
    assert recorder.active is None
    assert '"stage": "finish"' in (recorder.directory / "recorder.jsonl").read_text()


def test_transcoder_has_bounded_wait_and_cleans_partial(tmp_path, monkeypatch):
    from trajectory_tracking.visualization import browser_capture as capture
    monkeypatch.setattr(capture.shutil, 'which', lambda name: '/fake/ffmpeg')
    observed = []
    def timeout(command, **kwargs):
        observed.append(kwargs['timeout'])
        partial = tmp_path / '.out.partial.mp4'
        partial.write_bytes(b'incomplete')
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(capture.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        capture._transcode_to_mp4(tmp_path / 'input.webm', tmp_path / 'out.mp4', duration_s=60)
    assert observed == [180]
    assert not (tmp_path / '.out.partial.mp4').exists()
