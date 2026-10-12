"""Bounded, evaluation-scoped recording of a live browser (no simulator imports).

The producer publishes its first evaluation frame before ``frame_ready`` and
keeps its last evaluation presentation until ``scope`` exits. Short evaluations
therefore hold their real final scene; training cannot leak into the clip.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

OWNER = "evaluation_browser_v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rotate_owned(directory, run_id, *, keep=3, reserve=0):
    """Delete only clips whose sidecar declares this recorder and exact run."""
    owned = []
    for sidecar in Path(directory).glob("eval_*.json"):
        if sidecar.is_symlink():
            continue
        try:
            row = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue  # Unrelated or damaged metadata is never deletion authority.
        if not isinstance(row, dict):
            continue
        clip = sidecar.with_suffix(".mp4")
        if row.get("owner") == OWNER and row.get("run_id") == run_id and clip.is_file() and not clip.is_symlink():
            owned.append((sidecar, clip))
    removed = []
    for sidecar, clip in sorted(owned)[:max(0, len(owned) - keep + reserve)]:
        clip.unlink()
        sidecar.unlink()
        removed.append(clip.name)
    return removed


class EvaluationBrowserRecorder:
    def __init__(self, *, url, run_dir, commit, tmp_root, duration=60,
                 ready_timeout=120, finish_timeout=180):
        if duration <= 0 or ready_timeout <= 0 or finish_timeout <= 0:
            raise ValueError("recording durations/timeouts must be positive")
        self.url, self.run_dir, self.commit = url, Path(run_dir).resolve(), commit
        self.directory = self.run_dir / "evaluation_videos"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError("evaluation video directory cannot be a symlink")
        self.tmp_root = Path(tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.duration, self.ready_timeout, self.finish_timeout = duration, ready_timeout, finish_timeout
        self.active = None

    def event(self, event, **values):
        # Diagnostics must not turn an optional recording failure into a rank failure.
        try:
            row = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **values)
            with (self.directory / "recorder.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"[evaluation video] {event}: {values}", flush=True)
        except Exception as error:
            try:
                print(f"[evaluation video] {event}: {values}; logging failed: {error!r}",
                      file=sys.stderr, flush=True)
            except Exception:
                pass  # Even a closed stderr must not mask a scoring exception.

    @contextmanager
    def scope(self, **metadata):
        if self.active is not None:
            raise RuntimeError("nested evaluation recording is not supported")
        scratch = None
        lock_fd = None
        state = None
        try:
            try:
                # One nonblocking run lock covers reservation, capture and publication.
                # Never unlink it: replacing the inode would permit a second owner.
                lock_fd = os.open(self.directory / ".recorder.lock",
                                  os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                scratch = tempfile.TemporaryDirectory(prefix="evaluation-video-", dir=self.tmp_root)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                name = f"eval_{stamp}_u{metadata['update']:06d}_{metadata['split']}_{metadata.get('policy', 'learned')}"
                state = dict(scratch=Path(scratch.name), name=name, metadata=metadata,
                             process=None, log=None, attempted=False, lock_fd=lock_fd)
                self.active = state
                self.event("evaluation_started", **metadata)
            except Exception as error:
                self.event("recording_skipped", stage="setup_or_lock", error=repr(error), **metadata)
            try:
                yield  # Evaluation errors belong to the caller, never to the recorder.
            except BaseException:
                if state is not None:
                    state["metadata"]["evaluation_failed"] = True
                raise
            finally:
                if state is not None:
                    try:
                        self._finish(state, time.monotonic())
                    except Exception as error:
                        self.event("recording_error", stage="finish", error=repr(error), **metadata)
        finally:
            try:
                if state is not None:
                    self._cleanup(state)
            finally:
                self.active = None
                # Close, do not LOCK_UN: the worker inherits this open description.
                # If the parent dies, its live worker still excludes another capture.
                for resource, cleanup in ((scratch, lambda item: item.cleanup()),
                                          (lock_fd, os.close)):
                    if resource is not None:
                        try:
                            cleanup(resource)
                        except Exception as error:
                            self.event("recording_error", stage="release", error=repr(error))

    def frame_ready(self):
        """Called after publishing actual evaluation data; reads never step physics."""
        state = self.active
        if state is None or state["attempted"]:
            return
        state["attempted"] = True
        try:
            removed = rotate_owned(self.directory, self.run_dir.name, reserve=1)
            self.event("retention", removed=removed)
            request = dict(url=self.url, duration=self.duration,
                           parent_pid=os.getpid(), ready_timeout=self.ready_timeout)
            write_json(state["scratch"] / "request.json", request)
            state["log"] = (self.directory / "worker.log").open("a")
            state["process"] = subprocess.Popen(
                [sys.executable, "-m", __name__, str(state["scratch"])],
                stdout=state["log"], stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(state["lock_fd"],),
                cwd=Path.cwd())
            deadline = time.monotonic() + self.ready_timeout
            while not (state["scratch"] / "ready.json").exists():
                if state["process"].poll() is not None:
                    raise RuntimeError(f"browser worker exited {state['process'].returncode}; see worker.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("evaluation browser readiness timed out")
                time.sleep(.05)
            self.event("recording_started", **state["metadata"])
        except Exception as error:
            self.event("recording_error", stage="start", error=repr(error), **state["metadata"])
            self._cleanup(state)

    def _finish(self, state, ended):
        process = state["process"]
        if process is None:
            self.event("clip_missing", reason="no ready evaluation frame/browser", **state["metadata"])
            return
        # Wait before the caller restores training state. Other ranks reach their
        # normal evaluation barrier; no video exception escapes into collectives.
        code = process.wait(timeout=self.duration + self.finish_timeout)
        if code:
            raise RuntimeError(f"browser worker exited {code}; see worker.log")
        result = json.loads((state["scratch"] / "result.json").read_text())
        hold = max(0., result["recording_ended_monotonic"] - ended)
        metadata = dict(owner=OWNER, run_id=self.run_dir.name, commit=self.commit,
                        **state["metadata"], **result, hold_seconds=hold,
                        hold_policy="last_real_evaluation_scene_before_training_restore",
                        training_waited_for_recording=True)
        destination = self.directory / (state["name"] + ".mp4")
        source = state["scratch"] / "clip.mp4"
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("recording produced no encoded video")
        write_json(destination.with_suffix(".json"), metadata)
        source.replace(destination)
        self.event("clip_completed", path=destination.name, hold_seconds=hold, **state["metadata"])

    def _cleanup(self, state):
        # Each cleanup action is independent: a failed TERM/wait/log flush must
        # not skip KILL/reap or replace the evaluation's original exception.
        process = state["process"]
        if process is not None:
            reaped = False
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    self.event("recording_error", stage="signal", error=repr(error))
                if not reaped:
                    try:
                        process.wait(timeout=5)
                        reaped = True
                    except Exception as error:
                        self.event("recording_error", stage="reap", error=repr(error))
            state["process"] = None
        if state["log"] is not None:
            try:
                state["log"].close()
            except Exception as error:
                self.event("recording_error", stage="log_close", error=repr(error))
            finally:
                state["log"] = None


def record_worker(scratch):
    from playwright.sync_api import sync_playwright
    from .browser_capture import _record_clip, _pid_exists
    scratch = Path(scratch)
    request = json.loads((scratch / "request.json").read_text())
    deadline = time.monotonic() + request["ready_timeout"] + request["duration"] + 120
    started = None

    def ready():
        nonlocal started
        started = time.monotonic()
        write_json(scratch / "ready.json", dict(started_monotonic=started))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--enable-webgl",
            "--use-angle=swiftshader", "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding"])
        try:
            clip = _record_clip(browser, request["url"], scratch, request["duration"],
                should_continue=lambda: _pid_exists(request["parent_pid"]) and time.monotonic() < deadline,
                on_recording_started=ready)
            if clip is None:
                raise RuntimeError("recording cancelled before completion")
            clip.path.replace(scratch / "clip.mp4")
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height",
                "-of", "json", str(scratch / "clip.mp4")], check=True, capture_output=True, text=True, timeout=30)
            info = json.loads(probe.stdout)
            if abs(float(info["format"]["duration"]) - request["duration"]) > .25:
                raise RuntimeError(f"encoded clip duration {info['format']['duration']} differs from requested {request['duration']}")
            write_json(scratch / "result.json", dict(ffprobe=info,
                duration_seconds=request["duration"], renderer="live_Chromium_Threejs_MediaRecorder",
                recording_started_monotonic=started, recording_ended_monotonic=started + clip.wall_seconds,
                active_seconds=clip.active_seconds, paused_seconds=clip.paused_seconds))
        finally:
            browser.close()


if __name__ == "__main__":
    record_worker(sys.argv[1])
