"""Bounded real-browser clip capture shared by evaluation recording."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

_RENDERABLE_THREE_SCENE_EXPRESSION = """
() => {
  const scene = window.visualizationThreeSceneConfig;
  if (scene === null || scene === undefined) return true;
  const snapshot = window.visualizationPresentationSnapshot;
  return window.visualizationThreeAssetsReady === true &&
    snapshot !== null && typeof snapshot === 'object' &&
    Number.isSafeInteger(snapshot.actor_sequence) &&
    snapshot.actor_sequence > 0;
}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_recording_ready_file(path: Path) -> None:
    """Atomically signal that MediaRecorder is active and the rollout may start."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(_utc_now() + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _transcode_to_mp4(
    source: Path, destination: Path, *, duration_s: int | None
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for browser recording")
    partial = destination.with_name(f".{destination.stem}.partial.mp4")
    try:
        subprocess.run(
            (
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-an",
                *(
                    ("-vf", f"tpad=stop_mode=clone:stop_duration={duration_s}",
                     "-t", str(duration_s))
                    if duration_s is not None else ()
                ),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(partial),
            ),
            check=True,
            timeout=180,
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _RecordedClip:
    path: Path
    active_seconds: float
    paused_seconds: float
    wall_seconds: float


def _active_recording_increment(state: str, elapsed_s: float) -> float:
    """Return the part of an interval that consumes the clip time budget."""

    if not isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("recording clock elapsed time must be finite and non-negative")
    if state == "recording":
        return elapsed_s
    if state == "paused":
        return 0.0
    raise RuntimeError(f"browser recorder entered unexpected state: {state!r}")


def _record_clip(
    browser: Any,
    url: str,
    videos_dir: Path,
    duration_s: int,
    *,
    should_continue: Callable[[], bool] | None = None,
    on_recording_started: Callable[[], None] | None = None,
    completion_file: Path | None = None,
) -> _RecordedClip | None:
    continue_recording = should_continue or (lambda: True)
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    raw_path = videos_dir / ".browser_recording.partial.webm"
    raw_path.unlink(missing_ok=True)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        # A canvas can technically be recordable before its asynchronous URDF
        # assets and first actor snapshot are visible. Waiting for both prevents
        # the beginning of a synchronized rollout from becoming an empty scene.
        page.wait_for_function(
            _RENDERABLE_THREE_SCENE_EXPRESSION,
            timeout=120_000,
        )
        page.evaluate(
            "() => new Promise(resolve => requestAnimationFrame(() => "
            "requestAnimationFrame(resolve)))"
        )
        recorder = page.locator("#recorder")
        start_button = page.locator("#record-start")
        start_wait_active_seconds = 0.0
        previous_start_poll_at = time.monotonic()
        while continue_recording():
            now = time.monotonic()
            start_wait_elapsed = max(0.0, now - previous_start_poll_at)
            if recorder.get_attribute("data-evaluation-paused") != "true":
                start_wait_active_seconds += start_wait_elapsed
            if start_wait_active_seconds > 120.0:
                raise TimeoutError(
                    "browser recording did not become startable within 120 "
                    "non-evaluation seconds"
                )
            previous_start_poll_at = now
            state = recorder.get_attribute("data-state")
            if state in ("recording", "paused"):
                break
            if state != "idle":
                raise RuntimeError(
                    f"browser recorder cannot start from state {state!r}"
                )
            if start_button.is_enabled():
                start_button.click()
            page.wait_for_timeout(100)
        else:
            return None

        if on_recording_started is not None:
            on_recording_started()

        target_active_seconds = float(duration_s)
        active_seconds = 0.0
        paused_seconds = 0.0
        recording_started_at = time.monotonic()
        previous_poll_at = recording_started_at
        while active_seconds < target_active_seconds:
            if completion_file is not None and completion_file.is_file():
                # The producer atomically writes its result after the endpoint
                # frame. Let the browser display that final scene before stop.
                page.wait_for_timeout(500)
                break
            if not continue_recording():
                page.click("#record-stop")
                return None
            remaining_seconds = target_active_seconds - active_seconds
            page.wait_for_timeout(max(1, min(200, round(remaining_seconds * 1000.0))))
            now = time.monotonic()
            elapsed_seconds = max(0.0, now - previous_poll_at)
            state = recorder.get_attribute("data-state")
            active_increment = _active_recording_increment(
                "" if state is None else state,
                elapsed_seconds,
            )
            active_seconds += active_increment
            paused_seconds += elapsed_seconds - active_increment
            previous_poll_at = now

        if completion_file is not None and not completion_file.is_file():
            raise TimeoutError("evaluation did not complete within the clip timeout")
        wall_seconds = time.monotonic() - recording_started_at
        with page.expect_download(timeout=60_000) as download_info:
            page.click("#record-stop")
        download = download_info.value
        failure = download.failure()
        if failure is not None:
            raise RuntimeError(f"browser video download failed: {failure}")
        download.save_as(raw_path)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = videos_dir / f"browser_{timestamp}.mp4"
        _transcode_to_mp4(
            raw_path, destination,
            duration_s=duration_s if completion_file is None else None,
        )
        return _RecordedClip(
            path=destination,
            active_seconds=active_seconds,
            paused_seconds=paused_seconds,
            wall_seconds=wall_seconds,
        )
    finally:
        raw_path.unlink(missing_ok=True)
        page.close()
