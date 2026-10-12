"""Small synchronous RGB-frame sink backed by the system FFmpeg binary."""

from __future__ import annotations

from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


class FfmpegVideoWriter:
    """Encode fixed-size RGB frames to an atomically published H.264 MP4."""

    def __init__(
        self,
        output_path: Path,
        *,
        resolution: tuple[int, int],
        fps: float,
        crf: int = 23,
        ffmpeg_executable: str | None = None,
    ) -> None:
        output = Path(output_path).expanduser().resolve()
        width, height = (int(value) for value in resolution)
        if output.suffix.lower() != ".mp4":
            raise ValueError("video output path must use the .mp4 suffix")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(
                "H.264 video resolution must contain positive even values"
            )
        if not isfinite(fps) or fps <= 0.0:
            raise ValueError("video fps must be finite and positive")
        if (
            isinstance(crf, bool)
            or not isinstance(crf, int)
            or not 0 <= crf <= 51
        ):
            raise ValueError("video CRF must be an integer in [0, 51]")
        executable = ffmpeg_executable or shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("FFmpeg is required for automatic MP4 recording")

        output.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output
        self._temporary_path = output.with_name(
            f".{output.stem}.{os.getpid()}.partial.mp4"
        )
        self._resolution = (width, height)
        self._fps = float(fps)
        self._crf = crf
        self._executable = str(executable)
        self._process: subprocess.Popen[bytes] | None = None
        self._frame_count = 0
        self._closed = False

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is not None:
            return self._process
        width, height = self._resolution
        command = [
            self._executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{self._fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(self._crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(self._temporary_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self._process

    def __call__(self, frame: Any) -> None:
        """Write one contiguous uint8 HWC RGB frame."""

        if self._closed:
            raise RuntimeError("video writer is closed")
        width, height = self._resolution
        if (
            getattr(frame, "shape", None) != (height, width, 3)
            or str(getattr(frame, "dtype", "")) != "uint8"
        ):
            raise ValueError(
                "video frame must be uint8 HWC RGB with shape "
                f"({height}, {width}, 3)"
            )
        process = self._start()
        assert process.stdin is not None
        try:
            process.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError) as error:
            detail = self._finish_failed_process(process)
            raise RuntimeError(
                f"FFmpeg stopped while recording video: {detail}"
            ) from error
        self._frame_count += 1

    def _finish_failed_process(self, process: subprocess.Popen[bytes]) -> str:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        stderr = b"" if process.stderr is None else process.stderr.read()
        self._temporary_path.unlink(missing_ok=True)
        return stderr.decode("utf-8", errors="replace").strip() or "unknown error"

    def close(self) -> None:
        """Finalize the MP4 and atomically expose it at ``output_path``."""

        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            raise RuntimeError("video recording produced no frames")
        assert process.stdin is not None
        process.stdin.close()
        try:
            return_code = process.wait(timeout=60.0)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=5.0)
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError("FFmpeg did not finish video encoding") from error
        stderr = b"" if process.stderr is None else process.stderr.read()
        if return_code != 0:
            self._temporary_path.unlink(missing_ok=True)
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"FFmpeg video encoding failed with code {return_code}: {detail}"
            )
        if (
            not self._temporary_path.is_file()
            or self._temporary_path.stat().st_size == 0
        ):
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError("FFmpeg completed without a non-empty MP4")
        os.replace(self._temporary_path, self._output_path)

    def abort(self) -> None:
        """Stop an incomplete recording and remove only its private partial file."""

        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        self._temporary_path.unlink(missing_ok=True)


__all__ = ["FfmpegVideoWriter"]
