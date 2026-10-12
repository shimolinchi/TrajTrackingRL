"""Simulator-independent interactive camera state and command buffering."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, exp, isfinite, pi, sin, sqrt, tan
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class CameraCommand:
    """One browser input batch.

    Orbit and pan deltas are fractions of the displayed image height. Zoom is
    a wheel delta converted to CSS-pixel units.
    """

    orbit_x: float = 0.0
    orbit_y: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 0.0
    reset: bool = False
    client_sequence: int = 0


class CameraControlBuffer:
    """Accumulate browser input without exposing Isaac APIs to network threads."""

    _FIELDS = frozenset(
        {
            "orbit_x",
            "orbit_y",
            "pan_x",
            "pan_y",
            "zoom",
            "reset",
            "client_sequence",
        }
    )
    _MAX_ORBIT_PAN_DELTA = 4.0
    _MAX_ZOOM_DELTA = 2_000.0
    _MAX_CLIENT_SEQUENCE = (1 << 53) - 1

    def __init__(self) -> None:
        self._lock = Lock()
        self._orbit_x = 0.0
        self._orbit_y = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._zoom = 0.0
        self._reset = False
        self._client_sequence = 0

    @staticmethod
    def _number(payload: dict[str, Any], name: str) -> float:
        value = payload.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value

    @staticmethod
    def _bounded_sum(current: float, delta: float, limit: float) -> float:
        return max(-limit, min(limit, current + delta))

    def submit(self, payload: Any) -> None:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("camera command must be a non-empty JSON object")
        unknown = set(payload).difference(self._FIELDS)
        if unknown:
            raise ValueError(
                "unknown camera command fields: " + ", ".join(sorted(unknown))
            )
        reset = payload.get("reset", False)
        if not isinstance(reset, bool):
            raise ValueError("reset must be a boolean")
        client_sequence = payload.get("client_sequence", 0)
        if (
            isinstance(client_sequence, bool)
            or not isinstance(client_sequence, int)
            or not 0 <= client_sequence <= self._MAX_CLIENT_SEQUENCE
        ):
            raise ValueError(
                "client_sequence must be a non-negative JavaScript-safe integer"
            )

        orbit_x = self._number(payload, "orbit_x")
        orbit_y = self._number(payload, "orbit_y")
        pan_x = self._number(payload, "pan_x")
        pan_y = self._number(payload, "pan_y")
        zoom = self._number(payload, "zoom")

        with self._lock:
            if reset:
                self._orbit_x = 0.0
                self._orbit_y = 0.0
                self._pan_x = 0.0
                self._pan_y = 0.0
                self._zoom = 0.0
                self._reset = True
            orbit_pan_limit = self._MAX_ORBIT_PAN_DELTA
            self._orbit_x = self._bounded_sum(self._orbit_x, orbit_x, orbit_pan_limit)
            self._orbit_y = self._bounded_sum(self._orbit_y, orbit_y, orbit_pan_limit)
            self._pan_x = self._bounded_sum(self._pan_x, pan_x, orbit_pan_limit)
            self._pan_y = self._bounded_sum(self._pan_y, pan_y, orbit_pan_limit)
            self._zoom = self._bounded_sum(self._zoom, zoom, self._MAX_ZOOM_DELTA)
            self._client_sequence = max(
                self._client_sequence,
                client_sequence,
            )

    def consume(self) -> CameraCommand:
        with self._lock:
            command = CameraCommand(
                orbit_x=self._orbit_x,
                orbit_y=self._orbit_y,
                pan_x=self._pan_x,
                pan_y=self._pan_y,
                zoom=self._zoom,
                reset=self._reset,
                client_sequence=self._client_sequence,
            )
            self._orbit_x = 0.0
            self._orbit_y = 0.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._zoom = 0.0
            self._reset = False
            self._client_sequence = 0
            return command


class OrbitCameraState:
    """Pure-Python Z-up orbit camera used by the simulation thread."""

    _ORBIT_RADIANS_PER_VIEWPORT = 2.0 * pi
    _ZOOM_EXPONENT_PER_UNIT = 0.001
    _MAX_PITCH = 85.0 * pi / 180.0

    def __init__(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        *,
        vertical_fov: float = 36.0 * pi / 180.0,
    ) -> None:
        if len(eye) != 3 or len(target) != 3:
            raise ValueError("camera eye and target must be xyz triples")
        eye = tuple(float(value) for value in eye)
        self._default_target = tuple(float(value) for value in target)
        if not all(isfinite(value) for value in eye + self._default_target):
            raise ValueError("camera eye and target must contain finite values")
        offset = tuple(eye[index] - self._default_target[index] for index in range(3))
        distance = sqrt(sum(value * value for value in offset))
        if distance <= 0.0:
            raise ValueError("camera eye and target must differ")
        vertical_fov = float(vertical_fov)
        if not isfinite(vertical_fov) or not 0.0 < vertical_fov < pi:
            raise ValueError("vertical_fov must be finite and in (0, pi)")
        self._default_distance = distance
        self._vertical_fov = vertical_fov
        self._default_yaw = atan2(offset[1], offset[0])
        pitch = asin(max(-1.0, min(1.0, offset[2] / distance)))
        self._default_pitch = max(-self._MAX_PITCH, min(self._MAX_PITCH, pitch))
        self.reset()

    def reset(self) -> None:
        self._target = list(self._default_target)
        self._distance = self._default_distance
        self._yaw = self._default_yaw
        self._pitch = self._default_pitch

    def apply(self, command: CameraCommand) -> bool:
        changed = False
        if command.reset:
            self.reset()
            changed = True

        if command.orbit_x or command.orbit_y:
            self._yaw -= command.orbit_x * self._ORBIT_RADIANS_PER_VIEWPORT
            self._yaw = atan2(sin(self._yaw), cos(self._yaw))
            self._pitch += command.orbit_y * self._ORBIT_RADIANS_PER_VIEWPORT
            self._pitch = max(-self._MAX_PITCH, min(self._MAX_PITCH, self._pitch))
            changed = True

        if command.zoom:
            exponent = command.zoom * self._ZOOM_EXPONENT_PER_UNIT
            self._distance *= exp(max(-2.0, min(2.0, exponent)))
            self._distance = max(
                max(0.1, 0.1 * self._default_distance),
                min(10.0 * self._default_distance, self._distance),
            )
            changed = True

        if command.pan_x or command.pan_y:
            cos_pitch = cos(self._pitch)
            offset = (
                self._distance * cos_pitch * cos(self._yaw),
                self._distance * cos_pitch * sin(self._yaw),
                self._distance * sin(self._pitch),
            )
            forward = tuple(-value / self._distance for value in offset)
            right_length = sqrt(forward[0] ** 2 + forward[1] ** 2)
            right = (
                forward[1] / right_length,
                -forward[0] / right_length,
                0.0,
            )
            view_up = (
                right[1] * forward[2],
                -right[0] * forward[2],
                right[0] * forward[1] - right[1] * forward[0],
            )
            scale = 2.0 * self._distance * tan(0.5 * self._vertical_fov)
            for index in range(3):
                self._target[index] += scale * (
                    -command.pan_x * right[index] + command.pan_y * view_up[index]
                )
            changed = True

        return changed

    def pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        cos_pitch = cos(self._pitch)
        offset = (
            self._distance * cos_pitch * cos(self._yaw),
            self._distance * cos_pitch * sin(self._yaw),
            self._distance * sin(self._pitch),
        )
        target = tuple(self._target)
        eye = tuple(target[index] + offset[index] for index in range(3))
        return eye, target


__all__ = ["CameraCommand", "CameraControlBuffer", "OrbitCameraState"]
