"""Validated dashboard curves, logs, and bounded browser event transport."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
import re
from threading import Condition
from time import monotonic
from typing import Any


_METRIC_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}\Z")
_LOG_LEVELS = frozenset(("debug", "info", "warning", "error"))
_MAX_CURVES = 16
_MAX_LABEL_LENGTH = 80
_MAX_UNIT_LENGTH = 24
_MAX_LOG_MESSAGE_LENGTH = 2048
_MAX_LOG_SOURCE_LENGTH = 64
_MAX_RETAINED_EVENTS = 4096


def _validate_display_text(
    value: object,
    *,
    name: str,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (
        (not value and not allow_empty)
        or value != value.strip()
        or len(value) > max_length
        or (value and not value.isprintable())
    ):
        qualifier = "possibly empty, " if allow_empty else "non-empty, "
        raise ValueError(
            f"{name} must be {qualifier}trimmed, contain no control characters, "
            f"and be at most {max_length} characters"
        )
    return value


@dataclass(frozen=True, slots=True)
class VisualizationCurve:
    """One scalar metric rendered as a line in the browser curve panel."""

    metric_id: str
    label: str
    color: str = "#4FC3F7"
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str):
            raise TypeError("metric_id must be a string")
        if not _METRIC_ID_PATTERN.fullmatch(self.metric_id):
            raise ValueError(
                "metric_id must match [A-Za-z][A-Za-z0-9_-]{0,63}: "
                f"{self.metric_id!r}"
            )
        _validate_display_text(
            self.label,
            name="curve label",
            max_length=_MAX_LABEL_LENGTH,
        )
        if not isinstance(self.color, str):
            raise TypeError("curve color must be a string")
        if not _COLOR_PATTERN.fullmatch(self.color):
            raise ValueError(
                "curve color must be a six-digit CSS hex color such as "
                f"'#4FC3F7': {self.color!r}"
            )
        _validate_display_text(
            self.unit,
            name="curve unit",
            max_length=_MAX_UNIT_LENGTH,
            allow_empty=True,
        )


def normalize_curves(
    curves: Iterable[VisualizationCurve],
) -> tuple[VisualizationCurve, ...]:
    """Freeze and validate a curve collection in display order."""

    frozen = tuple(curves)
    if len(frozen) > _MAX_CURVES:
        raise ValueError(f"at most {_MAX_CURVES} dashboard curves are supported")
    for curve in frozen:
        if not isinstance(curve, VisualizationCurve):
            raise TypeError(
                "curves must contain only VisualizationCurve instances"
            )
    metric_ids = tuple(curve.metric_id for curve in frozen)
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("dashboard curve metric_id values must be unique")
    return frozen


@dataclass(frozen=True, slots=True)
class DashboardEventBatch:
    """Events retained after one browser sequence number."""

    sequence: int
    dropped: int
    events: tuple[dict[str, Any], ...]
    closed: bool


class DashboardEventStream:
    """Bounded, thread-safe telemetry stream from producers to browsers.

    Metric and log publishers never call browser or simulator APIs. Slow
    clients may lose the oldest telemetry events; the returned ``dropped``
    count makes that loss explicit. Button clicks use their separate reliable
    main-thread mailbox and are not part of this lossy stream.
    """

    def __init__(
        self,
        metric_ids: Iterable[str],
        *,
        log_enabled: bool,
        max_events: int = _MAX_RETAINED_EVENTS,
    ) -> None:
        frozen_ids = tuple(metric_ids)
        if any(not isinstance(metric_id, str) for metric_id in frozen_ids):
            raise TypeError("metric_ids must contain only strings")
        if any(not _METRIC_ID_PATTERN.fullmatch(metric_id) for metric_id in frozen_ids):
            raise ValueError("metric_ids contain an invalid identifier")
        if len(frozen_ids) != len(set(frozen_ids)):
            raise ValueError("metric_ids must be unique")
        if not isinstance(log_enabled, bool):
            raise TypeError("log_enabled must be a bool")
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an integer")
        if max_events <= 0:
            raise ValueError("max_events must be positive")

        self._metric_ids = frozenset(frozen_ids)
        self._log_enabled = log_enabled
        self._max_events = max_events
        self._condition = Condition()
        self._events: deque[dict[str, Any]] = deque()
        self._sequence = 0
        self._closed = False
        self._started_at = monotonic()
        self._last_metric_time: float | None = None

    @property
    def metric_ids(self) -> frozenset[str]:
        return self._metric_ids

    @property
    def log_enabled(self) -> bool:
        return self._log_enabled

    @property
    def sequence(self) -> int:
        with self._condition:
            return self._sequence

    def publish_metrics(
        self,
        values: Mapping[str, Real],
        *,
        time_s: float | None = None,
    ) -> int:
        """Publish one timestamped batch of registered scalar metrics."""

        if not isinstance(values, Mapping):
            raise TypeError("metric values must be a mapping")
        if not values:
            raise ValueError("metric values must not be empty")
        if any(not isinstance(metric_id, str) for metric_id in values):
            raise TypeError("metric value keys must be strings")
        unknown = set(values) - self._metric_ids
        if unknown:
            raise ValueError(f"unknown dashboard metric ids: {sorted(unknown)}")

        normalized: dict[str, float] = {}
        for metric_id, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"dashboard metric {metric_id!r} must be a real scalar"
                )
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(
                    f"dashboard metric {metric_id!r} must be finite"
                )
            normalized[metric_id] = numeric

        timestamp = self._normalize_time(time_s)
        with self._condition:
            if self._closed:
                raise RuntimeError("dashboard event stream is closed")
            if (
                self._last_metric_time is not None
                and timestamp < self._last_metric_time
            ):
                raise ValueError(
                    "dashboard metric time_s must be monotonically non-decreasing"
                )
            sequence = self._publish_locked(
                {
                    "type": "metrics",
                    "time_s": timestamp,
                    "values": normalized,
                }
            )
            self._last_metric_time = timestamp
            return sequence

    def publish_log(
        self,
        message: str,
        *,
        level: str = "info",
        source: str = "",
        time_s: float | None = None,
    ) -> int:
        """Publish one structured, single-line dashboard log record."""

        if not self._log_enabled:
            raise RuntimeError("dashboard output panel is not enabled")
        message = _validate_display_text(
            message,
            name="log message",
            max_length=_MAX_LOG_MESSAGE_LENGTH,
        )
        if not isinstance(level, str):
            raise TypeError("log level must be a string")
        if level not in _LOG_LEVELS:
            raise ValueError(f"log level must be one of {sorted(_LOG_LEVELS)}")
        source = _validate_display_text(
            source,
            name="log source",
            max_length=_MAX_LOG_SOURCE_LENGTH,
            allow_empty=True,
        )
        timestamp = self._normalize_time(time_s)
        with self._condition:
            return self._publish_locked(
                {
                    "type": "log",
                    "time_s": timestamp,
                    "level": level,
                    "source": source,
                    "message": message,
                }
            )

    def wait_after(
        self,
        sequence: int,
        *,
        timeout: float = 1.0,
    ) -> DashboardEventBatch:
        """Wait for and copy all retained events newer than ``sequence``."""

        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("dashboard sequence must be an integer")
        if sequence < 0:
            raise ValueError("dashboard sequence must be non-negative")
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise TypeError("dashboard timeout must be a real number")
        timeout_value = float(timeout)
        if not isfinite(timeout_value) or timeout_value < 0.0:
            raise ValueError("dashboard timeout must be finite and non-negative")

        with self._condition:
            if sequence > self._sequence:
                raise ValueError(
                    "dashboard sequence is newer than the current stream"
                )
            self._condition.wait_for(
                lambda: self._closed or self._sequence > sequence,
                timeout=timeout_value,
            )
            retained = tuple(
                self._copy_event(event)
                for event in self._events
                if event["sequence"] > sequence
            )
            oldest_sequence = (
                self._events[0]["sequence"]
                if self._events
                else self._sequence + 1
            )
            dropped = max(0, oldest_sequence - sequence - 1)
            return DashboardEventBatch(
                sequence=self._sequence,
                dropped=dropped,
                events=retained,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _normalize_time(self, value: float | None) -> float:
        timestamp = monotonic() - self._started_at if value is None else value
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
            raise TypeError("dashboard time_s must be a real number")
        normalized = float(timestamp)
        if not isfinite(normalized):
            raise ValueError("dashboard time_s must be finite")
        return normalized

    def _publish_locked(self, event: dict[str, Any]) -> int:
        if self._closed:
            raise RuntimeError("dashboard event stream is closed")
        self._sequence += 1
        stored = dict(event)
        stored["sequence"] = self._sequence
        self._events.append(stored)
        while len(self._events) > self._max_events:
            self._events.popleft()
        self._condition.notify_all()
        return self._sequence

    @staticmethod
    def _copy_event(event: dict[str, Any]) -> dict[str, Any]:
        copied = dict(event)
        if copied.get("type") == "metrics":
            copied["values"] = dict(copied["values"])
        return copied


__all__ = [
    "DashboardEventBatch",
    "DashboardEventStream",
    "VisualizationCurve",
    "normalize_curves",
]
