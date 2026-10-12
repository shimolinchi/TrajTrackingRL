"""Validated custom buttons and their thread-safe click mailbox."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import re
from threading import Lock


_BUTTON_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_BUTTON_STYLES = frozenset(("default", "primary", "danger"))
_MAX_BUTTONS = 16
_MAX_LABEL_LENGTH = 80
_MAX_PENDING_CLICKS = 256


@dataclass(frozen=True, slots=True)
class VisualizationButton:
    """One static browser button backed by a no-argument callback.

    The callback is never invoked by an HTTP worker.  The owning simulation
    script explicitly dispatches queued clicks from its main thread through
    :meth:`trajectory_tracking.visualization.Visualization.dispatch_button_callbacks`.
    """

    button_id: str
    label: str
    callback: Callable[[], object] = field(repr=False, compare=False)
    style: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.button_id, str):
            raise TypeError("button_id must be a string")
        if not _BUTTON_ID_PATTERN.fullmatch(self.button_id):
            raise ValueError(
                "button_id must match [A-Za-z][A-Za-z0-9_-]{0,63}: "
                f"{self.button_id!r}"
            )
        if not isinstance(self.label, str):
            raise TypeError("button label must be a string")
        if (
            not self.label
            or self.label != self.label.strip()
            or len(self.label) > _MAX_LABEL_LENGTH
            or not self.label.isprintable()
        ):
            raise ValueError(
                "button label must be non-empty, trimmed, contain no control "
                f"characters, and be at most {_MAX_LABEL_LENGTH} characters"
            )
        if not callable(self.callback):
            raise TypeError("button callback must be callable")
        if not isinstance(self.style, str):
            raise TypeError("button style must be a string")
        if self.style not in _BUTTON_STYLES:
            raise ValueError(
                "button style must be one of "
                f"{sorted(_BUTTON_STYLES)}: {self.style!r}"
            )


def normalize_buttons(
    buttons: Iterable[VisualizationButton],
) -> tuple[VisualizationButton, ...]:
    """Freeze and validate a custom button collection."""

    frozen = tuple(buttons)
    if len(frozen) > _MAX_BUTTONS:
        raise ValueError(f"at most {_MAX_BUTTONS} custom buttons are supported")
    for button in frozen:
        if not isinstance(button, VisualizationButton):
            raise TypeError(
                "buttons must contain only VisualizationButton instances"
            )
    button_ids = tuple(button.button_id for button in frozen)
    if len(button_ids) != len(set(button_ids)):
        raise ValueError("custom visualization button_id values must be unique")
    return frozen


class ButtonEventBuffer:
    """Bounded FIFO mailbox from HTTP workers to the simulation thread."""

    def __init__(
        self,
        button_ids: Iterable[str],
        *,
        max_pending: int = _MAX_PENDING_CLICKS,
    ) -> None:
        frozen_ids = tuple(button_ids)
        if any(not isinstance(button_id, str) for button_id in frozen_ids):
            raise TypeError("button_ids must contain only strings")
        allowed = frozenset(frozen_ids)
        if len(allowed) != len(frozen_ids):
            raise ValueError("button_ids must be unique")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int):
            raise TypeError("max_pending must be an integer")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._allowed = allowed
        self._max_pending = max_pending
        self._pending: deque[str] = deque()
        self._lock = Lock()
        self._closed = False

    def submit(self, button_id: object) -> None:
        """Queue one validated click without invoking user code."""

        if not isinstance(button_id, str) or button_id not in self._allowed:
            raise ValueError(f"unknown visualization button_id: {button_id!r}")
        with self._lock:
            if self._closed:
                raise RuntimeError("visualization button event buffer is closed")
            if len(self._pending) >= self._max_pending:
                raise OverflowError("too many pending visualization button clicks")
            self._pending.append(button_id)

    def consume_one(self) -> str | None:
        """Consume the oldest click, or return ``None`` when the queue is empty."""

        with self._lock:
            if not self._pending:
                return None
            return self._pending.popleft()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def button_ids(self) -> frozenset[str]:
        """Identifiers accepted by this mailbox."""

        return self._allowed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending.clear()


__all__ = ["ButtonEventBuffer", "VisualizationButton", "normalize_buttons"]
