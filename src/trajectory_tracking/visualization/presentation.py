"""Browser-only camera, actor, and trajectory presentation state.

This module deliberately contains no simulator imports.  A producer may use
the state for a browser-native geometry renderer without modifying the raw
camera JPEG, simulator scene, or policy observations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real
import re
from threading import Condition


_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _display_text(value: object, *, name: str, max_length: int = 96) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no outer whitespace")
    if len(value) > max_length or any(character in value for character in "\r\n\0"):
        raise ValueError(f"{name} is not valid display text")
    return value


def _hex_color(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _HEX_COLOR_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use #RRGGBB syntax")
    return value.upper()


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xyz(value: Sequence[Real], *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be an xyz triple")
    return tuple(
        _finite_number(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _finite_tuple(
    value: Sequence[Real],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(
        _finite_number(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )


@dataclass(frozen=True)
class BrowserSceneStyle:
    """Presentation-only palette for the browser canvas compositor.

    Colors, light washes, and the projected grid never enter USD, PhysX, an
    Isaac camera tensor, or the raw JPEG endpoints.
    """

    background_color: str = "#0B0F15"
    floor_tint: str = "#2D3B4E"
    grid_major_color: str = "#35557A"
    grid_minor_color: str = "#203447"
    x_axis_color: str = "#FFBE5C"
    y_axis_color: str = "#31D7C5"
    key_light_color: str = "#FFFFFF"
    fill_light_color: str = "#F6FAFF"
    rim_light_color: str = "#E8F3FF"
    grid_plane_z_m: float = -0.1
    grid_size_m: float = 6.0
    grid_divisions: int = 24
    color_grade_strength: float = 0.30
    light_strength: float = 0.22

    def __post_init__(self) -> None:
        for field_name in (
            "background_color",
            "floor_tint",
            "grid_major_color",
            "grid_minor_color",
            "x_axis_color",
            "y_axis_color",
            "key_light_color",
            "fill_light_color",
            "rim_light_color",
        ):
            object.__setattr__(
                self,
                field_name,
                _hex_color(getattr(self, field_name), name=field_name),
            )

        plane_z = _finite_number(self.grid_plane_z_m, name="grid_plane_z_m")
        grid_size = _finite_number(self.grid_size_m, name="grid_size_m")
        grade = _finite_number(
            self.color_grade_strength,
            name="color_grade_strength",
        )
        light = _finite_number(self.light_strength, name="light_strength")
        if grid_size <= 0.0:
            raise ValueError("grid_size_m must be positive")
        if isinstance(self.grid_divisions, bool) or not isinstance(
            self.grid_divisions, int
        ):
            raise TypeError("grid_divisions must be an integer")
        if not 2 <= self.grid_divisions <= 256:
            raise ValueError("grid_divisions must be in [2, 256]")
        if not 0.0 <= grade <= 1.0:
            raise ValueError("color_grade_strength must be in [0, 1]")
        if not 0.0 <= light <= 1.0:
            raise ValueError("light_strength must be in [0, 1]")
        object.__setattr__(self, "grid_plane_z_m", plane_z)
        object.__setattr__(self, "grid_size_m", grid_size)
        object.__setattr__(self, "color_grade_strength", grade)
        object.__setattr__(self, "light_strength", light)

    def as_payload(self) -> dict[str, object]:
        """Return the fixed JSON-compatible browser contract."""

        return {
            "background_color": self.background_color,
            "floor_tint": self.floor_tint,
            "grid_major_color": self.grid_major_color,
            "grid_minor_color": self.grid_minor_color,
            "x_axis_color": self.x_axis_color,
            "y_axis_color": self.y_axis_color,
            "key_light_color": self.key_light_color,
            "fill_light_color": self.fill_light_color,
            "rim_light_color": self.rim_light_color,
            "grid_plane_z_m": self.grid_plane_z_m,
            "grid_size_m": self.grid_size_m,
            "grid_divisions": self.grid_divisions,
            "color_grade_strength": self.color_grade_strength,
            "light_strength": self.light_strength,
        }


@dataclass(frozen=True)
class VisualizationTrail:
    """One bounded trajectory rendered over the browser frame.

    Points published for a trail use metres in the selected environment's
    local Z-up coordinate system.
    """

    trail_id: str
    label: str
    color: str = "#FFBE5C"
    max_points: int = 360
    line_width_px: float = 3.0
    marker_radius_px: float = 4.5
    tube_radius_m: float = 0.0035
    marker_radius_m: float = 0.008

    def __post_init__(self) -> None:
        if not isinstance(self.trail_id, str):
            raise TypeError("trail_id must be a string")
        if _IDENTIFIER_RE.fullmatch(self.trail_id) is None:
            raise ValueError(
                "trail_id must start with a letter and contain only letters, "
                "digits, '_' or '-'"
            )
        object.__setattr__(self, "label", _display_text(self.label, name="label"))
        object.__setattr__(self, "color", _hex_color(self.color, name="color"))
        if isinstance(self.max_points, bool) or not isinstance(self.max_points, int):
            raise TypeError("max_points must be an integer")
        if not 2 <= self.max_points <= 10_000:
            raise ValueError("max_points must be in [2, 10000]")
        width = _finite_number(self.line_width_px, name="line_width_px")
        radius = _finite_number(self.marker_radius_px, name="marker_radius_px")
        tube_radius = _finite_number(self.tube_radius_m, name="tube_radius_m")
        marker_radius = _finite_number(
            self.marker_radius_m,
            name="marker_radius_m",
        )
        if not 0.5 <= width <= 20.0:
            raise ValueError("line_width_px must be in [0.5, 20]")
        if not 1.0 <= radius <= 30.0:
            raise ValueError("marker_radius_px must be in [1, 30]")
        if not 0.0001 <= tube_radius <= 0.05:
            raise ValueError("tube_radius_m must be in [0.0001, 0.05]")
        if not 0.0005 <= marker_radius <= 0.1:
            raise ValueError("marker_radius_m must be in [0.0005, 0.1]")
        object.__setattr__(self, "line_width_px", width)
        object.__setattr__(self, "marker_radius_px", radius)
        object.__setattr__(self, "tube_radius_m", tube_radius)
        object.__setattr__(self, "marker_radius_m", marker_radius)

    def as_payload(self) -> dict[str, object]:
        return {
            "trail_id": self.trail_id,
            "label": self.label,
            "color": self.color,
            "line_width_px": self.line_width_px,
            "marker_radius_px": self.marker_radius_px,
            "tube_radius_m": self.tube_radius_m,
            "marker_radius_m": self.marker_radius_m,
        }


def normalize_trails(
    trails: Iterable[VisualizationTrail],
) -> tuple[VisualizationTrail, ...]:
    frozen = tuple(trails)
    if any(not isinstance(trail, VisualizationTrail) for trail in frozen):
        raise TypeError("trails must contain only VisualizationTrail instances")
    identifiers = tuple(trail.trail_id for trail in frozen)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("visualization trail identifiers must be unique")
    return frozen


@dataclass(frozen=True)
class BrowserActorState:
    """One visual actor pose and its ordered URDF joint positions."""

    actor_id: str
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    joint_positions: tuple[float, ...] = ()
    visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str):
            raise TypeError("actor_id must be a string")
        if _IDENTIFIER_RE.fullmatch(self.actor_id) is None:
            raise ValueError(
                "actor_id must start with a letter and contain only letters, "
                "digits, '_' or '-'"
            )
        position = _xyz(self.position_m, name="position_m")
        quaternion = _finite_tuple(
            self.quaternion_xyzw,
            length=4,
            name="quaternion_xyzw",
        )
        norm = sum(component * component for component in quaternion) ** 0.5
        if norm <= 1.0e-8:
            raise ValueError("quaternion_xyzw must have non-zero norm")
        normalized_quaternion = tuple(component / norm for component in quaternion)
        if isinstance(self.joint_positions, (str, bytes)):
            raise TypeError("joint_positions must be a numeric sequence")
        joint_positions = tuple(
            _finite_number(value, name=f"joint_positions[{index}]")
            for index, value in enumerate(self.joint_positions)
        )
        if not isinstance(self.visible, bool):
            raise TypeError("visible must be a bool")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", normalized_quaternion)
        object.__setattr__(self, "joint_positions", joint_positions)

    def as_payload(self) -> dict[str, object]:
        return {
            "position_m": self.position_m,
            "quaternion_xyzw": self.quaternion_xyzw,
            "joint_positions": self.joint_positions,
            "visible": self.visible,
        }


@dataclass(frozen=True)
class BrowserPresentationSnapshot:
    sequence: int
    recording_paused: bool
    view: dict[str, object] | None
    trails: dict[str, tuple[tuple[float, float, float], ...]]
    trail_sequences: dict[str, int]
    polyline_overlays: dict[
        str,
        tuple[tuple[tuple[float, float, float], ...], ...],
    ]
    polyline_overlay_sequences: dict[str, int]
    actors: dict[str, BrowserActorState]
    actor_sequence: int
    closed: bool


class BrowserPresentationState:
    """Latest-only camera and bounded 3-D trail snapshot for one browser."""

    def __init__(
        self,
        trails: Iterable[VisualizationTrail],
        *,
        actor_joint_names: Mapping[str, Sequence[str]] | None = None,
        polyline_limits: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        frozen = normalize_trails(trails)
        self._trail_specs = {trail.trail_id: trail for trail in frozen}
        self._trails = {
            trail.trail_id: deque(maxlen=trail.max_points) for trail in frozen
        }
        self._trail_sequences = {trail.trail_id: 0 for trail in frozen}
        self._polyline_limits: dict[str, tuple[int, int]] = {}
        for overlay_id, raw_limits in (polyline_limits or {}).items():
            if (
                not isinstance(overlay_id, str)
                or _IDENTIFIER_RE.fullmatch(overlay_id) is None
            ):
                raise ValueError(
                    f"invalid browser polyline overlay identifier: {overlay_id!r}"
                )
            if isinstance(raw_limits, (str, bytes)) or len(raw_limits) != 2:
                raise ValueError(
                    f"polyline limits for {overlay_id!r} must contain two integers"
                )
            max_polylines, max_points = raw_limits
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (max_polylines, max_points)
            ):
                raise ValueError(
                    f"polyline limits for {overlay_id!r} must be positive integers"
                )
            self._polyline_limits[overlay_id] = (max_polylines, max_points)
        self._polyline_overlays: dict[
            str,
            tuple[tuple[tuple[float, float, float], ...], ...],
        ] = {overlay_id: () for overlay_id in self._polyline_limits}
        self._polyline_overlay_sequences = {
            overlay_id: 0 for overlay_id in self._polyline_limits
        }
        self._condition = Condition()
        self._view: dict[str, object] | None = None
        self._actor_joint_names: dict[str, tuple[str, ...]] = {}
        for actor_id, names in (actor_joint_names or {}).items():
            if (
                not isinstance(actor_id, str)
                or _IDENTIFIER_RE.fullmatch(actor_id) is None
            ):
                raise ValueError(f"invalid browser actor identifier: {actor_id!r}")
            frozen_names = tuple(names)
            if any(not isinstance(name, str) or not name for name in frozen_names):
                raise ValueError(f"invalid joint names for actor {actor_id!r}")
            if len(frozen_names) != len(set(frozen_names)):
                raise ValueError(f"joint names for actor {actor_id!r} must be unique")
            self._actor_joint_names[actor_id] = frozen_names
        self._actors: dict[str, BrowserActorState] = {}
        self._actor_sequence = 0
        self._sequence = 0
        self._recording_paused = False
        self._closed = False

    @property
    def trail_ids(self) -> frozenset[str]:
        return frozenset(self._trail_specs)

    @property
    def actor_ids(self) -> frozenset[str]:
        return frozenset(self._actor_joint_names)

    @property
    def polyline_overlay_ids(self) -> frozenset[str]:
        return frozenset(self._polyline_limits)

    @property
    def recording_paused(self) -> bool:
        with self._condition:
            return self._recording_paused

    def set_recording_paused(self, paused: bool) -> int:
        """Publish a browser-wide MediaRecorder pause gate.

        Rendering and presentation updates continue while the gate is active;
        only an in-progress browser recording is automatically paused.
        """

        if not isinstance(paused, bool):
            raise TypeError("paused must be a bool")
        with self._condition:
            self._assert_open()
            if paused == self._recording_paused:
                return self._sequence
            self._recording_paused = paused
            return self._advance_locked()

    def update_view(
        self,
        eye: Sequence[Real],
        target: Sequence[Real],
        *,
        vertical_fov: Real,
        aspect: Real,
        client_sequence: int = 0,
    ) -> int:
        eye_xyz = _xyz(eye, name="eye")
        target_xyz = _xyz(target, name="target")
        fov = _finite_number(vertical_fov, name="vertical_fov")
        aspect_value = _finite_number(aspect, name="aspect")
        if not 0.0 < fov < pi:
            raise ValueError("vertical_fov must be in (0, pi)")
        if aspect_value <= 0.0:
            raise ValueError("aspect must be positive")
        if eye_xyz == target_xyz:
            raise ValueError("eye and target must differ")
        if (
            isinstance(client_sequence, bool)
            or not isinstance(client_sequence, int)
            or not 0 <= client_sequence <= (1 << 53) - 1
        ):
            raise ValueError(
                "client_sequence must be a non-negative JavaScript-safe integer"
            )
        view = {
            "eye": eye_xyz,
            "target": target_xyz,
            "vertical_fov": fov,
            "aspect": aspect_value,
            "client_sequence": client_sequence,
        }
        with self._condition:
            self._assert_open()
            if view == self._view:
                return self._sequence
            self._view = view
            return self._advance_locked()

    def append_trail_point(
        self,
        trail_id: str,
        point: Sequence[Real],
    ) -> int:
        normalized = _xyz(point, name="trail point")
        with self._condition:
            self._assert_open()
            trail = self._trail(trail_id)
            trail.append(normalized)
            return self._advance_locked(trail_ids=(trail_id,))

    def set_trail_points(
        self,
        trail_id: str,
        points: Iterable[Sequence[Real]],
    ) -> int:
        normalized = tuple(
            _xyz(point, name=f"trail point {index}")
            for index, point in enumerate(points)
        )
        with self._condition:
            self._assert_open()
            trail = self._trail(trail_id)
            trail.clear()
            trail.extend(normalized)
            return self._advance_locked(trail_ids=(trail_id,))

    def clear_trail(self, trail_id: str) -> int:
        with self._condition:
            self._assert_open()
            trail = self._trail(trail_id)
            if not trail:
                return self._sequence
            trail.clear()
            return self._advance_locked(trail_ids=(trail_id,))

    def set_polyline_overlay(
        self,
        overlay_id: str,
        polylines: Iterable[Iterable[Sequence[Real]]],
    ) -> int:
        """Replace one bounded browser-native polyline overlay."""

        normalized = self._normalize_polyline_overlays({overlay_id: polylines})
        with self._condition:
            self._assert_open()
            self._polyline_overlays[overlay_id] = normalized[overlay_id]
            return self._advance_locked(polyline_overlay_ids=(overlay_id,))

    def set_actor_states(
        self,
        states: Iterable[BrowserActorState],
    ) -> int:
        """Atomically replace the full configured actor-state snapshot."""

        frozen = self._normalize_actor_states(states)
        with self._condition:
            self._assert_open()
            self._actors = {state.actor_id: state for state in frozen}
            return self._advance_locked(actor_changed=True)

    def update_frame(
        self,
        *,
        actor_states: Iterable[BrowserActorState] | None = None,
        trail_points: Mapping[str, Sequence[Real]] | None = None,
        polyline_overlays: Mapping[
            str,
            Iterable[Iterable[Sequence[Real]]],
        ]
        | None = None,
    ) -> int:
        """Atomically publish actor poses, trails, and dynamic line batches."""

        frozen_states = (
            None if actor_states is None else self._normalize_actor_states(actor_states)
        )
        if trail_points is None:
            normalized_points: dict[str, tuple[float, float, float]] = {}
        else:
            if not isinstance(trail_points, Mapping):
                raise TypeError("trail_points must be a mapping")
            if any(not isinstance(trail_id, str) for trail_id in trail_points):
                raise TypeError("trail_points keys must be strings")
            normalized_points = {
                trail_id: _xyz(point, name=f"trail point {trail_id!r}")
                for trail_id, point in trail_points.items()
            }
            unknown = set(normalized_points) - set(self._trail_specs)
            if unknown:
                raise ValueError(f"unknown visualization trails: {sorted(unknown)}")
        normalized_polylines = self._normalize_polyline_overlays(
            {} if polyline_overlays is None else polyline_overlays
        )
        if frozen_states is None and not normalized_points and not normalized_polylines:
            raise ValueError("an atomic presentation frame must contain state")

        with self._condition:
            self._assert_open()
            if frozen_states is not None:
                self._actors = {state.actor_id: state for state in frozen_states}
            for trail_id, point in normalized_points.items():
                self._trails[trail_id].append(point)
            for overlay_id, polylines in normalized_polylines.items():
                self._polyline_overlays[overlay_id] = polylines
            return self._advance_locked(
                actor_changed=frozen_states is not None,
                trail_ids=tuple(normalized_points),
                polyline_overlay_ids=tuple(normalized_polylines),
            )

    def _normalize_polyline_overlays(
        self,
        values: Mapping[str, Iterable[Iterable[Sequence[Real]]]],
    ) -> dict[str, tuple[tuple[tuple[float, float, float], ...], ...]]:
        if not isinstance(values, Mapping):
            raise TypeError("polyline_overlays must be a mapping")
        if any(not isinstance(overlay_id, str) for overlay_id in values):
            raise TypeError("polyline_overlays keys must be strings")
        unknown = set(values) - set(self._polyline_limits)
        if unknown:
            raise ValueError(f"unknown polyline overlays: {sorted(unknown)}")
        normalized: dict[
            str,
            tuple[tuple[tuple[float, float, float], ...], ...],
        ] = {}
        for overlay_id, raw_polylines in values.items():
            polylines = tuple(
                tuple(
                    _xyz(point, name=f"polyline overlay {overlay_id!r} point")
                    for point in raw_polyline
                )
                for raw_polyline in raw_polylines
            )
            max_polylines, max_points = self._polyline_limits[overlay_id]
            if len(polylines) > max_polylines:
                raise ValueError(
                    f"polyline overlay {overlay_id!r} exceeds {max_polylines} lines"
                )
            if any(not polyline for polyline in polylines):
                raise ValueError("polyline overlays cannot contain empty polylines")
            if any(len(polyline) > max_points for polyline in polylines):
                raise ValueError(
                    f"polyline overlay {overlay_id!r} exceeds {max_points} points per line"
                )
            normalized[overlay_id] = polylines
        return normalized

    def _normalize_actor_states(
        self,
        states: Iterable[BrowserActorState],
    ) -> tuple[BrowserActorState, ...]:
        frozen = tuple(states)
        if any(not isinstance(state, BrowserActorState) for state in frozen):
            raise TypeError("states must contain only BrowserActorState instances")
        actor_ids = tuple(state.actor_id for state in frozen)
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("browser actor states must have unique identifiers")
        if set(actor_ids) != set(self._actor_joint_names):
            raise ValueError(
                "browser actor state identifiers do not match the scene contract"
            )
        for state in frozen:
            expected = len(self._actor_joint_names[state.actor_id])
            actual = len(state.joint_positions)
            if actual != expected:
                raise ValueError(
                    f"actor {state.actor_id!r} requires {expected} joint positions; "
                    f"received {actual}"
                )
        return frozen

    def wait_after(
        self,
        sequence: int,
        *,
        timeout: float = 1.0,
    ) -> BrowserPresentationSnapshot:
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("presentation sequence must be an integer")
        if sequence < 0:
            raise ValueError("presentation sequence must be non-negative")
        timeout_value = _finite_number(timeout, name="timeout")
        if timeout_value < 0.0:
            raise ValueError("timeout must be non-negative")

        with self._condition:
            if sequence > self._sequence:
                raise ValueError(
                    "presentation sequence is newer than the current state"
                )
            self._condition.wait_for(
                lambda: self._closed or self._sequence > sequence,
                timeout=timeout_value,
            )
            view = None if self._view is None else dict(self._view)
            trails = {
                trail_id: tuple(points) for trail_id, points in self._trails.items()
            }
            return BrowserPresentationSnapshot(
                sequence=self._sequence,
                recording_paused=self._recording_paused,
                view=view,
                trails=trails,
                trail_sequences=dict(self._trail_sequences),
                polyline_overlays=dict(self._polyline_overlays),
                polyline_overlay_sequences=dict(
                    self._polyline_overlay_sequences
                ),
                actors=dict(self._actors),
                actor_sequence=self._actor_sequence,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _trail(self, trail_id: str) -> deque[tuple[float, float, float]]:
        if not isinstance(trail_id, str):
            raise TypeError("trail_id must be a string")
        try:
            return self._trails[trail_id]
        except KeyError as error:
            raise ValueError(f"unknown visualization trail: {trail_id!r}") from error

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("browser presentation state is closed")

    def _advance_locked(
        self,
        *,
        actor_changed: bool = False,
        trail_ids: Sequence[str] = (),
        polyline_overlay_ids: Sequence[str] = (),
    ) -> int:
        self._sequence += 1
        if actor_changed:
            self._actor_sequence = self._sequence
        for trail_id in trail_ids:
            self._trail_sequences[trail_id] = self._sequence
        for overlay_id in polyline_overlay_ids:
            self._polyline_overlay_sequences[overlay_id] = self._sequence
        self._condition.notify_all()
        return self._sequence


__all__ = [
    "BrowserActorState",
    "BrowserPresentationSnapshot",
    "BrowserPresentationState",
    "BrowserSceneStyle",
    "VisualizationTrail",
    "normalize_trails",
]
