"""Simulator-independent contracts for a browser-native Three.js scene.

The producer explicitly describes visual actors and publishes their poses.  This
module never inspects an Isaac scene and never mutates USD, physics, or camera
observations.  URDF mesh files are exposed through an exact allowlist owned by
the visualization HTTP server.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from math import isfinite
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType
from urllib.parse import quote
import xml.etree.ElementTree as ET


_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SUPPORTED_MESH_SUFFIXES = frozenset({".glb", ".gltf", ".obj", ".stl"})


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must start with a letter and contain only letters, "
            "digits, '_' or '-'"
        )
    return value


def _color(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _HEX_COLOR_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use #RRGGBB syntax")
    return value.upper()


def _scene_color(
    value: str | Sequence[Real],
    *,
    name: str,
) -> str | tuple[float, float, float]:
    if isinstance(value, str):
        return _color(value, name=name)
    if isinstance(value, bytes) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three RGB components")
    components = tuple(
        _finite(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    if any(component < 0.0 or component > 1.0 for component in components):
        raise ValueError(f"{name} RGB components must be in [0, 1]")
    return components  # type: ignore[return-value]


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_tuple(
    value: Sequence[Real],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(
        _finite(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    if any(component <= 0.0 for component in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _display_text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value != value.strip() or any(character in value for character in "\r\n\0"):
        raise ValueError(f"{name} is not valid display text")
    if (not value and not allow_empty) or len(value) > 96:
        raise ValueError(f"{name} is not valid display text")
    return value


@dataclass(frozen=True)
class BrowserAssetFile:
    """One exact local file available at one browser-only HTTP path."""

    url_path: str
    local_path: Path
    content_type: str
    content_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.url_path, str) or not self.url_path.startswith(
            "/visualization-assets/"
        ):
            raise ValueError("url_path must use the visualization asset namespace")
        path = Path(self.local_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"browser asset is not a file: {path}")
        if not isinstance(self.content_type, str) or "/" not in self.content_type:
            raise ValueError("content_type must be a valid media type")
        object.__setattr__(self, "local_path", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.content_sha256 and self.content_sha256 != digest:
            raise ValueError("browser asset SHA-256 disagrees with its file")
        object.__setattr__(self, "content_sha256", digest)

    @property
    def versioned_url(self) -> str:
        """Return a content-addressed URL suitable for immutable caching."""

        return f"{self.url_path}?v={self.content_sha256}"


def _mesh_content_type(suffix: str) -> str:
    return {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "model/obj",
        ".stl": "model/stl",
    }[suffix]


@dataclass(frozen=True)
class BrowserUrdfActor:
    """A visual-only URDF actor rendered by the browser.

    Use :meth:`from_file` so collision geometry is removed and every remaining
    mesh reference is rewritten to an allowlisted same-origin URL.
    """

    actor_id: str
    urdf_xml: str
    joint_names: tuple[str, ...]
    link_colors: tuple[tuple[str, str], ...]
    fallback_color: str
    roughness: float
    metalness: float
    assets: tuple[BrowserAssetFile, ...]
    opacity: float = 1.0
    wireframe: bool = False

    @classmethod
    def from_file(
        cls,
        actor_id: str,
        urdf_path: str | Path,
        *,
        joint_names: Iterable[str],
        link_colors: Mapping[str, str] | None = None,
        fallback_color: str = "#AEB8C6",
        roughness: Real = 0.72,
        metalness: Real = 0.12,
        allow_external_mesh_symlinks: bool = False,
    ) -> BrowserUrdfActor:
        normalized_actor_id = _identifier(actor_id, name="actor_id")
        source_path = Path(urdf_path).resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"URDF path is not a file: {source_path}")
        source_root = source_path.parent

        root = ET.parse(source_path).getroot()
        if root.tag != "robot":
            raise ValueError(f"URDF root must be <robot>: {source_path}")

        # The browser scene is a visual twin, not a physics model.  Removing
        # collision elements prevents accidental display of coarse collision
        # meshes and keeps the HTTP allowlist limited to visible assets.
        for link in root.findall(".//link"):
            for collision in tuple(link.findall("collision")):
                link.remove(collision)

        available_links = {element.get("name") for element in root.findall(".//link")}
        available_joints = {element.get("name") for element in root.findall(".//joint")}
        normalized_joint_names = tuple(
            _identifier(name, name=f"joint_names[{index}]")
            for index, name in enumerate(joint_names)
        )
        if len(normalized_joint_names) != len(set(normalized_joint_names)):
            raise ValueError("joint_names must be unique")
        missing_joints = sorted(set(normalized_joint_names) - available_joints)
        if missing_joints:
            raise ValueError(f"URDF does not define joints: {missing_joints}")

        normalized_colors: list[tuple[str, str]] = []
        for link_name, value in (link_colors or {}).items():
            normalized_link = _identifier(link_name, name="link color name")
            if normalized_link not in available_links:
                raise ValueError(f"URDF does not define link: {normalized_link!r}")
            normalized_colors.append(
                (normalized_link, _color(value, name=f"color for {normalized_link}"))
            )
        normalized_colors.sort()

        assets_by_source: dict[Path, BrowserAssetFile] = {}
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                raise ValueError("URDF mesh is missing filename")
            if "://" in filename or filename.startswith("package:"):
                raise ValueError(
                    "browser URDF requires local relative mesh paths: " f"{filename!r}"
                )
            candidate = (source_root / filename).resolve(strict=True)
            try:
                relative = candidate.relative_to(source_root)
            except ValueError as error:
                # Trusted object packages may link individual visual meshes to
                # shared storage. Expose only that resolved file, never a root.
                relative = Path(filename)
                if (
                    not allow_external_mesh_symlinks
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise ValueError(
                        f"URDF mesh escapes its asset root: {filename!r}"
                    ) from error
            if not candidate.is_file():
                raise ValueError(f"URDF mesh is not a file: {candidate}")
            suffix = candidate.suffix.lower()
            if suffix not in _SUPPORTED_MESH_SUFFIXES:
                raise ValueError(f"unsupported browser mesh format: {candidate}")
            asset = assets_by_source.get(candidate)
            if asset is None:
                encoded_relative = "/".join(
                    quote(part, safe="") for part in relative.parts
                )
                url_path = (
                    f"/visualization-assets/{normalized_actor_id}/{encoded_relative}"
                )
                asset = BrowserAssetFile(
                    url_path=url_path,
                    local_path=candidate,
                    content_type=_mesh_content_type(suffix),
                )
                assets_by_source[candidate] = asset
            mesh.set("filename", asset.versioned_url)

        roughness_value = _finite(roughness, name="roughness")
        metalness_value = _finite(metalness, name="metalness")
        if not 0.0 <= roughness_value <= 1.0:
            raise ValueError("roughness must be in [0, 1]")
        if not 0.0 <= metalness_value <= 1.0:
            raise ValueError("metalness must be in [0, 1]")

        return cls(
            actor_id=normalized_actor_id,
            urdf_xml=ET.tostring(root, encoding="unicode"),
            joint_names=normalized_joint_names,
            link_colors=tuple(normalized_colors),
            fallback_color=_color(fallback_color, name="fallback_color"),
            roughness=roughness_value,
            metalness=metalness_value,
            assets=tuple(assets_by_source.values()),
        )

    def as_payload(self) -> dict[str, object]:
        payload = {
            "actor_id": self.actor_id,
            "urdf_xml": self.urdf_xml,
            "joint_names": self.joint_names,
            "link_colors": dict(self.link_colors),
            "fallback_color": self.fallback_color,
            "roughness": self.roughness,
            "metalness": self.metalness,
        }
        if self.opacity != 1.0 or self.wireframe:
            if not 0.0 <= self.opacity <= 1.0 or not isinstance(self.wireframe, bool):
                raise ValueError("invalid URDF presentation opacity/wireframe")
            payload.update(opacity=self.opacity, wireframe=self.wireframe)
        return payload


@dataclass(frozen=True)
class BrowserPrimitiveActor:
    """A colored box or cylinder whose pose is supplied at runtime."""

    actor_id: str
    geometry: str
    dimensions_m: tuple[float, ...]
    color: str | tuple[float, float, float]
    roughness: float = 0.82
    metalness: float = 0.06
    opacity: float = 1.0
    wireframe: bool = False
    label: str = ""

    @classmethod
    def box(
        cls,
        actor_id: str,
        size_m: Sequence[Real],
        *,
        color: str | Sequence[Real],
        roughness: Real = 0.82,
        metalness: Real = 0.06,
        opacity: Real = 1.0,
        wireframe: bool = False,
        label: str = "",
    ) -> BrowserPrimitiveActor:
        return cls(
            actor_id=actor_id,
            geometry="box",
            dimensions_m=_positive_tuple(size_m, length=3, name="size_m"),
            color=color,
            roughness=roughness,
            metalness=metalness,
            opacity=opacity,
            wireframe=wireframe,
            label=label,
        )

    @classmethod
    def cylinder(
        cls,
        actor_id: str,
        *,
        radius_m: Real,
        height_m: Real,
        color: str | Sequence[Real],
        roughness: Real = 0.82,
        metalness: Real = 0.06,
        opacity: Real = 1.0,
        wireframe: bool = False,
        label: str = "",
    ) -> BrowserPrimitiveActor:
        return cls(
            actor_id=actor_id,
            geometry="cylinder",
            dimensions_m=(
                _finite(radius_m, name="radius_m"),
                _finite(height_m, name="height_m"),
            ),
            color=color,
            roughness=roughness,
            metalness=metalness,
            opacity=opacity,
            wireframe=wireframe,
            label=label,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "actor_id", _identifier(self.actor_id, name="actor_id")
        )
        if self.geometry == "box":
            dimensions = _positive_tuple(
                self.dimensions_m,
                length=3,
                name="dimensions_m",
            )
        elif self.geometry == "cylinder":
            dimensions = _positive_tuple(
                self.dimensions_m,
                length=2,
                name="dimensions_m",
            )
        else:
            raise ValueError("geometry must be 'box' or 'cylinder'")
        roughness = _finite(self.roughness, name="roughness")
        metalness = _finite(self.metalness, name="metalness")
        opacity = _finite(self.opacity, name="opacity")
        if (
            not 0.0 <= roughness <= 1.0
            or not 0.0 <= metalness <= 1.0
            or not 0.0 < opacity <= 1.0
        ):
            raise ValueError(
                "roughness/metalness must be in [0, 1] and opacity in (0, 1]"
            )
        if not isinstance(self.wireframe, bool):
            raise TypeError("wireframe must be a bool")
        object.__setattr__(self, "dimensions_m", dimensions)
        object.__setattr__(self, "color", _scene_color(self.color, name="color"))
        object.__setattr__(self, "roughness", roughness)
        object.__setattr__(self, "metalness", metalness)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(
            self,
            "label",
            _display_text(self.label, name="label", allow_empty=True),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "geometry": self.geometry,
            "dimensions_m": self.dimensions_m,
            "color": self.color,
            "roughness": self.roughness,
            "metalness": self.metalness,
            "opacity": self.opacity,
            "wireframe": self.wireframe,
            "label": self.label,
        }


@dataclass(frozen=True)
class BrowserPolylineOverlay:
    """One bounded batch of dynamic browser-only 3-D polylines.

    The Three.js client allocates fixed-size line/point buffers once and only
    updates their contents.  This is intended for small vector fields and
    reference paths that would be wasteful to express as many tube meshes.
    """

    overlay_id: str
    label: str
    color: str
    max_polylines: int
    max_points_per_polyline: int
    point_size_px: float = 3.0
    opacity: float = 1.0
    depth_test: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overlay_id",
            _identifier(self.overlay_id, name="overlay_id"),
        )
        object.__setattr__(self, "label", _display_text(self.label, name="label"))
        object.__setattr__(self, "color", _color(self.color, name="color"))
        for name in ("max_polylines", "max_points_per_polyline"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= 4096:
                raise ValueError(f"{name} must be in [1, 4096]")
        if self.max_polylines * self.max_points_per_polyline > 100_000:
            raise ValueError("polyline overlay capacity must not exceed 100000 points")
        point_size = _finite(self.point_size_px, name="point_size_px")
        opacity = _finite(self.opacity, name="opacity")
        if not 1.0 <= point_size <= 24.0:
            raise ValueError("point_size_px must be in [1, 24]")
        if not 0.0 < opacity <= 1.0:
            raise ValueError("opacity must be in (0, 1]")
        if not isinstance(self.depth_test, bool):
            raise TypeError("depth_test must be a bool")
        object.__setattr__(self, "point_size_px", point_size)
        object.__setattr__(self, "opacity", opacity)

    def as_payload(self) -> dict[str, object]:
        return {
            "overlay_id": self.overlay_id,
            "label": self.label,
            "color": self.color,
            "max_polylines": self.max_polylines,
            "max_points_per_polyline": self.max_points_per_polyline,
            "point_size_px": self.point_size_px,
            "opacity": self.opacity,
            "depth_test": self.depth_test,
        }


@dataclass(frozen=True)
class BrowserOverlayVisibilityControl:
    """One browser-local button controlling a declared overlay group.

    The control only changes Three.js visibility.  It never changes producer
    publication, simulator state, policy observations, or recorded metrics.
    """

    control_id: str
    label: str
    overlay_ids: tuple[str, ...]
    default_visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_id",
            _identifier(self.control_id, name="control_id"),
        )
        object.__setattr__(self, "label", _display_text(self.label, name="label"))
        if isinstance(self.overlay_ids, (str, bytes)):
            raise TypeError("overlay_ids must be a sequence of identifiers")
        normalized_overlay_ids = tuple(
            _identifier(value, name=f"overlay_ids[{index}]")
            for index, value in enumerate(self.overlay_ids)
        )
        if not normalized_overlay_ids:
            raise ValueError("overlay_ids must not be empty")
        if len(normalized_overlay_ids) != len(set(normalized_overlay_ids)):
            raise ValueError("overlay_ids must be unique within a visibility control")
        if not isinstance(self.default_visible, bool):
            raise TypeError("default_visible must be a bool")
        object.__setattr__(self, "overlay_ids", normalized_overlay_ids)

    def as_payload(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "label": self.label,
            "overlay_ids": self.overlay_ids,
            "default_visible": self.default_visible,
        }


@dataclass(frozen=True)
class BrowserThreeScene:
    """Static browser-native scene definition and exact actor contract."""

    urdf_actors: tuple[BrowserUrdfActor, ...] = ()
    primitive_actors: tuple[BrowserPrimitiveActor, ...] = ()
    polyline_overlays: tuple[BrowserPolylineOverlay, ...] = ()
    overlay_visibility_controls: tuple[BrowserOverlayVisibilityControl, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(actor, BrowserUrdfActor) for actor in self.urdf_actors):
            raise TypeError("urdf_actors must contain BrowserUrdfActor instances")
        if any(
            not isinstance(actor, BrowserPrimitiveActor)
            for actor in self.primitive_actors
        ):
            raise TypeError(
                "primitive_actors must contain BrowserPrimitiveActor instances"
            )
        if any(
            not isinstance(overlay, BrowserPolylineOverlay)
            for overlay in self.polyline_overlays
        ):
            raise TypeError(
                "polyline_overlays must contain BrowserPolylineOverlay instances"
            )
        if any(
            not isinstance(control, BrowserOverlayVisibilityControl)
            for control in self.overlay_visibility_controls
        ):
            raise TypeError(
                "overlay_visibility_controls must contain "
                "BrowserOverlayVisibilityControl instances"
            )
        actor_ids = tuple(
            actor.actor_id for actor in (*self.urdf_actors, *self.primitive_actors)
        )
        if not actor_ids:
            raise ValueError("browser Three.js scene must contain at least one actor")
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("browser Three.js actor identifiers must be unique")
        overlay_ids = tuple(overlay.overlay_id for overlay in self.polyline_overlays)
        if len(overlay_ids) != len(set(overlay_ids)):
            raise ValueError("browser polyline overlay identifiers must be unique")
        control_ids = tuple(
            control.control_id for control in self.overlay_visibility_controls
        )
        if len(control_ids) != len(set(control_ids)):
            raise ValueError(
                "browser overlay visibility control identifiers must be unique"
            )
        controlled_overlay_ids = tuple(
            overlay_id
            for control in self.overlay_visibility_controls
            for overlay_id in control.overlay_ids
        )
        unknown_controlled_overlays = set(controlled_overlay_ids) - set(overlay_ids)
        if unknown_controlled_overlays:
            raise ValueError(
                "browser overlay visibility controls reference unknown overlays: "
                f"{sorted(unknown_controlled_overlays)}"
            )
        if len(controlled_overlay_ids) != len(set(controlled_overlay_ids)):
            raise ValueError(
                "a browser polyline overlay may belong to at most one visibility control"
            )
        asset_paths = tuple(
            asset.url_path for actor in self.urdf_actors for asset in actor.assets
        )
        if len(asset_paths) != len(set(asset_paths)):
            raise ValueError("browser asset URL paths must be unique")

    @property
    def actor_joint_names(self) -> Mapping[str, tuple[str, ...]]:
        values = {actor.actor_id: actor.joint_names for actor in self.urdf_actors}
        values.update({actor.actor_id: () for actor in self.primitive_actors})
        return MappingProxyType(values)

    @property
    def asset_files(self) -> Mapping[str, BrowserAssetFile]:
        return MappingProxyType(
            {
                asset.url_path: asset
                for actor in self.urdf_actors
                for asset in actor.assets
            }
        )

    @property
    def polyline_limits(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType(
            {
                overlay.overlay_id: (
                    overlay.max_polylines,
                    overlay.max_points_per_polyline,
                )
                for overlay in self.polyline_overlays
            }
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "urdf_actors": [actor.as_payload() for actor in self.urdf_actors],
            "primitive_actors": [actor.as_payload() for actor in self.primitive_actors],
            "polyline_overlays": [
                overlay.as_payload() for overlay in self.polyline_overlays
            ],
            "overlay_visibility_controls": [
                control.as_payload() for control in self.overlay_visibility_controls
            ],
        }


__all__ = [
    "BrowserAssetFile",
    "BrowserOverlayVisibilityControl",
    "BrowserPolylineOverlay",
    "BrowserPrimitiveActor",
    "BrowserThreeScene",
    "BrowserUrdfActor",
]
