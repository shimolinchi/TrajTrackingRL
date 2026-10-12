"""Pinned simulator source and physics configuration for trajectory tracking.

This module is intentionally independent from ``residual_finetune`` and safe to
import before Isaac Lab starts.  It validates the exact Interprior checkout and
assets used by the one-trajectory experiment; episode/reset semantics remain in
``env.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Literal
import xml.etree.ElementTree as ET


class TrajectorySceneConfigError(RuntimeError):
    """The pinned trajectory scene cannot be reproduced."""


@dataclass(frozen=True, slots=True)
class TrajectorySceneRuntimeSpec:
    """Explicitly selected runtime role for the tracked object."""

    object_physics_mode: Literal["dynamic", "disabled"] = "dynamic"
    contact_sensors_enabled: bool = False
    table_collision: Literal["source", "flat_mesh_v1"] = "source"

    def __post_init__(self) -> None:
        if self.table_collision not in ("source", "flat_mesh_v1"):
            raise ValueError("unsupported table collision representation")
        if self.object_physics_mode not in ("dynamic", "disabled"):
            raise ValueError("unsupported trajectory object physics mode")
        if not isinstance(self.contact_sensors_enabled, bool):
            raise TypeError("contact_sensors_enabled must be boolean")

    def to_manifest(self) -> dict[str, object]:
        disabled = self.object_physics_mode == "disabled"
        payload: dict[str, object] = {
            "object_physics_mode": self.object_physics_mode,
            "object_kinematic": disabled,
            "object_gravity_enabled": not disabled,
            "object_collision_enabled": not disabled,
        }
        if self.table_collision != "source":
            from .table_collision import TABLE_COLLISION_SEGMENTS
            payload["table_collision"] = {
                "representation": self.table_collision,
                "segments": TABLE_COLLISION_SEGMENTS,
                "approximation": "none",
            }
        if self.contact_sensors_enabled:
            payload["contact_sensors"] = {
                "enabled": True,
                "bodies": [
                    f"right_finger{index}_link4" for index in range(1, 6)
                ],
                "filter": "tracked Object prim in the same environment",
                "reported_force": "normal_force_world_N",
            }
        return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise TrajectorySceneConfigError(
            f"cannot inspect Interprior checkout at {root}: {detail}"
        )
    return result.stdout.strip()


def _project_relative_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise TrajectorySceneConfigError(f"scene path escapes source root: {relative}")
    if not candidate.is_file():
        raise TrajectorySceneConfigError(f"scene asset is missing: {candidate}")
    return candidate


@dataclass(frozen=True, slots=True)
class SceneAssetSpec:
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("scene asset paths must be source-relative")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("scene asset sha256 must be a lowercase digest")

    def to_manifest(self) -> dict[str, str]:
        return {"path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class GroundAssetSpec:
    root_usd: SceneAssetSpec
    dependencies: tuple[SceneAssetSpec, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "root_usd": self.root_usd.to_manifest(),
            "dependencies": [asset.to_manifest() for asset in self.dependencies],
        }


@dataclass(frozen=True, slots=True)
class ScenePhysicsSpec:
    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_type: int = 1
    enable_external_forces_every_iteration: bool = True
    min_position_iteration_count: int = 8
    max_position_iteration_count: int = 64
    min_velocity_iteration_count: int = 0
    max_velocity_iteration_count: int = 64
    robot_solver_position_iterations: int = 16
    robot_solver_velocity_iterations: int = 8
    bounce_threshold_velocity: float = 0.2
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025
    gpu_max_rigid_contact_count: int = 16_777_216
    gpu_max_rigid_patch_count: int = 8_388_608
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    default_friction: float = 0.5
    fingertip_friction: float = 1.5
    restitution: float = 0.0

    def to_manifest(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TrajectoryIsaacSceneSpec:
    version: str
    upstream_root_relative: str
    expected_revision: str
    robot: SceneAssetSpec
    table: SceneAssetSpec
    object_asset: SceneAssetSpec
    ground: GroundAssetSpec
    physics: ScenePhysicsSpec
    robot_self_collision_enabled: bool = False

    def __post_init__(self) -> None:
        root = Path(self.upstream_root_relative)
        if root.is_absolute() or ".." in root.parts:
            raise ValueError("scene upstream root must be project-relative")
        if len(self.expected_revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.expected_revision
        ):
            raise ValueError("scene revision must be a lowercase Git object ID")

    def to_manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "upstream_root": self.upstream_root_relative,
            "expected_revision": self.expected_revision,
            "robot": self.robot.to_manifest(),
            "table": self.table.to_manifest(),
            "object": self.object_asset.to_manifest(),
            "ground": self.ground.to_manifest(),
            "physics": self.physics.to_manifest(),
            "robot_self_collision_enabled": self.robot_self_collision_enabled,
        }


TRAJECTORY_ISAAC_SCENE = TrajectoryIsaacSceneSpec(
    version="trajectory_isaac_scene_v2",
    upstream_root_relative="submodules/Interprior",
    expected_revision="32198186b64bcef98d73b5f372c8adf74b3bbaf0",
    robot=SceneAssetSpec(
        "assets/urdf/xarm7_wuji_right_description/xarm7_wuji_right.urdf",
        "8b4cf0b496adb800b2403d04355210adfcf1683426af44c2939bdb42203a53b2",
    ),
    table=SceneAssetSpec(
        "assets/urdf/table_round_large.urdf",
        "7b798837ce6367aa6c44d5b460dbc7a1df4e3f9f2743578d58a8a73c1d38d2c3",
    ),
    object_asset=SceneAssetSpec(
        "assets/urdf/cube_0p06m.urdf",
        "2422f85dcbb91d3ada31b19bf6ce5067ee5cdf693af1ce667355f2f72e096f27",
    ),
    ground=GroundAssetSpec(
        root_usd=SceneAssetSpec(
            "Isaac/Environments/Grid/default_environment.usd",
            "8a21c317d638d33a4e6c20c958a7ed8f4d8c5195efd7e20f1b85178b4a15638a",
        ),
        dependencies=(
            SceneAssetSpec(
                "Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png",
                "940b96ee8c8e53bd1acbc770d4ca0ac98bed9c96439ab386d1b27dbc9790cc4e",
            ),
            SceneAssetSpec(
                "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png",
                "deab39088eaf39ed7be951966c5e6bd5e58b90ab64743b0cdc8e5579fd5a99d9",
            ),
            SceneAssetSpec(
                "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_blue.png",
                "44fc0e82007bb0bd94631cd7ec45986a71ed74b9413899fbffac13efe151be4b",
            ),
        ),
    ),
    physics=ScenePhysicsSpec(),
)


@dataclass(frozen=True, slots=True)
class ResolvedTrajectorySceneSource:
    spec: TrajectoryIsaacSceneSpec
    upstream_root: Path
    robot_urdf: Path
    table_urdf: Path
    object_urdf: Path
    ground_usd: Path
    object_override: "ResolvedSceneObjectOverride | None" = None
    object_variants: tuple["ResolvedTrajectorySceneSource", ...] = ()
    object_layout: tuple[int, ...] = ()

    @property
    def object_sha256(self) -> str:
        return (
            self.spec.object_asset.sha256
            if self.object_override is None
            else self.object_override.sha256
        )


@dataclass(frozen=True, slots=True)
class ResolvedSceneObjectOverride:
    """A checksummed task-data object asset without a personal absolute path."""

    root_env: str
    relative_path: str
    sha256: str
    object_uid: str
    object_id: str
    missing_inertial_density_kg_m3: float
    collider_type: Literal["convex_hull", "convex_decomposition"]
    replace_cylinders_with_capsules: bool
    object_scale: tuple[float, float, float]
    object_size_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("scene object override must be root-relative")
        if not self.root_env or not self.object_uid or not self.object_id:
            raise ValueError("scene object override identity must be non-empty")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("scene object override SHA-256 is invalid")
        if self.missing_inertial_density_kg_m3 <= 0.0:
            raise ValueError("scene object density must be positive")
        if self.collider_type not in ("convex_hull", "convex_decomposition"):
            raise ValueError("scene object collider type is unsupported")
        if not isinstance(self.replace_cylinders_with_capsules, bool):
            raise TypeError("scene object cylinder replacement must be boolean")
        for name, values in (
            ("object_scale", self.object_scale),
            ("object_size_m", self.object_size_m),
        ):
            if len(values) != 3 or any(value <= 0.0 for value in values):
                raise ValueError(f"scene {name} must contain three positive values")

    def to_manifest(self) -> dict[str, object]:
        return {
            "source": "trajectory_data_selection",
            "root_env": self.root_env,
            "path": self.relative_path,
            "sha256": self.sha256,
            "object_uid": self.object_uid,
            "object_id": self.object_id,
            "missing_inertial_density_kg_m3": self.missing_inertial_density_kg_m3,
            "collider_type": self.collider_type,
            "replace_cylinders_with_capsules": self.replace_cylinders_with_capsules,
            "object_scale": list(self.object_scale),
            "object_size_m": list(self.object_size_m),
        }


def override_trajectory_scene_object(
    source: ResolvedTrajectorySceneSource,
    *,
    object_urdf: str | Path,
    root_env: str,
    relative_path: str,
    sha256_digest: str,
    object_uid: str,
    object_id: str,
    missing_inertial_density_kg_m3: float,
    collider_type: Literal["convex_hull", "convex_decomposition"],
    replace_cylinders_with_capsules: bool,
    object_scale: tuple[float, float, float],
    object_size_m: tuple[float, float, float],
) -> ResolvedTrajectorySceneSource:
    """Return a scene source with one verified object selected by a data adapter."""

    from dataclasses import replace

    path = Path(object_urdf).resolve()
    if not path.is_file():
        raise TrajectorySceneConfigError(f"trajectory object URDF is missing: {path}")
    actual = _sha256_file(path)
    if actual != sha256_digest:
        raise TrajectorySceneConfigError(
            "trajectory object URDF hash mismatch: "
            f"expected={sha256_digest}, actual={actual}"
        )
    override = ResolvedSceneObjectOverride(
        root_env=root_env,
        relative_path=relative_path,
        sha256=sha256_digest,
        object_uid=object_uid,
        object_id=object_id,
        missing_inertial_density_kg_m3=missing_inertial_density_kg_m3,
        collider_type=collider_type,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        object_scale=object_scale,
        object_size_m=object_size_m,
    )
    return replace(source, object_urdf=path, object_override=override)


def override_trajectory_scene_physics(
    source: ResolvedTrajectorySceneSource,
    *,
    enable_external_forces_every_iteration: bool,
) -> ResolvedTrajectorySceneSource:
    """Return a scene source with a dataset-recorded PhysX compatibility flag."""

    from dataclasses import replace

    physics = replace(
        source.spec.physics,
        enable_external_forces_every_iteration=bool(
            enable_external_forces_every_iteration
        ),
    )
    return replace(source, spec=replace(source.spec, physics=physics))


def resolved_trajectory_scene_manifest(
    source: ResolvedTrajectorySceneSource,
) -> dict[str, object]:
    payload = source.spec.to_manifest()
    if source.object_override is not None:
        payload = {
            **payload,
            "version": f"{source.spec.version}+data_object_override_v1",
            "object": source.object_override.to_manifest(),
        }
    if source.object_variants:
        payload = {**payload, "version": f"{source.spec.version}+static_object_layout_v1",
                   "objects": [item.object_override.to_manifest() for item in source.object_variants],
                   "object_layout": list(source.object_layout),
                   "object_layout_semantics": "environment_index_modulo_layout_length"}
    return payload


def resolve_trajectory_scene_source(
    *,
    project_root: str | Path,
    scene_asset_root: str | Path,
    spec: TrajectoryIsaacSceneSpec = TRAJECTORY_ISAAC_SCENE,
) -> ResolvedTrajectorySceneSource:
    """Resolve current gitlinks and byte-identical scene assets."""

    project = Path(project_root).resolve()
    upstream = (project / spec.upstream_root_relative).resolve()
    if not upstream.is_relative_to(project) or not upstream.is_dir():
        raise TrajectorySceneConfigError("Interprior scene checkout is missing")
    revision = _git(upstream, "rev-parse", "HEAD")
    if revision != spec.expected_revision:
        raise TrajectorySceneConfigError(
            f"Interprior revision mismatch: expected={spec.expected_revision}, actual={revision}"
        )
    if _git(upstream, "status", "--porcelain"):
        raise TrajectorySceneConfigError("Interprior scene checkout must be clean")

    resolved_assets: list[Path] = []
    for asset in (spec.robot, spec.table, spec.object_asset):
        path = _project_relative_file(upstream, asset.relative_path)
        actual = _sha256_file(path)
        if actual != asset.sha256:
            raise TrajectorySceneConfigError(
                f"scene asset hash mismatch: expected={asset.sha256}, actual={actual}, path={path}"
            )
        resolved_assets.append(path)

    robot_urdf = resolved_assets[0]
    missing_meshes: list[str] = []
    for mesh in ET.parse(robot_urdf).getroot().findall(".//mesh"):
        relative = mesh.get("filename", "")
        mesh_path = (robot_urdf.parent / relative).resolve()
        if not relative or not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            missing_meshes.append(relative or "<empty>")
    if missing_meshes:
        raise TrajectorySceneConfigError(
            f"robot URDF has missing mesh assets: {missing_meshes}"
        )

    ground_root = Path(scene_asset_root).expanduser().resolve()
    ground_paths: list[Path] = []
    for asset in (spec.ground.root_usd, *spec.ground.dependencies):
        path = _project_relative_file(ground_root, asset.relative_path)
        actual = _sha256_file(path)
        if actual != asset.sha256:
            raise TrajectorySceneConfigError(
                f"ground asset hash mismatch: expected={asset.sha256}, actual={actual}, path={path}"
            )
        ground_paths.append(path)

    return ResolvedTrajectorySceneSource(
        spec=spec,
        upstream_root=upstream,
        robot_urdf=resolved_assets[0],
        table_urdf=resolved_assets[1],
        object_urdf=resolved_assets[2],
        ground_usd=ground_paths[0],
    )


__all__ = [
    "TRAJECTORY_ISAAC_SCENE",
    "GroundAssetSpec",
    "ResolvedTrajectorySceneSource",
    "ResolvedSceneObjectOverride",
    "SceneAssetSpec",
    "ScenePhysicsSpec",
    "TrajectoryIsaacSceneSpec",
    "TrajectorySceneRuntimeSpec",
    "TrajectorySceneConfigError",
    "resolve_trajectory_scene_source",
    "override_trajectory_scene_object",
    "override_trajectory_scene_physics",
    "resolved_trajectory_scene_manifest",
]
