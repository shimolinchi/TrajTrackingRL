"""Read-only schema-v10 trajectory loader without policy imports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from trajectory_tracking.configuration import (
    TrajectoryConfigError,
    TrajectoryDirectConfig,
)
from trajectory_tracking.data.reference import (
    ReferenceCollection,
    ReferenceTrajectory,
    ReferenceTrajectoryBank,
)
from trajectory_tracking.environment.scene_config import TRAJECTORY_ISAAC_SCENE
from trajectory_tracking.workspace import SCENE_ASSETS
from trajectory_tracking.utils.kinematics import (
    WujiHandKinematics,
    finite_difference_linear_velocity,
    quaternion_angular_velocity_wxyz,
)


INTERPRIOR_MANIPTRANS_TRAJECTORY_ADAPTER_VERSION = (
    "interprior_schema_v10_single_v3_xarm7_wuji25"
)
INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_ADAPTER_VERSION = (
    "interprior_schema_v10_selection_prefix_bank_v1_xarm7_wuji25"
)
INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_STRIDED_ADAPTER_VERSION = (
    "interprior_schema_v10_selection_prefix_bank_v2_source_stride_xarm7_wuji25"
)


def interprior_trajectory_adapter_version(config: TrajectoryDirectConfig) -> str:
    """Return the materialization contract selected by the data config."""

    if config.data.adapter != "interprior":
        from .registry import resolve_data_adapter

        return resolve_data_adapter(config.data.adapter).VERSION
    if config.timing.source_frame_stride != 1:
        return INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_STRIDED_ADAPTER_VERSION
    if config.data.trajectory_count == 1:
        return INTERPRIOR_MANIPTRANS_TRAJECTORY_ADAPTER_VERSION
    return INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_ADAPTER_VERSION


def verify_historical_policy_evaluation_adapter_version(
    *,
    training_version: str,
    runtime_version: str,
) -> str:
    """Require the same materialization adapter for training and evaluation."""

    if not training_version or not runtime_version:
        raise ValueError("adapter versions must be non-empty")
    if training_version != runtime_version:
        raise ValueError(
            "checkpoint adapter does not match the runtime trajectory adapter: "
            f"{(training_version, runtime_version)}"
        )
    return training_version


@dataclass(frozen=True, slots=True)
class InterpriorSelectionEntry:
    episode_uid: str
    attempt_uid: str
    shard_relative: str
    slot: int


@dataclass(frozen=True, slots=True)
class InterpriorTrajectorySource:
    entry: InterpriorSelectionEntry
    upstream_root: Path
    shard: Path


@dataclass(slots=True)
class _ShardRuntime:
    episode_table: Any
    frames: Any
    metadata_by_slot: dict[int, Mapping[str, Any]]


def _selection_entries(
    config: TrajectoryDirectConfig,
    project_root: Path,
) -> tuple[InterpriorSelectionEntry, ...]:
    payload = json.loads(
        config.data.selection_path(project_root).read_text(encoding="utf-8")
    )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != config.data.selection_count:
        raise TrajectoryConfigError("selection episodes are missing or changed")
    start = config.data.selection_index
    rows = episodes[start : start + config.data.trajectory_count]
    entries: list[InterpriorSelectionEntry] = []
    for selection_index, row in enumerate(rows, start=start):
        if not isinstance(row, Mapping):
            raise TrajectoryConfigError(
                f"selection row {selection_index} must be a mapping"
            )
        required = ("episode_uid", "attempt_uid", "shard_relative", "episode")
        if any(name not in row for name in required):
            raise TrajectoryConfigError(
                f"selection row {selection_index} is incomplete"
            )
        episode_uid = str(row["episode_uid"])
        attempt_uid = str(row["attempt_uid"])
        shard_relative = str(row["shard_relative"])
        slot = row["episode"]
        for name, digest in (
            ("episode_uid", episode_uid),
            ("attempt_uid", attempt_uid),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise TrajectoryConfigError(
                    f"selection row {selection_index} {name} is invalid"
                )
        relative = Path(shard_relative)
        if (
            relative.name != shard_relative
            or not shard_relative.startswith("derived_")
            or not shard_relative.removeprefix("derived_").isdigit()
        ):
            raise TrajectoryConfigError(
                f"selection row {selection_index} shard name is invalid"
            )
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise TrajectoryConfigError(
                f"selection row {selection_index} slot is invalid"
            )
        entries.append(
            InterpriorSelectionEntry(
                episode_uid=episode_uid,
                attempt_uid=attempt_uid,
                shard_relative=shard_relative,
                slot=slot,
            )
        )
    if entries[0].episode_uid != config.data.expected_episode_uid:
        raise TrajectoryConfigError("first selected episode UID disagrees with config")
    episode_ids = tuple(entry.episode_uid for entry in entries)
    if len(set(episode_ids)) != len(episode_ids):
        raise TrajectoryConfigError("selected trajectory prefix contains duplicate IDs")
    return tuple(entries)


def resolve_interprior_trajectory_sources(
    config: TrajectoryDirectConfig,
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[InterpriorTrajectorySource, ...]:
    """Resolve and verify the configured contiguous selection prefix once."""

    root = Path(project_root).resolve()
    if config.data.adapter != "interprior":
        from .registry import resolve_data_adapter

        return resolve_data_adapter(config.data.adapter).resolve_sources(
            config, root, environ
        )
    # The collector revision is provenance, not a dependency on its repository.
    upstream = root
    entries = _selection_entries(config, root)
    dataset_root = config.data.dataset_root(environ)
    shard_root = (dataset_root / "derived" / config.data.outcome / "shards").resolve()
    sources: list[InterpriorTrajectorySource] = []
    for entry in entries:
        shard = (shard_root / entry.shard_relative).resolve()
        if not shard.is_relative_to(shard_root):
            raise TrajectoryConfigError("selected shard escapes dataset root")
        if not shard.is_dir():
            raise FileNotFoundError(shard)
        sources.append(
            InterpriorTrajectorySource(
                entry=entry,
                upstream_root=upstream,
                shard=shard,
            )
        )
    return tuple(sources)


def resolve_trajectory_scene_source_for_data(
    config: TrajectoryDirectConfig,
    scene_source: Any,
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Apply an optional adapter-owned, checksummed scene-object override."""

    if config.data.adapter == "interprior":
        return scene_source
    from .registry import resolve_data_adapter

    resolver = getattr(resolve_data_adapter(config.data.adapter), "resolve_scene_source", None)
    if resolver is None:
        return scene_source
    return resolver(config, scene_source, Path(project_root).resolve(), environ)


def resolve_single_interprior_source(
    config: TrajectoryDirectConfig,
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> InterpriorTrajectorySource:
    """Compatibility wrapper for historical single-trajectory callers."""

    sources = resolve_interprior_trajectory_sources(
        config,
        project_root=project_root,
        environ=environ,
    )
    if len(sources) != 1:
        raise TrajectoryConfigError("single-source resolver requires trajectory_count=1")
    return sources[0]


def _metadata_by_slot(shard: Path) -> dict[int, Mapping[str, Any]]:
    table = shard / "source_episode_table.jsonl"
    result: dict[int, Mapping[str, Any]] = {}
    with table.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            slot = int(row.get("derived_slot", -1))
            raw_row = row.get("raw_episode_row")
            metadata = raw_row.get("metadata") if isinstance(raw_row, Mapping) else None
            if slot < 0 or not isinstance(metadata, Mapping) or slot in result:
                raise ValueError(f"invalid source metadata row in {table}")
            result[slot] = metadata
    return result


def _pose_env_local(metadata: Mapping[str, Any], name: str, np: Any) -> Any:
    origin = np.asarray(metadata.get("env_origin_w"), dtype=np.float32)
    pose = np.asarray(metadata.get(name), dtype=np.float32).copy()
    if origin.shape != (3,) or pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"metadata {name}/env_origin_w is invalid")
    pose[:3] -= origin
    return pose


def _validate_scene_metadata(metadata: Mapping[str, Any]) -> None:
    scene = TRAJECTORY_ISAAC_SCENE
    expected = {
        "object_urdf": scene.object_asset.relative_path,
        "object_urdf_sha256": scene.object_asset.sha256,
        "table_urdf": scene.table.relative_path,
        "table_urdf_sha256": scene.table.sha256,
        "robot_urdf_sha256": scene.robot.sha256,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"selected trajectory scene metadata {name} changed: "
                f"expected={value!r}, actual={metadata.get(name)!r}"
            )


def _quat_apply_wxyz(quaternion: Any, vector: Any, np: Any) -> Any:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("object pose contains a zero quaternion")
    quaternion = quaternion / norm
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., :1] * twice_cross + np.cross(xyz, twice_cross)


def _even_point_indices(total: int, selected: int, np: Any) -> Any:
    if selected <= 0 or selected > total:
        raise ValueError("selected object point count is invalid")
    if selected == 1:
        return np.asarray([total // 2], dtype=np.int64)
    return np.rint(np.linspace(0, total - 1, selected)).astype(np.int64)


def _load_shard_runtime(shard: Path, *, pq: Any, zarr: Any) -> _ShardRuntime:
    commit = json.loads((shard / "commit.json").read_text(encoding="utf-8"))
    required_commit = {
        "artifact_kind": "objflownet_training_full_shard",
        "schema_version": 10,
        "profile": "training_full",
        "coordinate_frame": "mixed_world_and_robot_base",
        "quaternion_order": "wxyz",
    }
    for name, expected in required_commit.items():
        if commit.get(name) != expected:
            raise ValueError(f"schema-v10 commit {name} changed in {shard}")
    return _ShardRuntime(
        episode_table=pq.read_table(shard / "episode_table.parquet"),
        frames=zarr.open_group(str(shard / "frames.zarr"), mode="r"),
        metadata_by_slot=_metadata_by_slot(shard),
    )


def _materialize_trajectory(
    config: TrajectoryDirectConfig,
    source: InterpriorTrajectorySource,
    *,
    project_root: Path,
    device: torch.device,
    np: Any,
    runtime: _ShardRuntime,
    kinematics: WujiHandKinematics | None,
    kinematics_joint_names: tuple[str, ...] | None,
    episode_validator: Any = None,
) -> tuple[ReferenceTrajectory, WujiHandKinematics]:
    entry = source.entry
    table = runtime.episode_table
    if entry.slot >= table.num_rows:
        raise IndexError("selected trajectory slot exceeds episode table")
    row = table.slice(entry.slot, 1).to_pylist()[0]
    expected_row = {
        "slot": entry.slot,
        "episode_uid": entry.episode_uid,
        "attempt_uid": entry.attempt_uid,
        "episode_status": "success",
        "collection_status": "completed",
        "trajectory_written": True,
        "training_eligible": True,
    }
    metadata = runtime.metadata_by_slot.get(entry.slot)
    if metadata is None:
        raise ValueError(f"derived slot {entry.slot} lacks source metadata")
    if episode_validator is None:
        for name, expected in expected_row.items():
            if row.get(name) != expected:
                raise ValueError(f"selected episode row {name} changed")
        if (
            metadata.get("episode_status") != "success"
            or metadata.get("collection_status") != "completed"
        ):
            raise ValueError("raw source trajectory is not success/completed")
        _validate_scene_metadata(metadata)
    else:
        episode_validator(config, source, row, metadata, runtime.frames)
    names = tuple(metadata.get("robot_joint_names_lab_order", ()))
    if len(names) != 27 or len(set(names)) != 27:
        raise ValueError("raw source lacks a unique 27-joint order")

    root = runtime.frames
    source_frames = int(root["valid_frames"][entry.slot])
    if source_frames < config.data.frame_stop:
        raise ValueError("selected episode is shorter than configured frame_stop")
    start, stop = config.data.frame_start, config.data.frame_stop
    source_stride = config.timing.source_frame_stride
    fields = {
        "time_s": (),
        "physics_step": (),
        "robot_joint_pos": (27,),
        "robot_joint_vel": (27,),
        "robot_joint_pos_target": (27,),
        "ee_current_pose": (7,),
        "object_root_current_pose": (7,),
        "object_root_linear_vel": (3,),
        "object_root_angular_vel": (3,),
    }
    source_arrays: dict[str, Any] = {}
    for name, tail in fields.items():
        if name not in root:
            raise ValueError(f"schema-v10 frames are missing {name}")
        value = np.asarray(root[name][entry.slot, start:stop])
        expected_shape = (stop - start, *tail)
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape changed: {value.shape} != {expected_shape}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        source_arrays[name] = value
    source_physics_delta = (
        config.timing.physics_hz // config.timing.source_control_hz
    )
    if not np.all(
        np.diff(source_arrays["physics_step"].astype(np.int64))
        == source_physics_delta
    ):
        raise ValueError("selected source window is not a gap-free 60 Hz reference")
    if not np.allclose(
        np.diff(source_arrays["time_s"]),
        1.0 / config.timing.source_control_hz,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("selected source-window timestamps are not exactly 60 Hz")
    arrays = {
        name: value[::source_stride]
        for name, value in source_arrays.items()
    }
    if source_stride != 1:
        # Source target[k] drives source state[k] -> state[k + 1].  A
        # policy-rate transition spans ``source_stride`` source transitions,
        # so its single held nominal target must use the final source target in
        # that interval.  The final materialized target is retained for tensor
        # shape consistency and is never used to step beyond the endpoint.
        arrays["robot_joint_pos_target"] = source_arrays[
            "robot_joint_pos_target"
        ][source_stride - 1 :: source_stride]
    for name, tail in fields.items():
        expected_shape = (config.materialized_frame_count, *tail)
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"materialized {name} shape changed: "
                f"{arrays[name].shape} != {expected_shape}"
            )
    ee_quaternion = arrays["ee_current_pose"][:, 3:]
    quaternion_norm_error = np.max(
        np.abs(np.linalg.norm(ee_quaternion, axis=-1) - 1.0)
    )
    if float(quaternion_norm_error) > 2.0e-3:
        raise ValueError("ee_current_pose quaternion is not unit length")
    expected_physics_delta = config.timing.physics_hz // config.timing.control_hz
    if not np.all(
        np.diff(arrays["physics_step"].astype(np.int64)) == expected_physics_delta
    ):
        raise ValueError(
            "materialized reference is not gap-free at the configured control rate"
        )
    if not np.allclose(
        np.diff(arrays["time_s"]),
        1.0 / config.timing.control_hz,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "materialized timestamps disagree with the configured control rate"
        )

    points_object = np.asarray(row["canonical_points_obj"], dtype=np.float32)
    if points_object.shape != (config.data.source_num_points, 3):
        raise ValueError("canonical source point count disagrees with config")
    point_indices = _even_point_indices(
        points_object.shape[0], config.observation.num_object_points, np
    )
    selected_points = points_object[point_indices]
    pose = np.asarray(arrays["object_root_current_pose"], dtype=np.float32)
    surface = (
        _quat_apply_wxyz(pose[:, None, 3:], selected_points[None, :, :], np)
        + pose[:, None, :3]
    ).astype(np.float32)

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(arrays[name], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    joint_position = tensor("robot_joint_pos")
    link7_pose = tensor("ee_current_pose")
    link7_linear_velocity = finite_difference_linear_velocity(
        link7_pose[:, :3], frequency_hz=config.timing.control_hz
    )
    link7_angular_velocity = quaternion_angular_velocity_wxyz(
        link7_pose[:, 3:], frequency_hz=config.timing.control_hz
    )
    if kinematics is None:
        urdf = SCENE_ASSETS / TRAJECTORY_ISAAC_SCENE.robot.relative_path
        if not urdf.is_file():
            raise FileNotFoundError(f"Wuji reference-FK URDF is missing: {urdf}")
        kinematics = WujiHandKinematics.from_urdf(
            urdf,
            canonical_joint_names=names,
            device=device,
        )
    elif kinematics_joint_names != names:
        raise ValueError("selected trajectories use different canonical joint orders")
    hand_keypoints = kinematics.keypoint_positions(link7_pose, joint_position)
    hand_keypoint_velocities = finite_difference_linear_velocity(
        hand_keypoints, frequency_hz=config.timing.control_hz
    )

    trajectory = ReferenceTrajectory(
        trajectory_id=entry.episode_uid,
        joint_names=names,
        control_hz=config.timing.control_hz,
        physics_hz=config.timing.physics_hz,
        joint_position=joint_position,
        joint_velocity=tensor("robot_joint_vel"),
        joint_position_target=tensor("robot_joint_pos_target"),
        link7_position_robot_base=link7_pose[:, :3],
        object_pose_robot_base=tensor("object_root_current_pose"),
        object_linear_velocity_robot_base=tensor("object_root_linear_vel"),
        object_angular_velocity_robot_base=tensor("object_root_angular_vel"),
        object_points_object_frame=torch.as_tensor(
            np.ascontiguousarray(selected_points), dtype=torch.float32, device=device
        ),
        object_surface_points_robot_base=torch.as_tensor(
            np.ascontiguousarray(surface), dtype=torch.float32, device=device
        ),
        robot_root_pose_env_local=torch.as_tensor(
            _pose_env_local(metadata, "robot_root_pose_w", np),
            dtype=torch.float32,
            device=device,
        ),
        table_root_pose_env_local=torch.as_tensor(
            _pose_env_local(metadata, "table_root_pose_w", np),
            dtype=torch.float32,
            device=device,
        ),
        link7_pose_robot_base=link7_pose,
        link7_linear_velocity_robot_base=link7_linear_velocity,
        link7_angular_velocity_robot_base=link7_angular_velocity,
        hand_keypoint_positions_robot_base=hand_keypoints,
        hand_keypoint_linear_velocities_robot_base=hand_keypoint_velocities,
        hand_kinematic_reference_available=True,
    )
    return trajectory, kinematics


def load_interprior_trajectory_collection(
    config: TrajectoryDirectConfig,
    *,
    project_root: str | Path,
    device: torch.device,
    environ: Mapping[str, str] | None = None,
    sources: tuple[InterpriorTrajectorySource, ...] | None = None,
) -> ReferenceCollection:
    """Materialize one trajectory or a homogeneous prefix bank on one device."""

    if config.data.adapter != "interprior":
        from .registry import resolve_data_adapter

        loader = getattr(resolve_data_adapter(config.data.adapter), "load_collection", None)
        if loader is not None:
            return loader(config, project_root=project_root, device=device, environ=environ, sources=sources)

    resolved_sources = (
        resolve_interprior_trajectory_sources(
            config,
            project_root=project_root,
            environ=environ,
        )
        if sources is None
        else tuple(sources)
    )
    if len(resolved_sources) != config.data.trajectory_count:
        raise ValueError("resolved source count disagrees with data.trajectory_count")
    try:
        import numpy as np
        import pyarrow.parquet as pq
        import zarr
    except ImportError as error:  # pragma: no cover - runtime environment concern.
        raise RuntimeError(
            "Interprior trajectory loading requires numpy, pyarrow, zarr"
        ) from error

    shard_cache: dict[Path, _ShardRuntime] = {}
    trajectories: list[ReferenceTrajectory] = []
    kinematics: WujiHandKinematics | None = None
    kinematics_joint_names: tuple[str, ...] | None = None
    root = Path(project_root).resolve()
    episode_validator = None
    if config.data.adapter != "interprior":
        from .registry import resolve_data_adapter

        episode_validator = resolve_data_adapter(config.data.adapter).validate_episode
    for source in resolved_sources:
        runtime = shard_cache.get(source.shard)
        if runtime is None:
            runtime = _load_shard_runtime(source.shard, pq=pq, zarr=zarr)
            shard_cache[source.shard] = runtime
        trajectory, kinematics = _materialize_trajectory(
            config,
            source,
            project_root=root,
            device=device,
            np=np,
            runtime=runtime,
            kinematics=kinematics,
            kinematics_joint_names=kinematics_joint_names,
            episode_validator=episode_validator,
        )
        if kinematics_joint_names is None:
            kinematics_joint_names = trajectory.joint_names
        trajectories.append(trajectory)
    if len(trajectories) == 1:
        return trajectories[0]
    return ReferenceTrajectoryBank(tuple(trajectories))


def load_single_interprior_trajectory(
    config: TrajectoryDirectConfig,
    *,
    project_root: str | Path,
    device: torch.device,
    environ: Mapping[str, str] | None = None,
) -> ReferenceTrajectory:
    """Compatibility wrapper retaining the historical single-trajectory API."""

    trajectory = load_interprior_trajectory_collection(
        config,
        project_root=project_root,
        device=device,
        environ=environ,
    )
    if not isinstance(trajectory, ReferenceTrajectory):
        raise TrajectoryConfigError("single loader requires trajectory_count=1")
    return trajectory


__all__ = [
    "INTERPRIOR_MANIPTRANS_TRAJECTORY_ADAPTER_VERSION",
    "INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_ADAPTER_VERSION",
    "INTERPRIOR_MANIPTRANS_TRAJECTORY_BANK_STRIDED_ADAPTER_VERSION",
    "InterpriorSelectionEntry",
    "InterpriorTrajectorySource",
    "interprior_trajectory_adapter_version",
    "load_interprior_trajectory_collection",
    "load_single_interprior_trajectory",
    "resolve_interprior_trajectory_sources",
    "resolve_single_interprior_source",
    "resolve_trajectory_scene_source_for_data",
    "verify_historical_policy_evaluation_adapter_version",
]
