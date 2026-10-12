"""Read-only reference-trajectory and episode-window contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import floor, isfinite
from typing import Literal

import torch
from torch import Tensor


def _float32(value: Tensor, *, name: str, shape_tail: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [F,{','.join(map(str, shape_tail))}]")
    if value.shape[0] <= 1 or value.dtype != torch.float32:
        raise ValueError(f"{name} must contain at least two float32 frames")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().clone()


def _pose(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != (7,):
        raise ValueError(f"{name} must have shape [7]")
    if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite float32")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """One fully materialized trajectory in canonical control order.

    ``joint_position_target[k]`` drives ``state[k] -> state[k + 1]``.  All
    object quantities use the robot-base frame and wxyz quaternions.
    """

    trajectory_id: str
    joint_names: tuple[str, ...]
    control_hz: int
    physics_hz: int
    joint_position: Tensor
    joint_velocity: Tensor
    joint_position_target: Tensor
    link7_position_robot_base: Tensor
    object_pose_robot_base: Tensor
    object_linear_velocity_robot_base: Tensor
    object_angular_velocity_robot_base: Tensor
    object_points_object_frame: Tensor
    object_surface_points_robot_base: Tensor
    robot_root_pose_env_local: Tensor
    table_root_pose_env_local: Tensor
    link7_pose_robot_base: Tensor | None = None
    link7_linear_velocity_robot_base: Tensor | None = None
    link7_angular_velocity_robot_base: Tensor | None = None
    hand_keypoint_positions_robot_base: Tensor | None = None
    hand_keypoint_linear_velocities_robot_base: Tensor | None = None
    hand_kinematic_reference_available: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hand_kinematic_reference_available, bool):
            raise ValueError("hand kinematic reference availability must be boolean")
        if (self.hand_keypoint_positions_robot_base is None) != (
            self.hand_keypoint_linear_velocities_robot_base is None
        ):
            raise ValueError(
                "hand keypoint position and velocity must be provided together"
            )
        if self.hand_kinematic_reference_available and (
            self.hand_keypoint_positions_robot_base is None
            or self.hand_keypoint_linear_velocities_robot_base is None
            or self.link7_pose_robot_base is None
        ):
            raise ValueError("enabled hand kinematic reference is incomplete")
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be non-empty")
        names = tuple(self.joint_names)
        if len(names) != 27 or len(set(names)) != 27:
            raise ValueError("trajectory must declare 27 unique canonical joints")
        if (
            isinstance(self.control_hz, bool)
            or not isinstance(self.control_hz, int)
            or self.control_hz <= 0
            or isinstance(self.physics_hz, bool)
            or not isinstance(self.physics_hz, int)
            or self.physics_hz <= 0
            or self.physics_hz % self.control_hz
        ):
            raise ValueError("trajectory timing must be positive and divisible")

        tensors = {
            "joint_position": _float32(
                self.joint_position, name="joint_position", shape_tail=(27,)
            ),
            "joint_velocity": _float32(
                self.joint_velocity, name="joint_velocity", shape_tail=(27,)
            ),
            "joint_position_target": _float32(
                self.joint_position_target,
                name="joint_position_target",
                shape_tail=(27,),
            ),
            "link7_position_robot_base": _float32(
                self.link7_position_robot_base,
                name="link7_position_robot_base",
                shape_tail=(3,),
            ),
            "object_pose_robot_base": _float32(
                self.object_pose_robot_base,
                name="object_pose_robot_base",
                shape_tail=(7,),
            ),
            "object_linear_velocity_robot_base": _float32(
                self.object_linear_velocity_robot_base,
                name="object_linear_velocity_robot_base",
                shape_tail=(3,),
            ),
            "object_angular_velocity_robot_base": _float32(
                self.object_angular_velocity_robot_base,
                name="object_angular_velocity_robot_base",
                shape_tail=(3,),
            ),
        }
        frame_count = tensors["joint_position"].shape[0]
        identity_pose = torch.zeros(
            frame_count, 7, dtype=torch.float32, device=self.joint_position.device
        )
        identity_pose[:, :3] = tensors["link7_position_robot_base"]
        identity_pose[:, 3] = 1.0
        hand_positions = self.hand_keypoint_positions_robot_base
        hand_velocities = self.hand_keypoint_linear_velocities_robot_base
        hand_keypoint_count = 20
        if hand_positions is not None:
            if hand_positions.ndim != 3 or hand_positions.shape[-1] != 3:
                raise ValueError(
                    "hand_keypoint_positions_robot_base must have shape [F,K,3]"
                )
            hand_keypoint_count = int(hand_positions.shape[1])
            if hand_keypoint_count <= 0:
                raise ValueError("hand keypoint trajectory must not be empty")
        optional_tensors = {
            "link7_pose_robot_base": _float32(
                (
                    identity_pose
                    if self.link7_pose_robot_base is None
                    else self.link7_pose_robot_base
                ),
                name="link7_pose_robot_base",
                shape_tail=(7,),
            ),
            "link7_linear_velocity_robot_base": _float32(
                (
                    torch.zeros(
                        frame_count,
                        3,
                        dtype=torch.float32,
                        device=self.joint_position.device,
                    )
                    if self.link7_linear_velocity_robot_base is None
                    else self.link7_linear_velocity_robot_base
                ),
                name="link7_linear_velocity_robot_base",
                shape_tail=(3,),
            ),
            "link7_angular_velocity_robot_base": _float32(
                (
                    torch.zeros(
                        frame_count,
                        3,
                        dtype=torch.float32,
                        device=self.joint_position.device,
                    )
                    if self.link7_angular_velocity_robot_base is None
                    else self.link7_angular_velocity_robot_base
                ),
                name="link7_angular_velocity_robot_base",
                shape_tail=(3,),
            ),
            "hand_keypoint_positions_robot_base": _float32(
                (
                    torch.zeros(
                        frame_count,
                        hand_keypoint_count,
                        3,
                        dtype=torch.float32,
                        device=self.joint_position.device,
                    )
                    if hand_positions is None
                    else hand_positions
                ),
                name="hand_keypoint_positions_robot_base",
                shape_tail=(hand_keypoint_count, 3),
            ),
            "hand_keypoint_linear_velocities_robot_base": _float32(
                (
                    torch.zeros(
                        frame_count,
                        hand_keypoint_count,
                        3,
                        dtype=torch.float32,
                        device=self.joint_position.device,
                    )
                    if hand_velocities is None
                    else hand_velocities
                ),
                name="hand_keypoint_linear_velocities_robot_base",
                shape_tail=(hand_keypoint_count, 3),
            ),
        }
        tensors.update(optional_tensors)
        if not torch.allclose(
            tensors["link7_pose_robot_base"][:, :3],
            tensors["link7_position_robot_base"],
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("link7 pose and legacy position trajectory disagree")
        points = self.object_surface_points_robot_base
        if (
            not isinstance(points, Tensor)
            or points.ndim != 3
            or points.shape[0] <= 1
            or points.shape[1] <= 0
            or points.shape[2] != 3
            or points.dtype != torch.float32
            or not torch.isfinite(points).all().item()
        ):
            raise ValueError(
                "object_surface_points_robot_base must be finite float32 [F,P,3]"
            )
        tensors["object_surface_points_robot_base"] = points.detach().clone()
        object_points = self.object_points_object_frame
        if (
            not isinstance(object_points, Tensor)
            or object_points.shape != (points.shape[1], 3)
            or object_points.dtype != torch.float32
            or object_points.device != points.device
            or not torch.isfinite(object_points).all().item()
        ):
            raise ValueError("object_points_object_frame must be finite float32 [P,3]")
        object.__setattr__(
            self, "object_points_object_frame", object_points.detach().clone()
        )
        frame_counts = {value.shape[0] for value in tensors.values()}
        if len(frame_counts) != 1:
            raise ValueError("all trajectory arrays must have the same frame count")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("all trajectory arrays must use one device")
        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(
            self,
            "robot_root_pose_env_local",
            _pose(self.robot_root_pose_env_local, name="robot_root_pose_env_local"),
        )
        object.__setattr__(
            self,
            "table_root_pose_env_local",
            _pose(self.table_root_pose_env_local, name="table_root_pose_env_local"),
        )
        if (
            self.robot_root_pose_env_local.device != self.device
            or self.table_root_pose_env_local.device != self.device
        ):
            raise ValueError("root poses and trajectory arrays must use one device")

    @property
    def frame_count(self) -> int:
        return self.joint_position.shape[0]

    @property
    def num_object_points(self) -> int:
        return self.object_surface_points_robot_base.shape[1]

    @property
    def device(self) -> torch.device:
        return self.joint_position.device

    @property
    def hand_keypoint_count(self) -> int:
        return int(self.hand_keypoint_positions_robot_base.shape[1])


class ReferenceTrajectoryBank:
    """A homogeneous, read-only stack of reference trajectories.

    Every environment selects one row independently at reset.  Frame access is
    always a joint ``(trajectory_index, frame_index)`` gather, which prevents a
    multi-trajectory run from accidentally reading frame data from trajectory
    zero while restoring another trajectory's state.
    """

    _FRAME_FIELDS = (
        "joint_position",
        "joint_velocity",
        "joint_position_target",
        "link7_position_robot_base",
        "object_pose_robot_base",
        "object_linear_velocity_robot_base",
        "object_angular_velocity_robot_base",
        "object_surface_points_robot_base",
        "link7_pose_robot_base",
        "link7_linear_velocity_robot_base",
        "link7_angular_velocity_robot_base",
        "hand_keypoint_positions_robot_base",
        "hand_keypoint_linear_velocities_robot_base",
    )

    def __init__(self, trajectories: tuple[ReferenceTrajectory, ...]) -> None:
        values = tuple(trajectories)
        if not values:
            raise ValueError("reference trajectory bank must not be empty")
        first = values[0]
        for index, trajectory in enumerate(values):
            if not isinstance(trajectory, ReferenceTrajectory):
                raise TypeError(f"trajectory bank row {index} is invalid")
            shared = (
                trajectory.joint_names == first.joint_names
                and trajectory.control_hz == first.control_hz
                and trajectory.physics_hz == first.physics_hz
                and trajectory.frame_count == first.frame_count
                and trajectory.num_object_points == first.num_object_points
                and trajectory.hand_keypoint_count == first.hand_keypoint_count
                and trajectory.device == first.device
                and trajectory.hand_kinematic_reference_available
                == first.hand_kinematic_reference_available
            )
            if not shared:
                raise ValueError(
                    "all reference trajectories must share timing, shape, device, "
                    "joint order, and kinematic-reference availability"
                )
        self.trajectory_ids = tuple(value.trajectory_id for value in values)
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("reference trajectory bank IDs must be unique")
        self.joint_names = first.joint_names
        self.control_hz = first.control_hz
        self.physics_hz = first.physics_hz
        self.hand_kinematic_reference_available = (
            first.hand_kinematic_reference_available
        )
        for name in self._FRAME_FIELDS:
            setattr(
                self,
                name,
                torch.stack([getattr(value, name) for value in values]),
            )
        self.object_points_object_frame = torch.stack(
            [value.object_points_object_frame for value in values]
        )
        self.robot_root_pose_env_local = torch.stack(
            [value.robot_root_pose_env_local for value in values]
        )
        self.table_root_pose_env_local = torch.stack(
            [value.table_root_pose_env_local for value in values]
        )
        encoded = "\n".join(self.trajectory_ids).encode("ascii")
        self.trajectory_set_sha256 = sha256(encoded).hexdigest()

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectory_ids)

    @property
    def trajectory_id(self) -> str:
        return f"trajectory_bank:{self.trajectory_set_sha256}"

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[1])

    def terminal_indices(self, trajectory_indices: Tensor) -> Tensor:
        """One valid endpoint per selected row; optional ragged banks override."""

        return torch.full_like(trajectory_indices, self.frame_count - 1)

    @property
    def num_object_points(self) -> int:
        return int(self.object_surface_points_robot_base.shape[2])

    @property
    def device(self) -> torch.device:
        return self.joint_position.device

    @property
    def hand_keypoint_count(self) -> int:
        return int(self.hand_keypoint_positions_robot_base.shape[2])

    def gather_frames(
        self,
        name: str,
        trajectory_indices: Tensor,
        frame_indices: Tensor,
    ) -> Tensor:
        if name not in self._FRAME_FIELDS:
            raise ValueError(f"unknown bank frame field: {name}")
        if (
            trajectory_indices.ndim != 1
            or trajectory_indices.dtype != torch.int64
            or trajectory_indices.device != self.device
            or frame_indices.dtype != torch.int64
            or frame_indices.device != self.device
            or frame_indices.shape[0] != trajectory_indices.shape[0]
        ):
            raise ValueError("trajectory/frame gather indices are incompatible")
        if (
            (trajectory_indices < 0).any().item()
            or (trajectory_indices >= self.trajectory_count).any().item()
            or (frame_indices < 0).any().item()
            or (frame_indices >= self.frame_count).any().item()
        ):
            raise ValueError("trajectory/frame gather index is out of range")
        expanded_trajectory_indices = trajectory_indices.reshape(
            trajectory_indices.shape[0],
            *((1,) * (frame_indices.ndim - 1)),
        ).expand_as(frame_indices)
        return getattr(self, name)[expanded_trajectory_indices, frame_indices]

    def gather_trajectory_values(
        self,
        name: str,
        trajectory_indices: Tensor,
    ) -> Tensor:
        if name not in (
            "object_points_object_frame",
            "robot_root_pose_env_local",
            "table_root_pose_env_local",
        ):
            raise ValueError(f"unknown bank trajectory field: {name}")
        if (
            trajectory_indices.ndim != 1
            or trajectory_indices.dtype != torch.int64
            or trajectory_indices.device != self.device
            or (trajectory_indices < 0).any().item()
            or (trajectory_indices >= self.trajectory_count).any().item()
        ):
            raise ValueError("trajectory gather indices are invalid")
        return getattr(self, name).index_select(0, trajectory_indices)

    def frame_sequence(self, name: str, trajectory_index: int) -> Tensor:
        """Return one trajectory view for browser-only visualization."""

        if name not in self._FRAME_FIELDS:
            raise ValueError(f"unknown bank frame field: {name}")
        if not 0 <= trajectory_index < self.trajectory_count:
            raise ValueError("trajectory index is out of range")
        return getattr(self, name)[trajectory_index]


ReferenceCollection = ReferenceTrajectory | ReferenceTrajectoryBank


def reference_object_lift_phase_boundaries(
    trajectory: ReferenceCollection,
    *,
    height_threshold_m: float,
) -> Tensor:
    """Return each trajectory's first frame lifted above its frame-0 height."""

    if (
        isinstance(height_threshold_m, bool)
        or not isinstance(height_threshold_m, (int, float))
        or not isfinite(height_threshold_m)
        or height_threshold_m <= 0.0
    ):
        raise ValueError("object lift height threshold must be finite and positive")
    object_pose = trajectory.object_pose_robot_base
    object_height = (
        object_pose[:, 2][None, :]
        if isinstance(trajectory, ReferenceTrajectory)
        else object_pose[:, :, 2]
    )
    lifted = object_height - object_height[:, :1] >= height_threshold_m
    has_lift = lifted.any(dim=1)
    if not has_lift.all().item():
        missing = torch.nonzero(~has_lift, as_tuple=False).flatten().tolist()
        raise ValueError(
            "reference trajectories never reach the configured object lift "
            f"threshold; trajectory_indices={missing}"
        )
    return lifted.to(dtype=torch.int64).argmax(dim=1).detach().clone()


@dataclass(frozen=True, slots=True)
class ReferenceResetState:
    """Recorded state restored before an optional perturbation pre-roll."""

    frame_index: Tensor
    joint_position: Tensor
    joint_velocity: Tensor
    previous_joint_target: Tensor
    object_pose_robot_base: Tensor
    object_linear_velocity_robot_base: Tensor
    object_angular_velocity_robot_base: Tensor
    object_points_object_frame: Tensor
    robot_root_pose_env_local: Tensor
    table_root_pose_env_local: Tensor


@dataclass(frozen=True, slots=True)
class ReferenceStepTargets:
    """Current state context plus post-action next-state targets."""

    cursor: Tensor
    joint_position: Tensor
    next_joint_position: Tensor
    next_link7_position_robot_base: Tensor
    next_link7_pose_robot_base: Tensor
    next_link7_linear_velocity_robot_base: Tensor
    next_link7_angular_velocity_robot_base: Tensor
    joint_velocity: Tensor
    next_joint_velocity: Tensor
    hand_keypoint_positions_robot_base: Tensor
    next_hand_keypoint_positions_robot_base: Tensor
    next_hand_keypoint_linear_velocities_robot_base: Tensor
    next_object_pose_robot_base: Tensor
    object_surface_points: Tensor
    next_object_surface_points: Tensor


@dataclass(frozen=True, slots=True)
class ReferenceObservationTargets:
    """Reference values and a sliding, terminal-clamped object-flow view."""

    cursor: Tensor
    joint_position: Tensor
    joint_velocity: Tensor
    link7_pose_robot_base: Tensor | None
    next_joint_position: Tensor
    next_link7_pose_robot_base: Tensor
    object_surface_points: Tensor
    reference_object_flow: Tensor


@dataclass(frozen=True, slots=True)
class ReferenceEpisodeBatch:
    """A reset-pinned episode that advances to the trajectory's final frame."""

    trajectory: ReferenceCollection
    start_indices: Tensor
    ppo_start_cursors: Tensor
    flow_anchor_offsets_steps: tuple[int, ...]
    trajectory_indices: Tensor | None = None

    def __post_init__(self) -> None:
        starts = self.start_indices
        cursors = self.ppo_start_cursors
        if (
            not isinstance(starts, Tensor)
            or starts.ndim != 1
            or starts.numel() == 0
            or starts.dtype != torch.int64
            or starts.device != self.trajectory.device
        ):
            raise ValueError(
                "start_indices must be non-empty int64 [N] on trajectory device"
            )
        if (
            not isinstance(cursors, Tensor)
            or cursors.shape != starts.shape
            or cursors.dtype != torch.int64
            or cursors.device != starts.device
        ):
            raise ValueError("ppo_start_cursors must match start_indices")
        offsets = tuple(self.flow_anchor_offsets_steps)
        if not offsets or offsets[0] < 0:
            raise ValueError("flow anchor offsets must be nonnegative")
        if any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("flow anchor offsets must be strictly increasing")
        object.__setattr__(self, "start_indices", starts.detach().clone())
        object.__setattr__(self, "ppo_start_cursors", cursors.detach().clone())
        object.__setattr__(self, "flow_anchor_offsets_steps", offsets)
        trajectory_count = (
            self.trajectory.trajectory_count
            if isinstance(self.trajectory, ReferenceTrajectoryBank)
            else 1
        )
        trajectory_indices = self.trajectory_indices
        if trajectory_indices is None:
            trajectory_indices = torch.zeros_like(starts)
        if (
            not isinstance(trajectory_indices, Tensor)
            or trajectory_indices.shape != starts.shape
            or trajectory_indices.dtype != torch.int64
            or trajectory_indices.device != starts.device
            or (trajectory_indices < 0).any().item()
            or (trajectory_indices >= trajectory_count).any().item()
        ):
            raise ValueError("trajectory_indices must be valid int64 [N]")
        if trajectory_count == 1 and trajectory_indices.any().item():
            raise ValueError("single-trajectory episodes require trajectory index zero")
        object.__setattr__(
            self,
            "trajectory_indices",
            trajectory_indices.detach().clone(),
        )
        if torch.logical_or(cursors < starts, cursors >= self.terminal_cursors).any().item():
            raise ValueError(
                "PPO cursors must not precede reset starts and must leave one transition"
            )

    @property
    def batch_size(self) -> int:
        return self.start_indices.numel()

    @property
    def episode_lengths(self) -> Tensor:
        return self.terminal_cursors - self.ppo_start_cursors

    @property
    def terminal_cursors(self) -> Tensor:
        if isinstance(self.trajectory, ReferenceTrajectoryBank):
            return self.trajectory.terminal_indices(self.trajectory_indices)
        return torch.full_like(self.start_indices, self.trajectory.frame_count - 1)

    @property
    def shared_episode_length(self) -> int:
        lengths = self.episode_lengths
        if lengths.ne(lengths[0]).any().item():
            raise RuntimeError(
                "synchronous PPO environments must share one start index"
            )
        return int(lengths[0].item())

    def _gather_frames(self, name: str, frame_indices: Tensor) -> Tensor:
        trajectory = self.trajectory
        if isinstance(trajectory, ReferenceTrajectoryBank):
            assert self.trajectory_indices is not None
            return trajectory.gather_frames(
                name,
                self.trajectory_indices,
                frame_indices,
            )
        return getattr(trajectory, name)[frame_indices]

    def _gather_trajectory_values(self, name: str) -> Tensor:
        trajectory = self.trajectory
        assert self.trajectory_indices is not None
        if isinstance(trajectory, ReferenceTrajectoryBank):
            return trajectory.gather_trajectory_values(name, self.trajectory_indices)
        value = getattr(trajectory, name)
        return value.expand(self.batch_size, *([-1] * value.ndim))

    def trajectory_values(self, name: str) -> Tensor:
        """Return one trajectory-level value per environment."""

        return self._gather_trajectory_values(name)

    def recorded_actions(self, frame_indices: Tensor) -> Tensor:
        """Gather recorded actions for the pre-training replay/pre-roll only."""

        if (
            not isinstance(frame_indices, Tensor)
            or frame_indices.dtype != torch.int64
            or frame_indices.device != self.trajectory.device
            or frame_indices.shape[0] != self.batch_size
        ):
            raise ValueError("recorded-action indices must start with episode batch")
        return self._gather_frames("joint_position_target", frame_indices)

    def frame_sequence(self, name: str, environment_index: int) -> Tensor:
        """Return the selected trajectory sequence for one visualization row."""

        if not 0 <= environment_index < self.batch_size:
            raise ValueError("environment index is out of range")
        trajectory = self.trajectory
        if isinstance(trajectory, ReferenceTrajectoryBank):
            assert self.trajectory_indices is not None
            return trajectory.frame_sequence(
                name,
                int(self.trajectory_indices[environment_index]),
            )
        return getattr(trajectory, name)

    def reset_state(self) -> ReferenceResetState:
        index = self.start_indices
        joint_position = self._gather_frames("joint_position", index)
        return ReferenceResetState(
            frame_index=index.detach().clone(),
            joint_position=joint_position,
            joint_velocity=self._gather_frames("joint_velocity", index),
            # No recorded action participates in a regular reset.  The task
            # clamps this state-derived hold target to runtime joint bounds.
            previous_joint_target=joint_position.clone(),
            object_pose_robot_base=self._gather_frames(
                "object_pose_robot_base", index
            ),
            object_linear_velocity_robot_base=(
                self._gather_frames("object_linear_velocity_robot_base", index)
            ),
            object_angular_velocity_robot_base=(
                self._gather_frames("object_angular_velocity_robot_base", index)
            ),
            object_points_object_frame=self._gather_trajectory_values(
                "object_points_object_frame"
            ),
            robot_root_pose_env_local=self._gather_trajectory_values(
                "robot_root_pose_env_local"
            ),
            table_root_pose_env_local=self._gather_trajectory_values(
                "table_root_pose_env_local"
            ),
        )

    def _cursor_tensor(
        self,
        cursor_or_episode_step: Tensor | int,
        *,
        allow_terminal: bool,
    ) -> Tensor:
        """Resolve either legacy shared progress or explicit per-env cursors."""

        if isinstance(cursor_or_episode_step, bool):
            raise ValueError("reference cursor must not be boolean")
        if isinstance(cursor_or_episode_step, int):
            if cursor_or_episode_step < 0:
                raise ValueError("episode_step must be non-negative")
            cursor = self.ppo_start_cursors + cursor_or_episode_step
        elif isinstance(cursor_or_episode_step, Tensor):
            cursor = cursor_or_episode_step
            if (
                cursor.shape != self.start_indices.shape
                or cursor.dtype != torch.int64
                or cursor.device != self.trajectory.device
            ):
                raise ValueError(
                    "per-environment reference cursors must be int64 [N] "
                    "on the trajectory device"
                )
        else:
            raise TypeError("reference cursor must be an int or Tensor")
        maximum = self.terminal_cursors - (0 if allow_terminal else 1)
        if torch.logical_or(cursor < self.start_indices, cursor > maximum).any().item():
            raise ValueError("reference cursor is outside its per-environment episode")
        return cursor

    def step_targets(
        self, cursor_or_episode_step: Tensor | int
    ) -> ReferenceStepTargets:
        cursor = self._cursor_tensor(
            cursor_or_episode_step,
            allow_terminal=False,
        )
        next_cursor = cursor + 1
        return ReferenceStepTargets(
            cursor=cursor.detach().clone(),
            joint_position=self._gather_frames("joint_position", cursor),
            next_joint_position=self._gather_frames("joint_position", next_cursor),
            next_link7_position_robot_base=(
                self._gather_frames("link7_position_robot_base", next_cursor)
            ),
            next_link7_pose_robot_base=(
                self._gather_frames("link7_pose_robot_base", next_cursor)
            ),
            next_link7_linear_velocity_robot_base=(
                self._gather_frames(
                    "link7_linear_velocity_robot_base", next_cursor
                )
            ),
            next_link7_angular_velocity_robot_base=(
                self._gather_frames(
                    "link7_angular_velocity_robot_base", next_cursor
                )
            ),
            joint_velocity=self._gather_frames("joint_velocity", cursor),
            next_joint_velocity=self._gather_frames("joint_velocity", next_cursor),
            hand_keypoint_positions_robot_base=(
                self._gather_frames(
                    "hand_keypoint_positions_robot_base", cursor
                )
            ),
            next_hand_keypoint_positions_robot_base=(
                self._gather_frames(
                    "hand_keypoint_positions_robot_base", next_cursor
                )
            ),
            next_hand_keypoint_linear_velocities_robot_base=(
                self._gather_frames(
                    "hand_keypoint_linear_velocities_robot_base", next_cursor
                )
            ),
            next_object_pose_robot_base=(
                self._gather_frames("object_pose_robot_base", next_cursor)
            ),
            object_surface_points=(
                self._gather_frames("object_surface_points_robot_base", cursor)
            ),
            next_object_surface_points=(
                self._gather_frames(
                    "object_surface_points_robot_base", next_cursor
                )
            ),
        )

    def joint_position_targets(self, cursor: Tensor, offsets: tuple[int, ...]) -> Tensor:
        """Read same-episode lookahead without advancing state or reading padding."""
        cursor = self._cursor_tensor(cursor, allow_terminal=True)
        if not offsets or any(type(step) is not int or step < 0 for step in offsets):
            raise ValueError("joint target offsets must be nonnegative integers")
        steps = torch.tensor(offsets, dtype=torch.int64, device=cursor.device)
        frames = torch.minimum(cursor[:, None] + steps, self.terminal_cursors[:, None])
        return self._gather_frames("joint_position", frames)

    def observation_targets(
        self,
        cursor_or_episode_step: Tensor | int,
        *,
        include_current_link7_pose: bool = False,
    ) -> ReferenceObservationTargets:
        cursor = self._cursor_tensor(
            cursor_or_episode_step,
            allow_terminal=True,
        )
        trajectory = self.trajectory
        offsets = torch.tensor(
            self.flow_anchor_offsets_steps,
            dtype=torch.int64,
            device=trajectory.device,
        )
        anchor_indices = torch.minimum(
            cursor[:, None] + offsets[None, :],
            self.terminal_cursors[:, None],
        )
        next_cursor = torch.minimum(
            cursor + 1,
            self.terminal_cursors,
        )
        return ReferenceObservationTargets(
            cursor=cursor.detach().clone(),
            joint_position=self._gather_frames("joint_position", cursor),
            joint_velocity=self._gather_frames("joint_velocity", cursor),
            link7_pose_robot_base=(
                self._gather_frames("link7_pose_robot_base", cursor)
                if include_current_link7_pose
                else None
            ),
            next_joint_position=self._gather_frames(
                "joint_position", next_cursor
            ),
            next_link7_pose_robot_base=self._gather_frames(
                "link7_pose_robot_base", next_cursor
            ),
            object_surface_points=(
                self._gather_frames("object_surface_points_robot_base", cursor)
            ),
            reference_object_flow=self._gather_frames(
                "object_surface_points_robot_base", anchor_indices
            ),
        )


class ReferenceStartSampler:
    """Reproducible fixed, uniform, or trajectory-phase start sampler."""

    def __init__(
        self,
        *,
        frame_count: int,
        pre_roll_policy_steps: int,
        maximum_trajectory_fraction: float,
        mode: Literal["fixed", "uniform_valid", "object_lift_phase_weighted"],
        fixed_index: int | None,
        minimum: int | None,
        maximum: int | None,
        generator: torch.Generator,
        device: torch.device,
        phase_boundary_indices: Tensor | None = None,
        pre_lift_reset_probability: float = 0.6,
    ) -> None:
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 3
            or isinstance(pre_roll_policy_steps, bool)
            or not isinstance(pre_roll_policy_steps, int)
            or pre_roll_policy_steps < 0
        ):
            raise ValueError(
                "frame count must be positive and pre-roll must be non-negative"
            )
        if (
            isinstance(maximum_trajectory_fraction, bool)
            or not isfinite(maximum_trajectory_fraction)
            or not 0.0 < maximum_trajectory_fraction <= 1.0
        ):
            raise ValueError("maximum trajectory fraction must be in (0,1]")
        legal_min = 0 if minimum == 0 or fixed_index == 0 else 1
        fraction_max = floor(frame_count * maximum_trajectory_fraction)
        legal_max = min(frame_count - 2 - pre_roll_policy_steps, fraction_max)
        if legal_max < legal_min:
            raise ValueError("trajectory is too short for pre-roll plus one PPO step")
        selected_min = legal_min if minimum is None else max(legal_min, minimum)
        selected_max = legal_max if maximum is None else min(legal_max, maximum)
        if selected_min > selected_max:
            raise ValueError("configured start-index range has no legal frame")
        if mode not in (
            "fixed",
            "uniform_valid",
            "object_lift_phase_weighted",
        ):
            raise ValueError("unsupported start-index mode")
        if mode == "fixed":
            if fixed_index is None or not selected_min <= fixed_index <= selected_max:
                raise ValueError("fixed start index is outside the legal range")
        elif fixed_index is not None:
            raise ValueError(f"{mode} start sampling must not set fixed_index")
        if (
            isinstance(pre_lift_reset_probability, bool)
            or not isinstance(pre_lift_reset_probability, (int, float))
            or not isfinite(pre_lift_reset_probability)
            or not 0.0 < pre_lift_reset_probability < 1.0
        ):
            raise ValueError("pre-lift reset probability must be in (0,1)")
        if mode == "object_lift_phase_weighted":
            if (
                not isinstance(phase_boundary_indices, Tensor)
                or phase_boundary_indices.ndim != 1
                or phase_boundary_indices.numel() == 0
                or phase_boundary_indices.dtype != torch.int64
                or phase_boundary_indices.device != device
            ):
                raise ValueError(
                    "phase-weighted sampling requires int64 phase boundaries"
                )
            if torch.logical_or(
                phase_boundary_indices <= selected_min,
                phase_boundary_indices > selected_max,
            ).any().item():
                raise ValueError(
                    "every phase boundary must leave legal pre- and post-lift starts"
                )
            boundaries = phase_boundary_indices.detach().clone()
        elif phase_boundary_indices is not None:
            raise ValueError("phase boundaries require phase-weighted start sampling")
        else:
            boundaries = None
        self.minimum = selected_min
        self.maximum = selected_max
        # Training deliberately samples only the configured range, but a
        # full-suffix checkpoint evaluation may need to start at the first
        # materialized reference frame.  Keep that broader physical range
        # explicit instead of weakening the ordinary reset contract.
        self.full_reference_minimum = 0
        self.full_reference_maximum = frame_count - 2 - pre_roll_policy_steps
        self.mode = mode
        self.fixed_index = fixed_index
        self.maximum_trajectory_fraction = float(maximum_trajectory_fraction)
        self.generator = generator
        self.device = device
        self.phase_boundary_indices = boundaries
        self.pre_lift_reset_probability = float(pre_lift_reset_probability)

    def sample(
        self,
        count: int,
        *,
        trajectory_indices: Tensor | None = None,
    ) -> Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("sample count must be positive")
        if self.mode == "fixed":
            return torch.full(
                (count,),
                int(self.fixed_index),
                dtype=torch.int64,
                device=self.device,
            )
        if self.mode == "object_lift_phase_weighted":
            boundaries = self.phase_boundary_indices
            assert boundaries is not None
            if (
                not isinstance(trajectory_indices, Tensor)
                or trajectory_indices.shape != (count,)
                or trajectory_indices.dtype != torch.int64
                or trajectory_indices.device != self.device
                or (trajectory_indices < 0).any().item()
                or (trajectory_indices >= boundaries.numel()).any().item()
            ):
                raise ValueError(
                    "phase-weighted sampling requires valid trajectory indices"
                )
            selected_boundaries = boundaries.index_select(0, trajectory_indices)
            choose_pre_lift = torch.rand(
                count,
                dtype=torch.float32,
                device=self.device,
                generator=self.generator,
            ) < self.pre_lift_reset_probability
            low = torch.where(
                choose_pre_lift,
                torch.full_like(selected_boundaries, self.minimum),
                selected_boundaries,
            )
            high = torch.where(
                choose_pre_lift,
                selected_boundaries - 1,
                torch.full_like(selected_boundaries, self.maximum),
            )
            unit = torch.rand(
                count,
                dtype=torch.float32,
                device=self.device,
                generator=self.generator,
            )
            return low + torch.floor(
                unit * (high - low + 1).to(dtype=torch.float32)
            ).to(dtype=torch.int64)
        return torch.randint(
            self.minimum,
            self.maximum + 1,
            (count,),
            dtype=torch.int64,
            device=self.device,
            generator=self.generator,
        )

    def validate_indices(
        self,
        start_indices: Tensor,
        *,
        count: int,
        range_mode: Literal["configured", "full_reference"] = "configured",
        trajectory_indices: Tensor | None = None,
    ) -> Tensor:
        """Validate an explicit reset batch without advancing sampler RNG.

        ``full_reference`` is reserved for independent checkpoint evaluation.
        It permits frame zero and ignores the training-only maximum-fraction
        cap, while still requiring a complete next-state transition and any
        configured pre-roll.
        """

        if (
            not isinstance(start_indices, Tensor)
            or start_indices.ndim != 1
            or start_indices.shape[0] != count
            or start_indices.dtype != torch.int64
            or start_indices.device != self.device
        ):
            raise ValueError(
                "explicit start indices must be int64 [count] on the sampler device"
            )
        if range_mode not in ("configured", "full_reference"):
            raise ValueError("unsupported explicit start-index range mode")
        minimum = (
            self.minimum
            if range_mode == "configured"
            else self.full_reference_minimum
        )
        maximum = (
            self.maximum
            if range_mode == "configured"
            else self.full_reference_maximum
        )
        if (
            torch.logical_or(
                start_indices < minimum,
                start_indices > maximum,
            )
            .any()
            .item()
        ):
            raise ValueError(
                f"explicit start indices are outside the {range_mode} legal "
                f"range [{minimum}, {maximum}]"
            )
        return start_indices.detach().clone()

    def evenly_spaced(self, count: int) -> Tensor:
        """Return fixed, unique indices spanning the configured legal range."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("evenly-spaced start count must be positive")
        available = self.maximum - self.minimum + 1
        if count > available:
            raise ValueError(
                "evenly-spaced start count exceeds the number of legal indices: "
                f"count={count} available={available}"
            )
        if count == 1:
            return torch.tensor(
                [self.minimum],
                dtype=torch.int64,
                device=self.device,
            )
        ordinal = torch.arange(count, dtype=torch.int64, device=self.device)
        offsets = torch.div(
            ordinal * (available - 1),
            count - 1,
            rounding_mode="floor",
        )
        return offsets + self.minimum


def build_reference_episode(
    trajectory: ReferenceCollection,
    *,
    start_indices: Tensor,
    pre_roll_policy_steps: int,
    flow_anchor_offsets_steps: tuple[int, ...],
    trajectory_indices: Tensor | None = None,
) -> ReferenceEpisodeBatch:
    """Build an episode whose reference cursor runs to the final trajectory frame."""

    if (
        isinstance(pre_roll_policy_steps, bool)
        or not isinstance(pre_roll_policy_steps, int)
        or pre_roll_policy_steps < 0
    ):
        raise ValueError("pre-roll must be a non-negative integer")
    if (
        not isinstance(start_indices, Tensor)
        or start_indices.ndim != 1
        or start_indices.numel() == 0
        or start_indices.dtype != torch.int64
        or start_indices.device != trajectory.device
    ):
        raise ValueError("start_indices must be int64 [N] on trajectory device")
    legal = torch.logical_and(
        start_indices >= 0,
        start_indices + pre_roll_policy_steps <= trajectory.frame_count - 2,
    )
    if not legal.all().item():
        raise ValueError(
            "start_indices violate the non-empty episode contract"
        )
    ppo_cursor = start_indices + pre_roll_policy_steps
    return ReferenceEpisodeBatch(
        trajectory=trajectory,
        start_indices=start_indices,
        ppo_start_cursors=ppo_cursor,
        flow_anchor_offsets_steps=flow_anchor_offsets_steps,
        trajectory_indices=trajectory_indices,
    )


__all__ = [
    "ReferenceObservationTargets",
    "ReferenceResetState",
    "ReferenceStartSampler",
    "ReferenceStepTargets",
    "ReferenceTrajectory",
    "ReferenceTrajectoryBank",
    "ReferenceCollection",
    "ReferenceEpisodeBatch",
    "build_reference_episode",
    "reference_object_lift_phase_boundaries",
]
