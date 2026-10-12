"""Small deterministic backend used only for CPU contract/PPO smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data.reference import ReferenceResetState, ReferenceTrajectory
from ..environment.actions import CanonicalActionSpec, make_ordered_action_spec
from ..environment.core import LiveState


def synthetic_action_spec() -> CanonicalActionSpec:
    names = tuple(f"synthetic_joint_{index:02d}" for index in range(27))
    return make_ordered_action_spec(
        joint_names=names,
        position_lower_rad=(-1.0,) * 27,
        position_upper_rad=(1.0,) * 27,
    )


def synthetic_reference_trajectory(
    *,
    frame_count: int = 768,
    device: torch.device = torch.device("cpu"),
) -> ReferenceTrajectory:
    if frame_count < 4:
        raise ValueError("synthetic reference must fit a non-zero start and transition")
    action_spec = synthetic_action_spec()
    zeros27 = torch.zeros(frame_count, 27, dtype=torch.float32, device=device)
    zeros3 = torch.zeros(frame_count, 3, dtype=torch.float32, device=device)
    fingertip_offsets = torch.tensor(
        (
            (0.04, -0.03, 0.01),
            (0.04, -0.015, 0.01),
            (0.04, 0.0, 0.01),
            (0.04, 0.015, 0.01),
            (0.04, 0.03, 0.01),
        ),
        dtype=torch.float32,
        device=device,
    )
    hand_keypoints = fingertip_offsets.repeat_interleave(5, dim=0)
    hand_keypoints = hand_keypoints.unsqueeze(0).expand(frame_count, -1, -1).clone()
    pose = torch.zeros(frame_count, 7, dtype=torch.float32, device=device)
    pose[:, 3] = 1.0
    points = torch.tensor(
        (
            (-0.02, -0.02, -0.02),
            (-0.02, 0.02, 0.02),
            (0.02, -0.02, 0.02),
            (0.02, 0.02, -0.02),
        ),
        dtype=torch.float32,
        device=device,
    )
    tracks = points.unsqueeze(0).expand(frame_count, -1, -1).clone()
    root = torch.tensor(
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        dtype=torch.float32,
        device=device,
    )
    return ReferenceTrajectory(
        trajectory_id="synthetic-single-trajectory",
        joint_names=action_spec.joint_names,
        control_hz=60,
        physics_hz=120,
        joint_position=zeros27,
        joint_velocity=zeros27,
        joint_position_target=zeros27,
        link7_position_robot_base=zeros3,
        object_pose_robot_base=pose,
        object_linear_velocity_robot_base=zeros3,
        object_angular_velocity_robot_base=zeros3,
        object_points_object_frame=points,
        object_surface_points_robot_base=tracks,
        robot_root_pose_env_local=root,
        table_root_pose_env_local=root,
        link7_pose_robot_base=pose,
        link7_linear_velocity_robot_base=zeros3,
        link7_angular_velocity_robot_base=zeros3,
        hand_keypoint_positions_robot_base=hand_keypoints,
        hand_keypoint_linear_velocities_robot_base=torch.zeros_like(hand_keypoints),
        hand_kinematic_reference_available=True,
    )


class LinearRecoveryBackend:
    """A stable differentiable-looking plant; it is not a physics substitute."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        error_decay: float = 0.92,
        action_to_error_gain: float = 1.0,
        hand_keypoint_count: int = 25,
    ) -> None:
        if (
            num_envs <= 0
            or not 0 <= error_decay < 1
            or action_to_error_gain <= 0
            or hand_keypoint_count not in (20, 25)
        ):
            raise ValueError("invalid synthetic backend settings")
        self.num_envs = num_envs
        self.device = device
        self.error_decay = float(error_decay)
        self.action_to_error_gain = float(action_to_error_gain)
        self.hand_keypoint_count = hand_keypoint_count
        points = synthetic_reference_trajectory(
            device=device
        ).object_points_object_frame
        self.object_points = points.unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.joint_position = torch.zeros(num_envs, 27, device=device)
        self.joint_velocity = torch.zeros_like(self.joint_position)
        self.error = torch.zeros(num_envs, 3, device=device)
        self.previous_error = torch.zeros_like(self.error)

    def restore(self, state: ReferenceResetState, *, env_ids: Tensor) -> LiveState:
        if (
            not isinstance(env_ids, Tensor)
            or env_ids.ndim != 1
            or env_ids.numel() == 0
            or env_ids.dtype != torch.int64
            or env_ids.device != self.device
            or (env_ids < 0).any().item()
            or (env_ids >= self.num_envs).any().item()
            or env_ids.unique().numel() != env_ids.numel()
        ):
            raise ValueError("synthetic env_ids must be unique valid int64 [M]")
        if state.joint_position.shape != (env_ids.numel(), 27):
            raise ValueError("synthetic reset state batch and env_ids disagree")
        restored_error = state.object_pose_robot_base[:, :3]
        self.joint_position.index_copy_(0, env_ids, state.joint_position)
        self.joint_velocity.index_copy_(0, env_ids, state.joint_velocity)
        self.object_points.index_copy_(
            0,
            env_ids,
            state.object_points_object_frame,
        )
        self.error.index_copy_(0, env_ids, restored_error)
        self.previous_error.index_copy_(
            0,
            env_ids,
            restored_error - state.object_linear_velocity_robot_base / 60.0,
        )
        return self._live_state()

    @torch.no_grad()
    def step(self, canonical_joint_target: Tensor) -> LiveState:
        if (
            canonical_joint_target.shape != (self.num_envs, 27)
            or canonical_joint_target.dtype != torch.float32
            or canonical_joint_target.device != self.device
            or not torch.isfinite(canonical_joint_target).all().item()
        ):
            raise ValueError("synthetic command must be finite float32 [N,27]")
        previous_joint = self.joint_position.clone()
        self.joint_position.copy_(canonical_joint_target)
        self.joint_velocity.copy_((self.joint_position - previous_joint) * 60.0)
        self.previous_error.copy_(self.error)
        self.error.mul_(self.error_decay).add_(
            self.action_to_error_gain * canonical_joint_target[:, :3]
        )
        return self._live_state()

    def _live_state(self) -> LiveState:
        linear_velocity = (self.error - self.previous_error) * 60.0
        pose = torch.zeros(self.num_envs, 7, dtype=torch.float32, device=self.device)
        pose[:, :3] = self.error
        pose[:, 3] = 1.0
        surface = self.object_points + self.error.unsqueeze(1)
        fingertip_offsets = torch.tensor(
            (
                (0.04, -0.03, 0.01),
                (0.04, -0.015, 0.01),
                (0.04, 0.0, 0.01),
                (0.04, 0.015, 0.01),
                (0.04, 0.03, 0.01),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        fingertips = self.error.unsqueeze(1) + fingertip_offsets.unsqueeze(0)
        points_per_finger = self.hand_keypoint_count // 5
        keypoints = fingertips.repeat_interleave(points_per_finger, dim=1)
        keypoint_velocities = linear_velocity.unsqueeze(1).expand(
            -1, self.hand_keypoint_count, -1
        )
        return LiveState(
            joint_position=self.joint_position.clone(),
            joint_velocity=self.joint_velocity.clone(),
            # Synthetic-only FK proxy: enough to exercise the independent
            # link7 termination contract without importing a simulator/FK stack.
            link7_position_robot_base=0.25 * self.joint_position[:, :3],
            object_pose_robot_base=pose,
            object_linear_velocity_robot_base=linear_velocity,
            object_angular_velocity_robot_base=torch.zeros_like(linear_velocity),
            object_surface_points_robot_base=surface,
            fingertip_positions_robot_base=fingertips,
            link7_orientation_robot_base=pose[:, 3:].clone(),
            link7_linear_velocity_robot_base=(0.25 * self.joint_velocity[:, :3]),
            link7_angular_velocity_robot_base=torch.zeros_like(linear_velocity),
            hand_keypoint_positions_robot_base=keypoints,
            hand_keypoint_linear_velocities_robot_base=keypoint_velocities,
            joint_effort=torch.zeros_like(self.joint_position),
        )


__all__ = [
    "LinearRecoveryBackend",
    "synthetic_action_spec",
    "synthetic_reference_trajectory",
]
