"""Small fixed-scale observation assembly for trajectory residual PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch
from torch import Tensor

from .actions import CanonicalActionSpec
from .reference_robot_observation import (
    CurrentReferenceRobotTargetFeature,
    assemble_current_reference_robot_target_features,
    current_reference_robot_target_feature_dim,
    current_reference_robot_target_feature_layout,
    current_reference_robot_target_manifest,
    current_reference_robot_target_requirements,
    normalize_current_reference_robot_target_features,
)


ObjectTargetRepresentation = Literal[
    "legacy_reference_flow",
    "direct_target_vectors",
    "absolute_live_reference",
    "error_reference_delta",
]
ReferenceRobotTargetRepresentation = Literal[
    "none",
    "next_link7_pose_and_hand_joint_position",
]


@dataclass(frozen=True, slots=True)
class TrajectoryObservationSpec:
    num_object_points: int
    flow_anchor_offsets_steps: tuple[int, ...]
    joint_velocity_scale_rad_s: float
    object_point_scale_m: float
    object_linear_velocity_scale_m_s: float
    object_angular_velocity_scale_rad_s: float
    fingertip_count: int = 5
    include_reference_robot_state: bool = True
    include_previous_policy_action: bool = False
    include_current_link7_pose: bool = False
    link7_position_scale_m: float = 1.0
    object_target_representation: ObjectTargetRepresentation = (
        "legacy_reference_flow"
    )
    reference_robot_target_representation: ReferenceRobotTargetRepresentation = (
        "none"
    )
    current_reference_robot_target_features: tuple[
        CurrentReferenceRobotTargetFeature, ...
    ] = ()
    robot_joint_temporal_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.robot_joint_temporal_context, bool):
            raise TypeError("robot_joint_temporal_context must be bool")
        if self.num_object_points <= 0 or self.fingertip_count <= 0:
            raise ValueError("observation point counts must be positive")
        for name in (
            "include_reference_robot_state",
            "include_previous_policy_action",
            "include_current_link7_pose",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        representations = (
            "legacy_reference_flow",
            "direct_target_vectors",
            "absolute_live_reference",
            "error_reference_delta",
        )
        if self.object_target_representation not in representations:
            raise ValueError(
                "object_target_representation must be one of "
                f"{representations}"
            )
        robot_target_representations = (
            "none",
            "next_link7_pose_and_hand_joint_position",
        )
        if self.reference_robot_target_representation not in (
            robot_target_representations
        ):
            raise ValueError(
                "reference_robot_target_representation must be one of "
                f"{robot_target_representations}"
            )
        if (
            self.include_reference_robot_state
            and self.reference_robot_target_representation != "none"
        ):
            raise ValueError(
                "legacy reference robot errors and explicit robot targets "
                "cannot both be enabled"
            )
        current_robot_target_features = (
            normalize_current_reference_robot_target_features(
                self.current_reference_robot_target_features
            )
        )
        if self.include_reference_robot_state and current_robot_target_features:
            raise ValueError(
                "legacy reference robot errors and current robot targets cannot "
                "both be enabled"
            )
        if (
            self.reference_robot_target_representation != "none"
            and current_robot_target_features
        ):
            raise ValueError(
                "next-reference robot targets and current-reference robot targets "
                "cannot both be enabled"
            )
        offsets = tuple(self.flow_anchor_offsets_steps)
        required_first_offset = (
            0 if self.object_target_representation == "legacy_reference_flow" else 1
        )
        if (
            not offsets
            or offsets[0] != required_first_offset
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                "flow anchor offsets must start at "
                f"{required_first_offset} and strictly increase for "
                f"{self.object_target_representation}"
            )
        if (
            self.object_target_representation == "error_reference_delta"
            and len(offsets) < 2
        ):
            raise ValueError(
                "error_reference_delta requires one next-step and at least one "
                "long-horizon anchor"
            )
        scales = torch.tensor(
            (
                self.joint_velocity_scale_rad_s,
                self.object_point_scale_m,
                self.object_linear_velocity_scale_m_s,
                self.object_angular_velocity_scale_rad_s,
                self.link7_position_scale_m,
            ),
            dtype=torch.float64,
        )
        if not torch.isfinite(scales).all().item() or (scales <= 0).any().item():
            raise ValueError("observation scales must be finite and positive")
        object.__setattr__(self, "flow_anchor_offsets_steps", offsets)
        object.__setattr__(
            self,
            "current_reference_robot_target_features",
            current_robot_target_features,
        )

    @property
    def current_reference_robot_target_feature_dim(self) -> int:
        return current_reference_robot_target_feature_dim(
            self.current_reference_robot_target_features
        )

    @property
    def feature_dim(self) -> int:
        points = 3 * self.num_object_points
        flow = len(self.flow_anchor_offsets_steps) * points
        fingertips = 3 * self.fingertip_count
        robot = 27 * (4 if self.include_reference_robot_state else 2)
        if self.include_previous_policy_action:
            robot += 27
        if self.include_current_link7_pose:
            robot += 9
        if self.reference_robot_target_representation != "none":
            robot += 9 + 20
        robot += self.current_reference_robot_target_feature_dim
        if self.robot_joint_temporal_context:
            from .robot_temporal_observation import FEATURE_DIM
            robot += FEATURE_DIM
        # q, qdot, optional q/qdot reference errors, point error, sliding
        # flow, optional previous policy action/link7 pose, object velocity,
        # fingertips, and progress.
        return robot + 6 + points + flow + fingertips + 1

    def feature_layout(self) -> tuple[tuple[str, int], ...]:
        robot: tuple[tuple[str, int], ...] = (
            ("joint_position_normalized", 27),
            ("joint_velocity_scaled", 27),
        )
        if self.include_reference_robot_state:
            robot += (
                ("reference_joint_position_error", 27),
                ("reference_joint_velocity_error", 27),
            )
        if self.include_previous_policy_action:
            robot += (("previous_policy_action", 27),)
        if self.include_current_link7_pose:
            robot += (("current_link7_pose_position_rot6d", 9),)
        if self.reference_robot_target_representation == (
            "next_link7_pose_and_hand_joint_position"
        ):
            robot += (
                ("next_reference_link7_pose_position_rot6d", 9),
                ("next_reference_hand_joint_position_normalized", 20),
            )
        robot += current_reference_robot_target_feature_layout(
            self.current_reference_robot_target_features
        )
        layout = robot + self._object_target_layout() + (
            ("object_linear_velocity_scaled", 3),
            ("object_angular_velocity_scaled", 3),
            ("fingertips_relative_to_object", 3 * self.fingertip_count),
            ("episode_progress", 1),
        )
        if self.robot_joint_temporal_context:
            from .robot_temporal_observation import feature_layout
            layout += feature_layout()
        return layout

    def _object_target_layout(self) -> tuple[tuple[str, int], ...]:
        points = 3 * self.num_object_points
        flow = len(self.flow_anchor_offsets_steps) * points
        if self.object_target_representation == "legacy_reference_flow":
            return (
                ("current_object_point_error", points),
                ("reference_object_flow", flow),
            )
        if self.object_target_representation == "direct_target_vectors":
            return (
                ("current_object_target_vector", points),
                ("future_object_target_vectors", flow),
            )
        if self.object_target_representation == "absolute_live_reference":
            return (
                ("current_live_object_points", points),
                ("future_reference_object_points", flow),
            )
        if self.object_target_representation == "error_reference_delta":
            return (
                ("current_object_target_vector", points),
                ("next_reference_object_point_delta", points),
                ("future_object_target_vectors", flow - points),
            )
        raise RuntimeError("validated object target representation is unsupported")

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "feature_dim": self.feature_dim,
            "layout": [
                {"name": name, "feature_dim": width}
                for name, width in self.feature_layout()
            ],
            "num_object_points": self.num_object_points,
            "flow_anchor_offsets_steps": list(self.flow_anchor_offsets_steps),
            "flow_definition": self._object_target_definition(),
            "frame": "robot_base",
            "scales": {
                "joint_velocity_rad_s": self.joint_velocity_scale_rad_s,
                "object_point_m": self.object_point_scale_m,
                "object_linear_velocity_m_s": self.object_linear_velocity_scale_m_s,
                "object_angular_velocity_rad_s": (
                    self.object_angular_velocity_scale_rad_s
                ),
            },
        }
        if self.object_target_representation != "legacy_reference_flow":
            payload[
                "object_target_representation"
            ] = self.object_target_representation
        if self.include_previous_policy_action:
            payload["previous_policy_action"] = {
                "semantic": "previous_valid_ppo_policy_output",
                "coordinates": "normalized_policy_action",
                "bounds": [-1.0, 1.0],
                "reset_value": 0.0,
                "reset_pre_roll_value": 0.0,
            }
        if self.include_current_link7_pose:
            payload["current_link7_pose"] = {
                "frame": "robot_base",
                "representation": "xyz_plus_wxyz_first_two_rotation_columns",
                "position_scale_m": self.link7_position_scale_m,
            }
        if self.reference_robot_target_representation != "none":
            payload["reference_robot_target"] = {
                "representation": self.reference_robot_target_representation,
                "timing": "post_action_t_plus_1_reward_target",
                "frame": "robot_base",
                "link7_pose": "xyz_plus_wxyz_first_two_rotation_columns",
                "link7_position_scale_m": self.link7_position_scale_m,
                "hand_joint_position": (
                    "20 canonical hand targets normalized by runtime joint bounds"
                ),
            }
        if self.current_reference_robot_target_features:
            payload["current_reference_robot_target"] = (
                current_reference_robot_target_manifest(
                    self.current_reference_robot_target_features,
                    link7_position_scale_m=self.link7_position_scale_m,
                )
            )
        if self.robot_joint_temporal_context:
            from .robot_temporal_observation import manifest as temporal_manifest
            payload["robot_joint_temporal_context"] = temporal_manifest()
        return payload

    def _object_target_definition(self) -> str:
        definitions = {
            "legacy_reference_flow": (
                "sliding_reference_anchor_points-current_reference_points; "
                "terminal indices are clamped"
            ),
            "direct_target_vectors": (
                "current_target=reference_current-live_current; "
                "future_targets=reference_anchor-live_current; terminal indices "
                "are clamped"
            ),
            "absolute_live_reference": (
                "current_live_points and absolute reference_anchor_points in "
                "robot_base; terminal indices are clamped"
            ),
            "error_reference_delta": (
                "current_target=reference_current-live_current; next_reference_delta="
                "reference_t_plus_1-reference_current; later_targets="
                "reference_anchor-live_current; terminal indices are clamped"
            ),
        }
        return definitions[self.object_target_representation]


@dataclass(frozen=True, slots=True)
class TrajectoryObservationInput:
    joint_position: Tensor
    joint_velocity: Tensor
    reference_joint_position: Tensor | None
    reference_joint_velocity: Tensor | None
    live_object_points_robot_base: Tensor
    reference_object_points_robot_base: Tensor
    reference_object_flow: Tensor
    object_pose_robot_base: Tensor
    object_linear_velocity_robot_base: Tensor
    object_angular_velocity_robot_base: Tensor
    fingertip_positions_robot_base: Tensor
    episode_progress: Tensor
    previous_policy_action: Tensor | None = None
    link7_pose_robot_base: Tensor | None = None
    reference_link7_pose_robot_base: Tensor | None = None
    temporal_joint_positions: Tensor | None = None


def _wxyz_to_rotation_6d(quaternion: Tensor) -> Tensor:
    """Convert normalized wxyz quaternions to the first two rotation columns."""

    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if (norm <= torch.finfo(quaternion.dtype).eps).any().item():
        raise ValueError("link7_pose_robot_base contains a zero quaternion")
    w, x, y, z = (quaternion / norm).unbind(dim=-1)
    two = quaternion.new_tensor(2.0)
    rotation = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


class TrajectoryObservationAssembler:
    """Assemble one normalized flat tensor without histories or learned encoders."""

    def __init__(
        self,
        action_spec: CanonicalActionSpec,
        observation_spec: TrajectoryObservationSpec,
    ) -> None:
        self.action_spec = action_spec
        self.spec = observation_spec

    def _validate(self, values: TrajectoryObservationInput) -> int:
        self.action_spec.validate(
            values.joint_position, name="joint_position", check_bounds=False
        )
        count = values.joint_position.shape[0]
        device = values.joint_position.device
        points = self.spec.num_object_points
        anchors = len(self.spec.flow_anchor_offsets_steps)
        fingertips = self.spec.fingertip_count
        expected: dict[str, tuple[int, ...]] = {
            "joint_velocity": (count, 27),
            "live_object_points_robot_base": (count, points, 3),
            "reference_object_points_robot_base": (count, points, 3),
            "reference_object_flow": (count, anchors, points, 3),
            "object_pose_robot_base": (count, 7),
            "object_linear_velocity_robot_base": (count, 3),
            "object_angular_velocity_robot_base": (count, 3),
            "fingertip_positions_robot_base": (count, fingertips, 3),
            "episode_progress": (count, 1),
        }
        if self.spec.include_reference_robot_state:
            expected.update(
                {
                    "reference_joint_position": (count, 27),
                    "reference_joint_velocity": (count, 27),
                }
            )
        if self.spec.reference_robot_target_representation != "none":
            expected.update(
                {
                    "reference_joint_position": (count, 27),
                    "reference_link7_pose_robot_base": (count, 7),
                }
            )
        needs_current_reference_joint_position, needs_current_reference_link7_pose = (
            current_reference_robot_target_requirements(
                self.spec.current_reference_robot_target_features
            )
        )
        if needs_current_reference_joint_position:
            expected["reference_joint_position"] = (count, 27)
        if self.spec.robot_joint_temporal_context:
            expected["temporal_joint_positions"] = (count, 8, 27)
        if needs_current_reference_link7_pose:
            expected["reference_link7_pose_robot_base"] = (count, 7)
        if self.spec.include_previous_policy_action:
            expected["previous_policy_action"] = (count, 27)
        if self.spec.include_current_link7_pose:
            expected["link7_pose_robot_base"] = (count, 7)
        for name, shape in expected.items():
            value = getattr(values, name)
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != shape
                or value.dtype != torch.float32
                or value.device != device
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"{name} must be finite float32 {shape} on one device")
        if (
            torch.logical_or(values.episode_progress < 0, values.episode_progress > 1)
            .any()
            .item()
        ):
            raise ValueError("episode_progress must be in [0,1]")
        if self.spec.include_previous_policy_action:
            assert values.previous_policy_action is not None
            tolerance = 32.0 * torch.finfo(torch.float32).eps
            if (
                torch.logical_or(
                    values.previous_policy_action < -1.0 - tolerance,
                    values.previous_policy_action > 1.0 + tolerance,
                )
                .any()
                .item()
            ):
                raise ValueError("previous_policy_action must stay in [-1,1]")
        return count

    def assemble(self, values: TrajectoryObservationInput) -> Tensor:
        count = self._validate(values)
        lower, upper = self.action_spec.bounds(values.joint_position)
        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        joint_position = (values.joint_position - center) / half_range
        velocity_scale = self.spec.joint_velocity_scale_rad_s
        joint_velocity = values.joint_velocity / velocity_scale
        object_target_features = self._object_target_features(values)
        object_position = values.object_pose_robot_base[:, None, :3]
        fingertip_relative = (
            values.fingertip_positions_robot_base - object_position
        ) / self.spec.object_point_scale_m
        robot_features: tuple[Tensor, ...] = (joint_position, joint_velocity)
        if self.spec.include_reference_robot_state:
            if (
                values.reference_joint_position is None
                or values.reference_joint_velocity is None
            ):
                raise RuntimeError("validated reference robot state is missing")
            robot_features += (
                (values.joint_position - values.reference_joint_position) / half_range,
                (values.joint_velocity - values.reference_joint_velocity)
                / velocity_scale,
            )
        if self.spec.include_previous_policy_action:
            if values.previous_policy_action is None:
                raise RuntimeError("validated previous policy action is missing")
            robot_features += (values.previous_policy_action,)
        if self.spec.include_current_link7_pose:
            if values.link7_pose_robot_base is None:
                raise RuntimeError("validated link7 pose is missing")
            link7_pose = values.link7_pose_robot_base
            link7_pose_features = torch.cat(
                (
                    link7_pose[:, :3] / self.spec.link7_position_scale_m,
                    _wxyz_to_rotation_6d(link7_pose[:, 3:]),
                ),
                dim=-1,
            )
            robot_features += (link7_pose_features,)
        if self.spec.reference_robot_target_representation == (
            "next_link7_pose_and_hand_joint_position"
        ):
            if (
                values.reference_joint_position is None
                or values.reference_link7_pose_robot_base is None
            ):
                raise RuntimeError("validated reference robot target is missing")
            reference_link7_pose = values.reference_link7_pose_robot_base
            reference_link7_features = torch.cat(
                (
                    reference_link7_pose[:, :3]
                    / self.spec.link7_position_scale_m,
                    _wxyz_to_rotation_6d(reference_link7_pose[:, 3:]),
                ),
                dim=-1,
            )
            reference_joint_position = (
                values.reference_joint_position - center
            ) / half_range
            hand_indices = self.action_spec.groups[1].indices
            robot_features += (
                reference_link7_features,
                reference_joint_position[:, hand_indices],
            )
        robot_features += assemble_current_reference_robot_target_features(
            self.spec.current_reference_robot_target_features,
            action_spec=self.action_spec,
            reference_joint_position=values.reference_joint_position,
            reference_link7_pose_robot_base=(
                values.reference_link7_pose_robot_base
            ),
            joint_lower=lower,
            joint_upper=upper,
            link7_position_scale_m=self.spec.link7_position_scale_m,
        )
        features = torch.cat(
            robot_features
            + object_target_features
            + (
                values.object_linear_velocity_robot_base
                / self.spec.object_linear_velocity_scale_m_s,
                values.object_angular_velocity_robot_base
                / self.spec.object_angular_velocity_scale_rad_s,
                fingertip_relative.reshape(count, -1),
                2.0 * values.episode_progress - 1.0,
            ),
            dim=-1,
        )
        if self.spec.robot_joint_temporal_context:
            from .robot_temporal_observation import normalize
            assert values.temporal_joint_positions is not None
            features = torch.cat((features, normalize(
                values.temporal_joint_positions, lower, upper)), dim=-1)
        if features.shape != (count, self.spec.feature_dim):
            raise RuntimeError(
                "observation feature layout and assembled width disagree"
            )
        if not torch.isfinite(features).all().item():
            raise ValueError("assembled observation contains non-finite values")
        return features

    def _object_target_features(
        self, values: TrajectoryObservationInput
    ) -> tuple[Tensor, ...]:
        count = values.joint_position.shape[0]
        live = values.live_object_points_robot_base
        current_reference = values.reference_object_points_robot_base
        future_reference = values.reference_object_flow
        scale = self.spec.object_point_scale_m
        representation = self.spec.object_target_representation
        if representation == "legacy_reference_flow":
            return (
                ((live - current_reference) / scale).reshape(count, -1),
                (
                    (future_reference - future_reference[:, :1]) / scale
                ).reshape(count, -1),
            )
        if representation == "direct_target_vectors":
            return (
                ((current_reference - live) / scale).reshape(count, -1),
                ((future_reference - live[:, None]) / scale).reshape(count, -1),
            )
        if representation == "absolute_live_reference":
            return (
                (live / scale).reshape(count, -1),
                (future_reference / scale).reshape(count, -1),
            )
        if representation == "error_reference_delta":
            return (
                ((current_reference - live) / scale).reshape(count, -1),
                ((future_reference[:, 0] - current_reference) / scale).reshape(
                    count, -1
                ),
                ((future_reference[:, 1:] - live[:, None]) / scale).reshape(
                    count, -1
                ),
            )
        raise RuntimeError("validated object target representation is unsupported")


__all__ = [
    "CurrentReferenceRobotTargetFeature",
    "ObjectTargetRepresentation",
    "ReferenceRobotTargetRepresentation",
    "TrajectoryObservationAssembler",
    "TrajectoryObservationInput",
    "TrajectoryObservationSpec",
]
