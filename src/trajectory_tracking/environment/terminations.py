"""Pure-Torch robot/object tracking diagnostics and reset classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch
from torch import Tensor

from .actions import CanonicalActionSpec


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingTerminationSpec:
    """Robot/object reset policy with legacy checkpoint-compatible defaults.

    Legacy ManipTrans runs do not terminate on the stored joint/link7 diagnostic
    thresholds.  New runs can explicitly select the position-dominant link7 pose
    rule without changing any existing manifest or checkpoint identity.
    """

    mean_absolute_joint_position_error_threshold_rad: float
    link7_position_error_threshold_m: float
    object_departure_position_error_threshold_m: float | None = None
    object_departure_failure_warmup_steps: int = 0
    object_departure_reset_enabled: bool = False
    robot_tracking_reset_mode: Literal[
        "maniptrans_keypoints", "link7_weighted_pose"
    ] = "maniptrans_keypoints"
    link7_weighted_pose_error_threshold_m: float | None = None
    link7_orientation_error_weight_m_per_rad: float = 0.0
    link7_pose_failure_warmup_steps: int = 20

    def __post_init__(self) -> None:
        thresholds = (
            self.mean_absolute_joint_position_error_threshold_rad,
            self.link7_position_error_threshold_m,
        )
        if not all(isfinite(value) and value > 0.0 for value in thresholds):
            raise ValueError("tracking diagnostic thresholds must be positive")
        threshold = self.object_departure_position_error_threshold_m
        if threshold is not None and (not isfinite(threshold) or threshold <= 0.0):
            raise ValueError("object-departure threshold must be positive or None")
        if (
            isinstance(self.object_departure_failure_warmup_steps, bool)
            or self.object_departure_failure_warmup_steps < 0
        ):
            raise ValueError("object-departure warmup must be non-negative")
        if not isinstance(self.object_departure_reset_enabled, bool):
            raise TypeError("object-departure reset switch must be bool")
        if self.object_departure_reset_enabled and threshold is None:
            raise ValueError("object-departure reset requires a threshold")
        if self.robot_tracking_reset_mode not in (
            "maniptrans_keypoints",
            "link7_weighted_pose",
        ):
            raise ValueError(
                "robot-tracking reset mode must be maniptrans_keypoints or "
                "link7_weighted_pose"
            )
        orientation_weight = self.link7_orientation_error_weight_m_per_rad
        if (
            isinstance(orientation_weight, bool)
            or not isfinite(orientation_weight)
            or orientation_weight < 0.0
        ):
            raise ValueError("link7 orientation-error weight must be finite and >= 0")
        if (
            isinstance(self.link7_pose_failure_warmup_steps, bool)
            or not isinstance(self.link7_pose_failure_warmup_steps, int)
            or self.link7_pose_failure_warmup_steps < 0
        ):
            raise ValueError("link7 pose failure warmup must be non-negative")
        pose_threshold = self.link7_weighted_pose_error_threshold_m
        if pose_threshold is not None and (
            isinstance(pose_threshold, bool)
            or not isfinite(pose_threshold)
            or pose_threshold <= 0.0
        ):
            raise ValueError(
                "link7 weighted-pose threshold must be positive or None"
            )
        if (
            self.robot_tracking_reset_mode == "maniptrans_keypoints"
            and (orientation_weight != 0.0 or pose_threshold is not None)
        ):
            raise ValueError(
                "link7 weighted-pose fields require link7 pose reset mode"
            )
        if (
            self.robot_tracking_reset_mode == "link7_weighted_pose"
            and pose_threshold is None
        ):
            raise ValueError("link7 pose reset mode requires its weighted threshold")

    def to_manifest(self) -> dict[str, object]:
        if self.robot_tracking_reset_mode == "link7_weighted_pose":
            payload: dict[str, object] = {
                "timing": "post_action_t_plus_1_vs_reference_t_plus_1",
                "robot_tracking_reset_mode": self.robot_tracking_reset_mode,
                "causes": {
                    "link7_weighted_pose": {
                        "body": "link7",
                        "coordinate_frame": "robot_base",
                        "reference_field": "ee_current_pose_xyz_quaternion_wxyz",
                        "position_metric": (
                            "euclidean_distance(actual_xyz,reference_xyz)"
                        ),
                        "orientation_metric": (
                            "quaternion_geodesic_angle_rad_sign_invariant"
                        ),
                        "formula_m": (
                            "position_error_m + "
                            "orientation_weight_m_per_rad * orientation_error_rad"
                        ),
                        "orientation_weight_m_per_rad": (
                            self.link7_orientation_error_weight_m_per_rad
                        ),
                        "threshold_m": self.link7_weighted_pose_error_threshold_m,
                        "warmup_control_steps": (
                            self.link7_pose_failure_warmup_steps
                        ),
                    },
                    "numerical_sanity_guard": {
                        "enabled": True,
                        "purpose": "catastrophic_simulation_state_only",
                    },
                },
                "disabled_robot_reset_inputs": [
                    "all_27_joint_position_error",
                    "Wuji_hand_keypoint_error",
                ],
                "comparison": "strictly_greater_than_threshold",
                "combination": "any_failure",
                "simultaneous_failure_precedence": [
                    "link7_weighted_pose",
                    "numerical_sanity_guard",
                ],
                "ppo_result": "terminated",
                "reset": "asynchronous_on_next_global_control_step",
            }
            if self.object_departure_position_error_threshold_m is not None:
                causes = payload["causes"]
                assert isinstance(causes, dict)
                causes["object_departure"] = self._object_departure_manifest()
            return payload

        payload: dict[str, object] = {
            "timing": "post_action_t_plus_1_vs_reference_t_plus_1",
            "causes": {
                "joint_position": {
                    "joint_scope": "all_27_canonical_robot_joints",
                    "metric": "mean(abs(actual_q-reference_q))",
                    "threshold_rad": (
                        self.mean_absolute_joint_position_error_threshold_rad
                    ),
                },
                "link7_position": {
                    "body": "link7",
                    "coordinate_frame": "robot_base",
                    "reference_field": "ee_current_pose_xyz",
                    "metric": "euclidean_distance(actual_xyz,reference_xyz)",
                    "threshold_m": self.link7_position_error_threshold_m,
                },
            },
            "comparison": "strictly_greater_than_threshold",
            "combination": "any_failure",
            "simultaneous_failure_precedence": [
                "joint_position",
                "link7_position",
            ],
            "ppo_result": "terminated",
            "reset": "asynchronous_on_next_global_control_step",
        }
        if self.object_departure_position_error_threshold_m is not None:
            causes = payload["causes"]
            assert isinstance(causes, dict)
            causes["object_departure"] = self._object_departure_manifest()
        return payload

    def _object_departure_manifest(self) -> dict[str, object]:
        return {
            "coordinate_frame": "robot_base",
            "metric": "norm(actual_object_xyz-reference_object_xyz)",
            "threshold_m": self.object_departure_position_error_threshold_m,
            "warmup_control_steps": self.object_departure_failure_warmup_steps,
            "reset_enabled": self.object_departure_reset_enabled,
            "reported_metric": "object_drop_or_escape_proxy",
        }


@dataclass(frozen=True, slots=True)
class TrajectoryTaskSuccessSpec:
    """Endpoint success rule, separate from reward and reset plugins."""

    mode: Literal[
        "robot_endpoint", "robot_object_trajectory", "object_trajectory"
    ] = "robot_endpoint"
    episode_mean_flow_error_threshold_m: float | None = None
    final_flow_error_threshold_m: float | None = None

    def __post_init__(self) -> None:
        thresholds = (
            self.episode_mean_flow_error_threshold_m,
            self.final_flow_error_threshold_m,
        )
        if self.mode == "robot_endpoint":
            if any(value is not None for value in thresholds):
                raise ValueError("robot-endpoint success must not configure flow limits")
            return
        if self.mode not in ("robot_object_trajectory", "object_trajectory"):
            raise ValueError(f"unsupported task-success mode: {self.mode!r}")
        if any(value is None for value in thresholds):
            raise ValueError("object-tracking success requires mean and final flow limits")
        if not all(isfinite(float(value)) and float(value) > 0.0 for value in thresholds):
            raise ValueError("task-success flow thresholds must be positive")

    def to_manifest(self) -> dict[str, object]:
        if self.mode == "robot_endpoint":
            return {
                "mode": self.mode,
                "definition": "reference_endpoint_and_robot_tracking_threshold",
            }
        return {
            "mode": self.mode,
            "definition": (
                "reference_endpoint_and_robot_tracking_and_object_flow_thresholds"
                if self.mode == "robot_object_trajectory"
                else "reference_endpoint_and_object_flow_thresholds_without_robot_tracking"
            ),
            "episode_mean_flow_error_threshold_m": (
                self.episode_mean_flow_error_threshold_m
            ),
            "final_flow_error_threshold_m": self.final_flow_error_threshold_m,
            "reject_if_object_departed": True,
        }


@dataclass(frozen=True, slots=True)
class JointPositionTracking:
    mean_absolute_error_rad: Tensor


@dataclass(frozen=True, slots=True)
class Link7PositionTracking:
    position_error_m: Tensor


@dataclass(frozen=True, slots=True)
class Link7WeightedPoseFailure:
    weighted_pose_error_m: Tensor
    failed: Tensor


@dataclass(frozen=True, slots=True)
class ObjectDepartureTracking:
    position_error_m: Tensor
    downward_error_m: Tensor
    detected: Tensor
    failed: Tensor


def joint_position_tracking_error(
    actual_joint_position_rad: Tensor,
    next_reference_joint_position_rad: Tensor,
    action_spec: CanonicalActionSpec,
) -> JointPositionTracking:
    """Return post-action all-joint MAE against reference `t+1`."""

    batch = action_spec.validate(
        actual_joint_position_rad,
        name="actual_joint_position_rad",
        check_bounds=False,
    )
    reference_batch = action_spec.validate(
        next_reference_joint_position_rad,
        name="next_reference_joint_position_rad",
        check_bounds=False,
    )
    if (
        reference_batch != batch
        or next_reference_joint_position_rad.device != actual_joint_position_rad.device
    ):
        raise ValueError("actual and reference joint positions must share batch/device")
    return JointPositionTracking(
        mean_absolute_error_rad=(
            actual_joint_position_rad - next_reference_joint_position_rad
        )
        .abs()
        .mean(dim=-1)
    )


def link7_position_tracking_error(
    actual_link7_position_robot_base_m: Tensor,
    next_reference_link7_position_robot_base_m: Tensor,
) -> Link7PositionTracking:
    """Return post-action link7 XYZ distance in the robot-base frame."""

    actual = actual_link7_position_robot_base_m
    reference = next_reference_link7_position_robot_base_m
    for name, value in (
        ("actual_link7_position_robot_base_m", actual),
        ("next_reference_link7_position_robot_base_m", reference),
    ):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 2
            or value.shape[1] != 3
            or value.dtype != torch.float32
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{name} must be finite float32 [N,3]")
    if actual.shape != reference.shape or actual.device != reference.device:
        raise ValueError("actual and reference link7 positions must share shape/device")
    return Link7PositionTracking(
        position_error_m=torch.linalg.vector_norm(actual - reference, dim=-1)
    )


def link7_weighted_pose_failure(
    position_error_m: Tensor,
    orientation_error_rad: Tensor,
    numerical_sanity_failed: Tensor,
    episode_steps: Tensor,
    spec: TrajectoryTrackingTerminationSpec,
) -> Link7WeightedPoseFailure:
    """Classify the position-dominant link7 pose reset without joint inputs."""

    if spec.robot_tracking_reset_mode != "link7_weighted_pose":
        raise ValueError("link7 weighted-pose failure requires its reset mode")
    if not all(
        isinstance(value, Tensor)
        for value in (
            position_error_m,
            orientation_error_rad,
            numerical_sanity_failed,
            episode_steps,
        )
    ):
        raise ValueError("link7 pose-reset inputs must be tensors")
    shape = position_error_m.shape
    device = position_error_m.device
    if (
        position_error_m.ndim != 1
        or orientation_error_rad.shape != shape
        or numerical_sanity_failed.shape != shape
        or episode_steps.shape != shape
        or position_error_m.dtype != torch.float32
        or orientation_error_rad.dtype != torch.float32
        or numerical_sanity_failed.dtype != torch.bool
        or episode_steps.dtype != torch.int64
        or orientation_error_rad.device != device
        or numerical_sanity_failed.device != device
        or episode_steps.device != device
        or not torch.isfinite(position_error_m).all().item()
        or not torch.isfinite(orientation_error_rad).all().item()
        or (position_error_m < 0.0).any().item()
        or (orientation_error_rad < 0.0).any().item()
    ):
        raise ValueError("link7 pose-reset inputs must be finite aligned [N] tensors")
    weighted_error = (
        position_error_m
        + spec.link7_orientation_error_weight_m_per_rad * orientation_error_rad
    )
    threshold_m = spec.link7_weighted_pose_error_threshold_m
    assert threshold_m is not None
    threshold_exceeded = (
        position_error_m.to(dtype=torch.float64)
        + spec.link7_orientation_error_weight_m_per_rad
        * orientation_error_rad.to(dtype=torch.float64)
        > threshold_m
    )
    failed = (
        threshold_exceeded
        & (episode_steps >= spec.link7_pose_failure_warmup_steps)
    ) | numerical_sanity_failed
    return Link7WeightedPoseFailure(
        weighted_pose_error_m=weighted_error,
        failed=failed,
    )


def object_departure_tracking(
    actual_object_pose_robot_base: Tensor,
    next_reference_object_pose_robot_base: Tensor,
    episode_steps: Tensor,
    spec: TrajectoryTrackingTerminationSpec,
) -> ObjectDepartureTracking:
    """Measure reference-relative object escape and optionally request reset."""

    actual = actual_object_pose_robot_base
    reference = next_reference_object_pose_robot_base
    for name, value in (
        ("actual_object_pose_robot_base", actual),
        ("next_reference_object_pose_robot_base", reference),
    ):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 2
            or value.shape[1] != 7
            or value.dtype != torch.float32
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{name} must be finite float32 [N,7]")
    if actual.shape != reference.shape or actual.device != reference.device:
        raise ValueError("actual and reference object poses must share shape/device")
    if (
        not isinstance(episode_steps, Tensor)
        or episode_steps.shape != actual.shape[:1]
        or episode_steps.dtype != torch.int64
        or episode_steps.device != actual.device
    ):
        raise ValueError("object-departure episode steps must be int64 [N]")
    delta = actual[:, :3] - reference[:, :3]
    position_error = torch.linalg.vector_norm(delta, dim=-1)
    downward_error = (reference[:, 2] - actual[:, 2]).clamp_min(0.0)
    detected = torch.zeros_like(episode_steps, dtype=torch.bool)
    threshold = spec.object_departure_position_error_threshold_m
    if threshold is not None:
        classification_error = torch.sqrt(
            delta.square().sum(dim=-1, dtype=torch.float64)
        )
        detected = (
            classification_error > threshold
        ) & (episode_steps >= spec.object_departure_failure_warmup_steps)
    return ObjectDepartureTracking(
        position_error_m=position_error,
        downward_error_m=downward_error,
        detected=detected,
        failed=detected & spec.object_departure_reset_enabled,
    )


__all__ = [
    "JointPositionTracking",
    "Link7PositionTracking",
    "Link7WeightedPoseFailure",
    "ObjectDepartureTracking",
    "TrajectoryTaskSuccessSpec",
    "TrajectoryTrackingTerminationSpec",
    "joint_position_tracking_error",
    "link7_position_tracking_error",
    "link7_weighted_pose_failure",
    "object_departure_tracking",
]
