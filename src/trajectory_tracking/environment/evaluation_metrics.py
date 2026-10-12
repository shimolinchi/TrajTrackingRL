"""Motion-conditioned tracking diagnostics for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# Motions below one micrometre per 60 Hz control step are numerical/static for
# this dataset.  Keeping this threshold in the metric contract prevents tiny FK
# or quaternion round-off from giving the static prefix non-zero weight.
REFERENCE_MOTION_EPSILON_M_PER_STEP = 1.0e-6


@dataclass(frozen=True, slots=True)
class MotionConditionedTrackingDiagnostics:
    """Per-transition object and robot motion diagnostics.

    Position errors are later weighted by the corresponding reference motion.
    Motion errors compare actual and reference displacement vectors directly.
    Consequently, a static reference step contributes neither weight nor a
    falsely easy zero-error sample, while a static policy on a moving reference
    has normalized motion error one.
    """

    reference_object_motion_m: Tensor
    object_motion_error_m: Tensor
    object_flow_error_m: Tensor
    object_reference_moving: Tensor
    reference_robot_keypoint_motion_m: Tensor
    robot_keypoint_motion_error_m: Tensor
    robot_keypoint_position_error_m: Tensor
    robot_reference_moving: Tensor


def _validate_point_tracks(
    values: tuple[tuple[str, Tensor], ...],
) -> tuple[int, torch.device]:
    first_name, first = values[0]
    if (
        not isinstance(first, Tensor)
        or first.ndim != 3
        or first.shape[0] <= 0
        or first.shape[1] <= 0
        or first.shape[2] != 3
        or first.dtype != torch.float32
        or not torch.isfinite(first).all().item()
    ):
        raise ValueError(f"{first_name} must be finite float32 [N,P,3]")
    batch = int(first.shape[0])
    device = first.device
    for name, value in values[1:]:
        if (
            not isinstance(value, Tensor)
            or value.shape != first.shape
            or value.dtype != torch.float32
            or value.device != device
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(
                f"{name} must match {first_name}'s finite float32 [N,P,3] contract"
            )
    return batch, device


def motion_conditioned_tracking_diagnostics(
    *,
    actual_object_points_before: Tensor,
    actual_object_points_after: Tensor,
    reference_object_points_before: Tensor,
    reference_object_points_after: Tensor,
    actual_robot_keypoints_before: Tensor,
    actual_robot_keypoints_after: Tensor,
    reference_robot_keypoints_before: Tensor,
    reference_robot_keypoints_after: Tensor,
) -> MotionConditionedTrackingDiagnostics:
    """Compare t->t+1 motion and next-state position on moving references."""

    _validate_point_tracks(
        (
            ("actual_object_points_before", actual_object_points_before),
            ("actual_object_points_after", actual_object_points_after),
            ("reference_object_points_before", reference_object_points_before),
            ("reference_object_points_after", reference_object_points_after),
        )
    )
    _validate_point_tracks(
        (
            ("actual_robot_keypoints_before", actual_robot_keypoints_before),
            ("actual_robot_keypoints_after", actual_robot_keypoints_after),
            ("reference_robot_keypoints_before", reference_robot_keypoints_before),
            ("reference_robot_keypoints_after", reference_robot_keypoints_after),
        )
    )
    if actual_object_points_before.shape[0] != actual_robot_keypoints_before.shape[0]:
        raise ValueError("object and robot motion diagnostics must share batch size")
    if actual_object_points_before.device != actual_robot_keypoints_before.device:
        raise ValueError("object and robot motion diagnostics must share device")

    reference_object_delta = (
        reference_object_points_after - reference_object_points_before
    )
    actual_object_delta = actual_object_points_after - actual_object_points_before
    reference_object_motion = torch.linalg.vector_norm(
        reference_object_delta, dim=-1
    ).mean(-1)
    object_motion_error = torch.linalg.vector_norm(
        actual_object_delta - reference_object_delta, dim=-1
    ).mean(-1)
    object_flow_error = torch.linalg.vector_norm(
        actual_object_points_after - reference_object_points_after, dim=-1
    ).mean(-1)

    reference_robot_delta = (
        reference_robot_keypoints_after - reference_robot_keypoints_before
    )
    actual_robot_delta = actual_robot_keypoints_after - actual_robot_keypoints_before
    reference_robot_motion = torch.linalg.vector_norm(
        reference_robot_delta, dim=-1
    ).mean(-1)
    robot_motion_error = torch.linalg.vector_norm(
        actual_robot_delta - reference_robot_delta, dim=-1
    ).mean(-1)
    robot_position_error = torch.linalg.vector_norm(
        actual_robot_keypoints_after - reference_robot_keypoints_after, dim=-1
    ).mean(-1)

    return MotionConditionedTrackingDiagnostics(
        reference_object_motion_m=reference_object_motion,
        object_motion_error_m=object_motion_error,
        object_flow_error_m=object_flow_error,
        object_reference_moving=(
            reference_object_motion > REFERENCE_MOTION_EPSILON_M_PER_STEP
        ),
        reference_robot_keypoint_motion_m=reference_robot_motion,
        robot_keypoint_motion_error_m=robot_motion_error,
        robot_keypoint_position_error_m=robot_position_error,
        robot_reference_moving=(
            reference_robot_motion > REFERENCE_MOTION_EPSILON_M_PER_STEP
        ),
    )


__all__ = [
    "MotionConditionedTrackingDiagnostics",
    "REFERENCE_MOTION_EPSILON_M_PER_STEP",
    "motion_conditioned_tracking_diagnostics",
]
