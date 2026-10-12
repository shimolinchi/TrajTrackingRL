"""Composable current-reference robot targets for trajectory observations."""

from __future__ import annotations

from typing import Literal, cast

from torch import Tensor

from .actions import CanonicalActionSpec


CurrentReferenceRobotTargetFeature = Literal[
    "arm_joint_position",
    "link7_position",
    "hand_joint_position",
]
CURRENT_REFERENCE_ROBOT_TARGET_OBSERVATION_VERSION = (
    "current_reference_robot_target_v1"
)

_FEATURE_LAYOUT: tuple[tuple[CurrentReferenceRobotTargetFeature, str, int], ...] = (
    (
        "arm_joint_position",
        "current_reference_arm_joint_position_normalized",
        7,
    ),
    (
        "link7_position",
        "current_reference_link7_position_scaled",
        3,
    ),
    (
        "hand_joint_position",
        "current_reference_hand_joint_position_normalized",
        20,
    ),
)
_SUPPORTED_FEATURES = tuple(item[0] for item in _FEATURE_LAYOUT)


def normalize_current_reference_robot_target_features(
    features: tuple[str, ...],
) -> tuple[CurrentReferenceRobotTargetFeature, ...]:
    """Validate a selection and return one checkpoint-stable canonical order."""

    if not isinstance(features, tuple):
        raise TypeError("current reference robot target features must be a tuple")
    unknown = tuple(
        feature for feature in features if feature not in _SUPPORTED_FEATURES
    )
    if unknown:
        raise ValueError(
            "unsupported current reference robot target features: "
            f"{unknown}; expected a subset of {_SUPPORTED_FEATURES}"
        )
    if len(set(features)) != len(features):
        raise ValueError("current reference robot target features must be unique")
    selected = set(features)
    return cast(
        tuple[CurrentReferenceRobotTargetFeature, ...],
        tuple(feature for feature in _SUPPORTED_FEATURES if feature in selected),
    )


def current_reference_robot_target_feature_layout(
    features: tuple[CurrentReferenceRobotTargetFeature, ...],
) -> tuple[tuple[str, int], ...]:
    selected = set(features)
    return tuple(
        (layout_name, width)
        for feature, layout_name, width in _FEATURE_LAYOUT
        if feature in selected
    )


def current_reference_robot_target_feature_dim(
    features: tuple[CurrentReferenceRobotTargetFeature, ...],
) -> int:
    return sum(
        width
        for _, width in current_reference_robot_target_feature_layout(features)
    )


def current_reference_robot_target_requirements(
    features: tuple[CurrentReferenceRobotTargetFeature, ...],
) -> tuple[bool, bool]:
    """Return whether current reference joint state and link7 pose are needed."""

    selected = set(features)
    return (
        bool(selected.intersection(("arm_joint_position", "hand_joint_position"))),
        "link7_position" in selected,
    )


def assemble_current_reference_robot_target_features(
    features: tuple[CurrentReferenceRobotTargetFeature, ...],
    *,
    action_spec: CanonicalActionSpec,
    reference_joint_position: Tensor | None,
    reference_link7_pose_robot_base: Tensor | None,
    joint_lower: Tensor,
    joint_upper: Tensor,
    link7_position_scale_m: float,
) -> tuple[Tensor, ...]:
    """Assemble selected targets from the current reference cursor.

    Joint targets use the same runtime hard-limit normalization as live joint
    positions.  Link7 contributes xyz only, expressed in robot-base coordinates.
    """

    if not features:
        return ()
    needs_joint_position, needs_link7_pose = (
        current_reference_robot_target_requirements(features)
    )
    normalized_joint_position: Tensor | None = None
    if needs_joint_position:
        if reference_joint_position is None:
            raise RuntimeError("current reference joint position is missing")
        center = 0.5 * (joint_lower + joint_upper)
        half_range = 0.5 * (joint_upper - joint_lower)
        normalized_joint_position = (
            reference_joint_position - center
        ) / half_range
    if needs_link7_pose and reference_link7_pose_robot_base is None:
        raise RuntimeError("current reference link7 pose is missing")

    assembled: list[Tensor] = []
    for feature in features:
        if feature == "arm_joint_position":
            assert normalized_joint_position is not None
            assembled.append(
                normalized_joint_position[:, action_spec.groups[0].indices]
            )
        elif feature == "link7_position":
            assert reference_link7_pose_robot_base is not None
            assembled.append(
                reference_link7_pose_robot_base[:, :3] / link7_position_scale_m
            )
        elif feature == "hand_joint_position":
            assert normalized_joint_position is not None
            assembled.append(
                normalized_joint_position[:, action_spec.groups[1].indices]
            )
        else:  # pragma: no cover - normalized selections make this unreachable.
            raise RuntimeError(f"unsupported current reference feature: {feature}")
    return tuple(assembled)


def current_reference_robot_target_manifest(
    features: tuple[CurrentReferenceRobotTargetFeature, ...],
    *,
    link7_position_scale_m: float,
) -> dict[str, object]:
    """Describe the causal source and normalization of the selected features."""

    return {
        "implementation_version": CURRENT_REFERENCE_ROBOT_TARGET_OBSERVATION_VERSION,
        "features": list(features),
        "timing": "pre_action_current_reference_cursor_t",
        "trajectory_source": {
            "joint_position": "recorded reference robot joint_position at cursor t",
            "link7_position": "recorded reference link7 pose xyz at cursor t",
        },
        "joint_position": (
            "canonical arm/hand subsets normalized by runtime joint hard bounds"
        ),
        "link7_position": {
            "frame": "robot_base",
            "axes": ["x", "y", "z"],
            "scale_m": link7_position_scale_m,
            "orientation_included": False,
        },
        "canonical_feature_order": list(_SUPPORTED_FEATURES),
    }


__all__ = [
    "CURRENT_REFERENCE_ROBOT_TARGET_OBSERVATION_VERSION",
    "CurrentReferenceRobotTargetFeature",
    "assemble_current_reference_robot_target_features",
    "current_reference_robot_target_feature_dim",
    "current_reference_robot_target_feature_layout",
    "current_reference_robot_target_manifest",
    "current_reference_robot_target_requirements",
    "normalize_current_reference_robot_target_features",
]
