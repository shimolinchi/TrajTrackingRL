"""Wrist/hand kinematic imitation and absolute object tracking reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, isfinite, log
from typing import Literal

import torch
from torch import Tensor

from ..utils.kinematics import (
    WUJI_KEYPOINT_COUNT,
    WUJI_LEVEL_1_KEYPOINT_INDICES,
    WUJI_LEVEL_2_KEYPOINT_INDICES,
    WUJI_TIP_KEYPOINT_INDICES,
)
from .actions import CanonicalActionSpec


MANIPTRANS_REWARD_COMBINATION = "maniptrans_wrist_hand_kinematics"
MANIPTRANS_SOURCE_REVISION = "a3d08cfe3c3a5868a7f057533bcaf759c5af4705"


@dataclass(frozen=True, slots=True)
class ManipTransComponentWeights:
    """Weights for the xArm7/Wuji adaptation of the ManipTrans reward."""

    wrist_position: float = 0.1
    wrist_orientation: float = 0.6
    thumb_tip: float = 0.9
    index_tip: float = 0.8
    middle_tip: float = 0.75
    ring_tip: float = 0.6
    pinky_tip: float = 0.6
    level_1: float = 0.5
    level_2: float = 0.3
    wrist_linear_velocity: float = 0.1
    wrist_angular_velocity: float = 0.05
    hand_keypoint_velocity: float = 0.1
    fingertip_velocity: float = 0.0
    hand_power: float = 0.5
    wrist_power: float = 0.5
    hand_joint_position: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(self.to_manifest(include_disabled=True).values())
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "ManipTrans component weights must be finite and non-negative"
            )
        if sum(values) <= 0.0:
            raise ValueError(
                "at least one ManipTrans component weight must be positive"
            )

    @property
    def total(self) -> float:
        return fsum(self.to_manifest(include_disabled=True).values())

    def to_manifest(self, *, include_disabled: bool = False) -> dict[str, float]:
        values = {
            "wrist_position": self.wrist_position,
            "wrist_orientation": self.wrist_orientation,
            "thumb_tip": self.thumb_tip,
            "index_tip": self.index_tip,
            "middle_tip": self.middle_tip,
            "ring_tip": self.ring_tip,
            "pinky_tip": self.pinky_tip,
            "level_1": self.level_1,
            "level_2": self.level_2,
            "wrist_linear_velocity": self.wrist_linear_velocity,
            "wrist_angular_velocity": self.wrist_angular_velocity,
            "hand_keypoint_velocity": self.hand_keypoint_velocity,
            "hand_power": self.hand_power,
            "wrist_power": self.wrist_power,
        }
        if include_disabled or self.fingertip_velocity > 0.0:
            values["fingertip_velocity"] = self.fingertip_velocity
        if include_disabled or self.hand_joint_position > 0.0:
            values["hand_joint_position"] = self.hand_joint_position
        return values


@dataclass(frozen=True, slots=True)
class ManipTransRewardSpec:
    """Curriculum and component weights for the supported reward profile."""

    source_profile: Literal["dexhandimitator_xarm7_wuji_v1"]
    tighten_factor: float
    tighten_simulation_steps: int
    failure_warmup_steps: int
    component_weights: ManipTransComponentWeights = field(
        default_factory=ManipTransComponentWeights
    )

    def __post_init__(self) -> None:
        if self.source_profile != "dexhandimitator_xarm7_wuji_v1":
            raise ValueError(f"unsupported ManipTrans profile: {self.source_profile!r}")
        if not 0.0 < self.tighten_factor <= 1.0:
            raise ValueError("ManipTrans tighten_factor must be in (0,1]")
        if self.tighten_simulation_steps <= 0 or self.failure_warmup_steps < 0:
            raise ValueError("ManipTrans curriculum steps are invalid")

    @property
    def weight_sum(self) -> float:
        return self.component_weights.total

    def to_manifest(self) -> dict[str, object]:
        source_weights = self.component_weights == ManipTransComponentWeights()
        adaptation = {
            "reward_normalization": (
                "source_weighted_sum/6.4*maximum_reward"
                if source_weights
                else "configured_weighted_sum/weight_sum*maximum_reward"
            ),
            "wrist": "xArm7 link7 pose and spatial velocity",
            "finger_identity": (
                "Wuji finger1=thumb, finger2=index, finger3=middle, "
                "finger4=ring, finger5=pinky"
            ),
            "hand_keypoints": (
                "25 Wuji points: link1/link2/link3/link4/fingertip per finger"
            ),
            "level_1_keypoints": "link1/link2 origins for every Wuji finger",
            "level_2_keypoints": "link3/link4 origins for every Wuji finger",
            "hand_power": "sum(abs(hand_joint_effort*joint_velocity))",
            "wrist_power": "abs(sum(arm_joint_effort*joint_velocity))",
            "baseline_joint_and_link7_terminations": "disabled_for_profile",
        }
        exponential_decays: dict[str, object] = {
            "wrist_position": 40.0,
            "wrist_orientation": 1.0,
            "thumb_tip": 100.0,
            "index_tip": 90.0,
            "middle_tip": 80.0,
            "ring_tip": 60.0,
            "pinky_tip": 60.0,
            "level_1": 50.0,
            "level_2": 40.0,
            "wrist_linear_velocity": 1.0,
            "wrist_angular_velocity": 1.0,
            "hand_keypoint_velocity": 1.0,
            "hand_power": 10.0,
            "wrist_power": 2.0,
        }
        if self.component_weights.hand_joint_position > 0.0:
            adaptation["hand_joint_position"] = (
                "20-joint post-action RMSE against reference t+1"
            )
            exponential_decays["hand_joint_position"] = (
                "exp(-ln(2)*(rmse/hand_half_reward_error_rad)^2)"
            )
        if self.component_weights.fingertip_velocity > 0.0:
            adaptation["fingertip_velocity"] = (
                "mean absolute XYZ velocity error over five fingertips only"
            )
            exponential_decays["fingertip_velocity"] = 1.0
        payload: dict[str, object] = {
            "source": "submodules/ManipTrans DexHandImitator reward math",
            "source_revision": MANIPTRANS_SOURCE_REVISION,
            "source_profile": self.source_profile,
            "adaptation": adaptation,
            "exponential_decays": exponential_decays,
            "component_weights": self.component_weights.to_manifest(),
            "early_termination": {
                "failure_warmup_steps": self.failure_warmup_steps,
                "base_threshold_m": {
                    "thumb_tip": 0.04,
                    "index_tip": 0.045,
                    "middle_tip": 0.05,
                    "ring_tip": 0.06,
                    "pinky_tip": 0.06,
                    "level_1": 0.07,
                    "level_2": 0.08,
                },
                "tighten_method": "exp_decay",
                "tighten_factor": self.tighten_factor,
                "tighten_simulation_steps": self.tighten_simulation_steps,
                "simulation_step_rate_hz": 60,
            },
        }
        if not source_weights:
            payload["component_weight_sum"] = self.weight_sum
        return payload




@dataclass(frozen=True, slots=True)
class TrajectoryTrackingRewardSpec:
    """ManipTrans imitation with optional, independently switchable add-ons."""

    maximum_reward: float
    object_flow_diagnostic_half_error_m: float
    arm_joint_position_half_reward_error_rad: float
    hand_joint_position_half_reward_error_rad: float
    maniptrans: ManipTransRewardSpec
    combination: Literal[
        "maniptrans_wrist_hand_kinematics",
    ] = (
        MANIPTRANS_REWARD_COMBINATION
    )
    object_flow_reward_weight: float = 0.0
    robot_tracking_reward_enabled: bool = True
    joint_velocity_penalty_weight: float = 0.0
    arm_joint_velocity_penalty_scale_rad_s: float = 4.0
    hand_joint_velocity_penalty_scale_rad_s: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.robot_tracking_reward_enabled, bool):
            raise TypeError("robot_tracking_reward_enabled must be boolean")
        values = (
            self.maximum_reward,
            self.object_flow_diagnostic_half_error_m,
            self.arm_joint_position_half_reward_error_rad,
            self.hand_joint_position_half_reward_error_rad,
            self.arm_joint_velocity_penalty_scale_rad_s,
            self.hand_joint_velocity_penalty_scale_rad_s,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("reward scales must be finite and positive")
        if self.maximum_reward != 100.0:
            raise ValueError(
                "the trajectory tracking reward requires maximum_reward=100"
            )
        if self.combination not in (
            MANIPTRANS_REWARD_COMBINATION,
        ):
            raise ValueError("unsupported trajectory reward profile")
        for name in (
            "object_flow_reward_weight",
            "joint_velocity_penalty_weight",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.joint_velocity_penalty_weight > 1.0:
            raise ValueError("joint_velocity_penalty_weight must not exceed one")
        if not self.robot_tracking_reward_enabled and (
            self.combination != MANIPTRANS_REWARD_COMBINATION
            or self.object_flow_reward_weight <= 0.0
            or self.joint_velocity_penalty_weight > 0.0
        ):
            raise ValueError(
                "object-only tracking requires the ManipTrans diagnostic profile, "
                "a positive absolute object-flow weight, and no reward add-ons"
            )

    @property
    def raw_weight_sum(self) -> float:
        return self.maniptrans.weight_sum

    @property
    def learner_scale_weight_sum(self) -> float:
        """Map the public score to the learner reward scale."""

        return self.maniptrans.weight_sum

    def to_manifest(self) -> dict[str, object]:
        profile = self.maniptrans.to_manifest()
        if self.maniptrans.component_weights.hand_joint_position > 0.0:
            profile["hand_joint_position_half_reward_error_rad"] = (
                self.hand_joint_position_half_reward_error_rad
            )
        payload: dict[str, object] = {
            "objective": "maniptrans_style_wrist_and_hand_next_state_imitation",
            "combination": self.combination,
            "reward_range": [0.0, self.maximum_reward],
            "maximum_reward": self.maximum_reward,
            "object_flow": {
                "role": "diagnostic_only",
                "enabled_in_reward": False,
                "diagnostic_half_error_m": self.object_flow_diagnostic_half_error_m,
            },
            "profile": profile,
            "success_terminates_episode": False,
        }
        if not self.robot_tracking_reward_enabled:
            payload.update(
                {
                    "objective": "object_flow_next_state_tracking_only",
                    "robot_tracking": {
                        "role": "diagnostic_and_termination_only",
                        "enabled_in_reward": False,
                    },
                    "object_flow": {
                        "role": "sole_reward_component_and_diagnostic",
                        "enabled_in_reward": True,
                        "diagnostic_half_error_m": (
                            self.object_flow_diagnostic_half_error_m
                        ),
                        "quality_formula": (
                            "exp(-ln(2)*(mean_point_error_m/"
                            "diagnostic_half_error_m)^2)"
                        ),
                    },
                    "object_flow_reward_composition": {
                        "formula": "q_flow",
                        "range_preserved": [0.0, self.maximum_reward],
                        "benchmark_relative_weight": (
                            self.object_flow_reward_weight
                        ),
                        "benchmark_relative_weight_role": (
                            "object-reward enablement lineage; no mixture scaling"
                        ),
                    },
                }
            )
            return payload
        if self.object_flow_reward_weight > 0.0:
            flow = payload["object_flow"]
            assert isinstance(flow, dict)
            flow.update(
                {
                    "role": "reward_component_and_diagnostic",
                    "enabled_in_reward": True,
                    "quality_formula": (
                        "exp(-ln(2)*(mean_point_error_m/"
                        "diagnostic_half_error_m)^2)"
                    ),
                    "weight_relative_to_maniptrans": (
                        self.object_flow_reward_weight
                    ),
                }
            )
            payload["objective"] = (
                "maniptrans_style_wrist_hand_and_object_flow_"
                "next_state_imitation"
            )
            composition_formula = (
                "(q_maniptrans+w_flow*q_flow)/(1+w_flow)"
            )
            payload["object_flow_reward_composition"] = {
                "formula": composition_formula,
                "range_preserved": [0.0, self.maximum_reward],
            }
        if self.joint_velocity_penalty_weight > 0.0:
            payload["joint_velocity_penalty"] = {
                "source": "actual_post_action_joint_velocity",
                "reference_velocity_used": False,
                "group_balance": "mean_of_arm_and_hand_normalized_squared_speed",
                "arm_scale_rad_s": self.arm_joint_velocity_penalty_scale_rad_s,
                "hand_scale_rad_s": self.hand_joint_velocity_penalty_scale_rad_s,
                "clamp": [0.0, 1.0],
                "weight": self.joint_velocity_penalty_weight,
                "formula": "clamp(q_primary-w_velocity*p_velocity,0,1)",
            }
        return payload


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingReward:
    """Reward plus the compact diagnostics consumed by training reports."""

    mean_point_error_m: Tensor
    object_flow_quality: Tensor
    arm_joint_position_rmse_rad: Tensor
    hand_joint_position_rmse_rad: Tensor
    arm_joint_position_quality: Tensor
    hand_joint_position_quality: Tensor
    robot_joint_position_quality: Tensor
    joint_velocity_penalty: Tensor
    reward_quality: Tensor
    reward: Tensor
    within_robot_success_threshold: Tensor


@dataclass(frozen=True, slots=True)
class ManipTransTrackingDiagnostics:
    wrist_position_error_m: Tensor
    wrist_orientation_error_rad: Tensor
    thumb_tip_error_m: Tensor
    index_tip_error_m: Tensor
    middle_tip_error_m: Tensor
    ring_tip_error_m: Tensor
    pinky_tip_error_m: Tensor
    level_1_error_m: Tensor
    level_2_error_m: Tensor
    curriculum_scale: float
    within_tightened_threshold: Tensor
    numerical_sanity_failed: Tensor
    failed: Tensor


def _validate_flow_points(live: Tensor, reference: Tensor) -> None:
    if (
        not isinstance(live, Tensor)
        or not isinstance(reference, Tensor)
        or live.ndim != 3
        or live.shape != reference.shape
        or live.shape[0] <= 0
        or live.shape[1] <= 0
        or live.shape[2] != 3
        or live.dtype != torch.float32
        or reference.dtype != torch.float32
        or live.device != reference.device
    ):
        raise ValueError("flow points must be matching float32 [N,P,3] tensors")
    if not torch.logical_and(
        torch.isfinite(live).all(), torch.isfinite(reference).all()
    ).item():
        raise ValueError("flow points contain non-finite values")


def object_flow_tracking_quality(
    live_points_robot_base_m: Tensor,
    reference_points_robot_base_m: Tensor,
    *,
    half_reward_error_m: float,
) -> tuple[Tensor, Tensor]:
    """Return four-point mean error and its diagnostic 0--1 quality."""

    _validate_flow_points(live_points_robot_base_m, reference_points_robot_base_m)
    if not isfinite(half_reward_error_m) or half_reward_error_m <= 0.0:
        raise ValueError("object-flow half-reward error must be finite and positive")
    point_error = torch.linalg.vector_norm(
        live_points_robot_base_m - reference_points_robot_base_m,
        dim=-1,
    ).mean(dim=-1)
    quality = torch.exp(-log(2.0) * (point_error / half_reward_error_m).square())
    return point_error, quality


def _joint_tracking_diagnostics(
    live_points_robot_base_m: Tensor,
    reference_points_robot_base_m: Tensor,
    actual_joint_position_rad: Tensor,
    next_reference_joint_position_rad: Tensor,
    action_spec: CanonicalActionSpec,
    spec: TrajectoryTrackingRewardSpec,
) -> TrajectoryTrackingReward:
    """Build reward-independent flow and joint diagnostics for one next state."""

    _validate_flow_points(live_points_robot_base_m, reference_points_robot_base_m)
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
        batch != live_points_robot_base_m.shape[0]
        or reference_batch != batch
        or actual_joint_position_rad.device != live_points_robot_base_m.device
        or next_reference_joint_position_rad.device != live_points_robot_base_m.device
    ):
        raise ValueError("flow points and robot states must share batch and device")

    point_error, flow_quality = object_flow_tracking_quality(
        live_points_robot_base_m,
        reference_points_robot_base_m,
        half_reward_error_m=spec.object_flow_diagnostic_half_error_m,
    )
    joint_error = actual_joint_position_rad - next_reference_joint_position_rad
    arm_indices = action_spec.groups[0].indices
    hand_indices = action_spec.groups[1].indices
    arm_rmse = joint_error[:, arm_indices].square().mean(dim=-1).sqrt()
    hand_rmse = joint_error[:, hand_indices].square().mean(dim=-1).sqrt()
    arm_quality = torch.exp(
        -log(2.0) * (arm_rmse / spec.arm_joint_position_half_reward_error_rad).square()
    )
    hand_quality = torch.exp(
        -log(2.0)
        * (hand_rmse / spec.hand_joint_position_half_reward_error_rad).square()
    )
    zero = torch.zeros_like(point_error)
    false = torch.zeros_like(point_error, dtype=torch.bool)
    return TrajectoryTrackingReward(
        mean_point_error_m=point_error,
        object_flow_quality=flow_quality,
        arm_joint_position_rmse_rad=arm_rmse,
        hand_joint_position_rmse_rad=hand_rmse,
        arm_joint_position_quality=arm_quality,
        hand_joint_position_quality=hand_quality,
        robot_joint_position_quality=zero,
        joint_velocity_penalty=zero,
        reward_quality=zero,
        reward=zero,
        within_robot_success_threshold=false,
    )


def _finite_tensor(
    value: Tensor, *, name: str, shape: tuple[int, ...], device: torch.device
) -> None:
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.float32
        or value.device != device
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{name} must be finite float32 {shape} on one device")


def maniptrans_tracking_reward(
    live_points_robot_base_m: Tensor,
    reference_points_robot_base_m: Tensor,
    actual_joint_position_rad: Tensor,
    next_reference_joint_position_rad: Tensor,
    actual_joint_velocity_rad_s: Tensor,
    next_reference_joint_velocity_rad_s: Tensor,
    actual_link7_pose_robot_base: Tensor,
    next_reference_link7_pose_robot_base: Tensor,
    actual_link7_linear_velocity_robot_base: Tensor,
    next_reference_link7_linear_velocity_robot_base: Tensor,
    actual_link7_angular_velocity_robot_base: Tensor,
    next_reference_link7_angular_velocity_robot_base: Tensor,
    actual_hand_keypoints_robot_base: Tensor,
    next_reference_hand_keypoints_robot_base: Tensor,
    actual_hand_keypoint_velocities_robot_base: Tensor,
    next_reference_hand_keypoint_velocities_robot_base: Tensor,
    actual_joint_effort: Tensor,
    episode_steps: Tensor,
    simulation_frame_count: int,
    action_spec: CanonicalActionSpec,
    spec: TrajectoryTrackingRewardSpec,
) -> tuple[TrajectoryTrackingReward, ManipTransTrackingDiagnostics]:
    """Evaluate the checked-out ManipTrans reward structure on xArm7/Wuji."""

    batch = action_spec.validate(
        actual_joint_position_rad,
        name="actual_joint_position_rad",
        check_bounds=False,
    )
    action_spec.validate(
        next_reference_joint_position_rad,
        name="next_reference_joint_position_rad",
        check_bounds=False,
    )
    action_spec.validate(
        actual_joint_velocity_rad_s,
        name="actual_joint_velocity_rad_s",
        check_bounds=False,
    )
    action_spec.validate(
        next_reference_joint_velocity_rad_s,
        name="next_reference_joint_velocity_rad_s",
        check_bounds=False,
    )
    device = actual_joint_position_rad.device
    base = _joint_tracking_diagnostics(
        live_points_robot_base_m,
        reference_points_robot_base_m,
        actual_joint_position_rad,
        next_reference_joint_position_rad,
        action_spec,
        spec,
    )
    expected = {
        "actual_link7_pose_robot_base": (batch, 7),
        "next_reference_link7_pose_robot_base": (batch, 7),
        "actual_link7_linear_velocity_robot_base": (batch, 3),
        "next_reference_link7_linear_velocity_robot_base": (batch, 3),
        "actual_link7_angular_velocity_robot_base": (batch, 3),
        "next_reference_link7_angular_velocity_robot_base": (batch, 3),
        "actual_hand_keypoints_robot_base": (batch, WUJI_KEYPOINT_COUNT, 3),
        "next_reference_hand_keypoints_robot_base": (
            batch,
            WUJI_KEYPOINT_COUNT,
            3,
        ),
        "actual_hand_keypoint_velocities_robot_base": (
            batch,
            WUJI_KEYPOINT_COUNT,
            3,
        ),
        "next_reference_hand_keypoint_velocities_robot_base": (
            batch,
            WUJI_KEYPOINT_COUNT,
            3,
        ),
        "actual_joint_effort": (batch, action_spec.action_dim),
    }
    values = locals()
    for name, shape in expected.items():
        _finite_tensor(values[name], name=name, shape=shape, device=device)
    if (
        not isinstance(episode_steps, Tensor)
        or episode_steps.shape != (batch,)
        or episode_steps.dtype != torch.int64
        or episode_steps.device != device
        or simulation_frame_count < 0
    ):
        raise ValueError("ManipTrans progress inputs are invalid")

    wrist_position_error = torch.linalg.vector_norm(
        actual_link7_pose_robot_base[:, :3]
        - next_reference_link7_pose_robot_base[:, :3],
        dim=-1,
    )
    actual_quaternion = actual_link7_pose_robot_base[:, 3:]
    reference_quaternion = next_reference_link7_pose_robot_base[:, 3:]
    actual_quaternion = actual_quaternion / torch.linalg.vector_norm(
        actual_quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    reference_quaternion = reference_quaternion / torch.linalg.vector_norm(
        reference_quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    quaternion_dot = (actual_quaternion * reference_quaternion).sum(-1).abs()
    wrist_orientation_error = 2.0 * torch.acos(quaternion_dot.clamp(max=1.0))

    keypoint_distance = torch.linalg.vector_norm(
        actual_hand_keypoints_robot_base - next_reference_hand_keypoints_robot_base,
        dim=-1,
    )
    tip_errors = tuple(
        keypoint_distance[:, index] for index in WUJI_TIP_KEYPOINT_INDICES
    )
    thumb_error, index_error, middle_error, ring_error, pinky_error = tip_errors
    level_1_error = keypoint_distance[:, WUJI_LEVEL_1_KEYPOINT_INDICES].mean(-1)
    level_2_error = keypoint_distance[:, WUJI_LEVEL_2_KEYPOINT_INDICES].mean(-1)

    wrist_linear_velocity_error = (
        (
            actual_link7_linear_velocity_robot_base
            - next_reference_link7_linear_velocity_robot_base
        )
        .abs()
        .mean(-1)
    )
    wrist_angular_velocity_error = (
        (
            actual_link7_angular_velocity_robot_base
            - next_reference_link7_angular_velocity_robot_base
        )
        .abs()
        .mean(-1)
    )
    keypoint_velocity_delta = (
        actual_hand_keypoint_velocities_robot_base
        - next_reference_hand_keypoint_velocities_robot_base
    ).abs()
    keypoint_velocity_error = keypoint_velocity_delta.mean(dim=(-1, -2))
    fingertip_velocity_error = keypoint_velocity_delta[
        :, WUJI_TIP_KEYPOINT_INDICES
    ].mean(dim=(-1, -2))

    arm_indices = action_spec.groups[0].indices
    hand_indices = action_spec.groups[1].indices
    wrist_power = (
        (
            actual_joint_effort[:, arm_indices]
            * actual_joint_velocity_rad_s[:, arm_indices]
        )
        .sum(-1)
        .abs()
    )
    hand_power = (
        (
            actual_joint_effort[:, hand_indices]
            * actual_joint_velocity_rad_s[:, hand_indices]
        )
        .abs()
        .sum(-1)
    )

    weights = spec.maniptrans.component_weights
    raw_reward = (
        weights.wrist_position * torch.exp(-40.0 * wrist_position_error)
        + weights.wrist_orientation * torch.exp(-wrist_orientation_error)
        + weights.thumb_tip * torch.exp(-100.0 * thumb_error)
        + weights.index_tip * torch.exp(-90.0 * index_error)
        + weights.middle_tip * torch.exp(-80.0 * middle_error)
        + weights.ring_tip * torch.exp(-60.0 * ring_error)
        + weights.pinky_tip * torch.exp(-60.0 * pinky_error)
        + weights.level_1 * torch.exp(-50.0 * level_1_error)
        + weights.level_2 * torch.exp(-40.0 * level_2_error)
        + weights.wrist_linear_velocity * torch.exp(-wrist_linear_velocity_error)
        + weights.wrist_angular_velocity * torch.exp(-wrist_angular_velocity_error)
        + weights.hand_keypoint_velocity * torch.exp(-keypoint_velocity_error)
        + weights.fingertip_velocity * torch.exp(-fingertip_velocity_error)
        + weights.hand_power * torch.exp(-10.0 * hand_power)
        + weights.wrist_power * torch.exp(-2.0 * wrist_power)
        + weights.hand_joint_position * base.hand_joint_position_quality
    )
    maniptrans_quality = raw_reward / spec.raw_weight_sum
    quality = (
        maniptrans_quality
        if spec.robot_tracking_reward_enabled
        else base.object_flow_quality
    )
    if (
        spec.robot_tracking_reward_enabled
        and spec.object_flow_reward_weight > 0.0
    ):
        weight = spec.object_flow_reward_weight
        quality = (quality + weight * base.object_flow_quality) / (1.0 + weight)

    joint_velocity_penalty = torch.zeros_like(quality)
    if spec.joint_velocity_penalty_weight > 0.0:
        arm_speed_penalty = (
            actual_joint_velocity_rad_s[:, arm_indices].abs()
            / spec.arm_joint_velocity_penalty_scale_rad_s
        ).square().mean(-1).clamp(max=1.0)
        hand_speed_penalty = (
            actual_joint_velocity_rad_s[:, hand_indices].abs()
            / spec.hand_joint_velocity_penalty_scale_rad_s
        ).square().mean(-1).clamp(max=1.0)
        joint_velocity_penalty = 0.5 * (
            arm_speed_penalty + hand_speed_penalty
        )
        quality = (
            quality
            - spec.joint_velocity_penalty_weight * joint_velocity_penalty
        ).clamp(0.0, 1.0)
    reward = spec.maximum_reward * quality

    curriculum_scale = (torch.e * 2.0) ** (
        -float(simulation_frame_count) / spec.maniptrans.tighten_simulation_steps
    ) * (1.0 - spec.maniptrans.tighten_factor) + spec.maniptrans.tighten_factor
    threshold_multiplier = curriculum_scale / 0.7
    within_threshold = (
        (thumb_error <= 0.04 * threshold_multiplier)
        & (index_error <= 0.045 * threshold_multiplier)
        & (middle_error <= 0.05 * threshold_multiplier)
        & (ring_error <= 0.06 * threshold_multiplier)
        & (pinky_error <= 0.06 * threshold_multiplier)
        & (level_1_error <= 0.07 * threshold_multiplier)
        & (level_2_error <= 0.08 * threshold_multiplier)
    )
    sanity_failure = (
        (
            torch.linalg.vector_norm(actual_link7_linear_velocity_robot_base, dim=-1)
            > 100.0
        )
        | (
            torch.linalg.vector_norm(actual_link7_angular_velocity_robot_base, dim=-1)
            > 200.0
        )
        | (
            torch.linalg.vector_norm(
                actual_hand_keypoint_velocities_robot_base, dim=-1
            ).mean(-1)
            > 100.0
        )
        | (actual_joint_velocity_rad_s[:, hand_indices].abs().mean(-1) > 200.0)
    )
    failed = (
        (~within_threshold) & (episode_steps >= spec.maniptrans.failure_warmup_steps)
    ) | sanity_failure
    tracking = TrajectoryTrackingReward(
        mean_point_error_m=base.mean_point_error_m,
        object_flow_quality=base.object_flow_quality,
        arm_joint_position_rmse_rad=base.arm_joint_position_rmse_rad,
        hand_joint_position_rmse_rad=base.hand_joint_position_rmse_rad,
        arm_joint_position_quality=base.arm_joint_position_quality,
        hand_joint_position_quality=base.hand_joint_position_quality,
        robot_joint_position_quality=maniptrans_quality,
        joint_velocity_penalty=joint_velocity_penalty,
        reward_quality=quality,
        reward=reward,
        within_robot_success_threshold=within_threshold & ~sanity_failure,
    )
    diagnostics = ManipTransTrackingDiagnostics(
        wrist_position_error_m=wrist_position_error,
        wrist_orientation_error_rad=wrist_orientation_error,
        thumb_tip_error_m=thumb_error,
        index_tip_error_m=index_error,
        middle_tip_error_m=middle_error,
        ring_tip_error_m=ring_error,
        pinky_tip_error_m=pinky_error,
        level_1_error_m=level_1_error,
        level_2_error_m=level_2_error,
        curriculum_scale=float(curriculum_scale),
        within_tightened_threshold=within_threshold,
        numerical_sanity_failed=sanity_failure,
        failed=failed,
    )
    return tracking, diagnostics




def trajectory_tracking_reward(
    *args: object,
    spec: TrajectoryTrackingRewardSpec,
) -> tuple[TrajectoryTrackingReward, ManipTransTrackingDiagnostics]:
    """Dispatch an explicitly configured reward profile."""

    return maniptrans_tracking_reward(*args, spec=spec)  # type: ignore[arg-type]



__all__ = [
    "MANIPTRANS_REWARD_COMBINATION",
    "MANIPTRANS_SOURCE_REVISION",
    "ManipTransComponentWeights",
    "ManipTransRewardSpec",
    "ManipTransTrackingDiagnostics",
    "TrajectoryTrackingReward",
    "TrajectoryTrackingRewardSpec",
    "maniptrans_tracking_reward",
    "trajectory_tracking_reward",
    "object_flow_tracking_quality",
]
