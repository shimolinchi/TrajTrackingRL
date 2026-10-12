"""Strict, path-safe configuration for the independent trajectory task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import floor, isclose, isfinite
import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from .environment.actions import DirectActionSpec
from .environment.action_smoothness import ActionSmoothnessSpec
from .environment.joint_acceleration import JointAccelerationSpec
from .environment.observations import TrajectoryObservationSpec
from .environment.perturbations import RandomActionPerturbationSpec
from .environment.scene_config import TrajectorySceneRuntimeSpec
from .environment.rewards import (
    ManipTransComponentWeights,
    ManipTransRewardSpec,
    TrajectoryTrackingRewardSpec,
)
from .environment.terminations import (
    TrajectoryTaskSuccessSpec,
    TrajectoryTrackingTerminationSpec,
)
from .training.ppo import (
    PPOBenchmarkLineageSpec,
    PPOConfig,
    TRAJECTORY_PPO_BENCHMARK_HIDDEN_DIMS,
    TRAJECTORY_PPO_BENCHMARK_INITIAL_STD,
)


class TrajectoryConfigError(ValueError):
    """Raised when a trajectory task config is incomplete or ambiguous."""


TRAJECTORY_REFERENCE_SOURCE_CONTROL_HZ = 60


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise TrajectoryConfigError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _section(value: object, name: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryConfigError(f"{name} must be a mapping")
    data = cast(Mapping[str, Any], value)
    missing = sorted(set(keys) - set(data))
    unexpected = sorted(set(data) - set(keys))
    if missing or unexpected:
        raise TrajectoryConfigError(
            f"{name} keys disagree; missing={missing}, unexpected={unexpected}"
        )
    return data


def _section_with_optional(
    value: object,
    name: str,
    required_keys: tuple[str, ...],
    optional_keys: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryConfigError(f"{name} must be a mapping")
    data = cast(Mapping[str, Any], value)
    missing = sorted(set(required_keys) - set(data))
    unexpected = sorted(set(data) - set(required_keys) - set(optional_keys))
    if missing or unexpected:
        raise TrajectoryConfigError(
            f"{name} keys disagree; missing={missing}, unexpected={unexpected}"
        )
    return data


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryConfigError(f"{name} must be a non-empty string")
    return value


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TrajectoryConfigError(f"{name} must be a sequence of strings")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def _integer(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrajectoryConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, allow_zero=allow_zero)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrajectoryConfigError(f"{name} must be boolean")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryConfigError(f"{name} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise TrajectoryConfigError(f"{name} must be finite")
    if positive and result <= 0:
        raise TrajectoryConfigError(f"{name} must be positive")
    return result


def _optional_positive_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name, positive=True)


def _digest(value: object, name: str, *, length: int = 64) -> str:
    result = _text(value, name).lower()
    if len(result) != length or any(char not in "0123456789abcdef" for char in result):
        raise TrajectoryConfigError(f"{name} must be a {length}-character hex digest")
    return result


def _resolve_maniptrans_component_weights(
    value: object,
) -> ManipTransComponentWeights:
    keys = (
        "wrist_position",
        "wrist_orientation",
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
        "level_1",
        "level_2",
        "wrist_linear_velocity",
        "wrist_angular_velocity",
        "hand_keypoint_velocity",
        "hand_power",
        "wrist_power",
    )
    data = _section_with_optional(
        value,
        "reward.maniptrans.component_weights",
        keys,
        ("fingertip_velocity", "hand_joint_position"),
    )
    values = {
        key: _number(
            data[key],
            f"reward.maniptrans.component_weights.{key}",
        )
        for key in keys
    }
    values["hand_joint_position"] = _number(
        data.get("hand_joint_position", 0.0),
        "reward.maniptrans.component_weights.hand_joint_position",
    )
    values["fingertip_velocity"] = _number(
        data.get("fingertip_velocity", 0.0),
        "reward.maniptrans.component_weights.fingertip_velocity",
    )
    return ManipTransComponentWeights(**values)


def _resolve_maniptrans_reward(value: object) -> ManipTransRewardSpec:
    data = _section_with_optional(
        value,
        "reward.maniptrans",
        (
            "source_profile",
            "tighten_factor",
            "tighten_simulation_steps",
            "failure_warmup_steps",
        ),
        ("component_weights",),
    )
    return ManipTransRewardSpec(
        source_profile=_text(
            data["source_profile"], "reward.maniptrans.source_profile"
        ),  # type: ignore[arg-type]
        tighten_factor=_number(
            data["tighten_factor"], "reward.maniptrans.tighten_factor", positive=True
        ),
        tighten_simulation_steps=_integer(
            data["tighten_simulation_steps"],
            "reward.maniptrans.tighten_simulation_steps",
        ),
        failure_warmup_steps=_integer(
            data["failure_warmup_steps"],
            "reward.maniptrans.failure_warmup_steps",
            allow_zero=True,
        ),
        component_weights=(
            ManipTransComponentWeights()
            if data.get("component_weights") is None
            else _resolve_maniptrans_component_weights(data["component_weights"])
        ),
    )






def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrajectoryDataSpec:
    upstream_root_relative: str
    expected_revision: str
    selection_relative: str
    selection_sha256: str
    selection_count: int
    frame_start: int
    frame_stop: int
    selection_index: int
    expected_episode_uid: str
    dataset_root_env: str
    outcome: str
    source_num_points: int
    trajectory_count: int = 1
    frame_coverage_policy: Literal[
        "selection_manifest", "source_valid_frames"
    ] = "selection_manifest"
    adapter: str = "interprior"
    selection_root: Literal["upstream", "project"] = "upstream"

    def __post_init__(self) -> None:
        root = Path(self.upstream_root_relative)
        selection = Path(self.selection_relative)
        if root.is_absolute() or ".." in root.parts:
            raise TrajectoryConfigError("data.upstream_root must be project-relative")
        if selection.is_absolute() or ".." in selection.parts:
            raise TrajectoryConfigError("data.selection must be upstream-relative")
        _digest(self.expected_revision, "data.expected_revision", length=40)
        _digest(self.selection_sha256, "data.selection_sha256")
        _digest(self.expected_episode_uid, "data.expected_episode_uid")
        if (
            self.selection_count <= 0
            or not 0 <= self.selection_index < self.selection_count
            or self.trajectory_count <= 0
            or self.selection_index + self.trajectory_count > self.selection_count
        ):
            raise TrajectoryConfigError("selection index/count/prefix are inconsistent")
        if self.frame_start < 0 or self.frame_stop <= self.frame_start + 1:
            raise TrajectoryConfigError(
                "data frame window requires 0 <= frame_start < frame_stop - 1"
            )
        if self.frame_coverage_policy not in (
            "selection_manifest",
            "source_valid_frames",
        ):
            raise TrajectoryConfigError("unsupported data.frame_coverage_policy")
        if self.outcome != "success" or self.source_num_points <= 0:
            raise TrajectoryConfigError("unsupported dataset outcome/point count")
        if not self.adapter or self.selection_root not in ("upstream", "project"):
            raise TrajectoryConfigError("invalid data adapter/selection root")

    def upstream_root(self, project_root: Path) -> Path:
        root = (project_root / self.upstream_root_relative).resolve()
        if not root.is_relative_to(project_root.resolve()):
            raise TrajectoryConfigError("upstream root escapes project")
        return root

    def selection_path(self, project_root: Path) -> Path:
        upstream = (
            project_root.resolve()
            if self.selection_root == "project"
            else self.upstream_root(project_root)
        )
        path = (upstream / self.selection_relative).resolve()
        if not path.is_relative_to(upstream):
            raise TrajectoryConfigError("selection path escapes upstream root")
        return path

    def dataset_root(self, environ: Mapping[str, str] | None = None) -> Path:
        environment = os.environ if environ is None else environ
        value = environment.get(self.dataset_root_env)
        if not value:
            raise TrajectoryConfigError(
                f"set {self.dataset_root_env} to the mounted dataset root"
            )
        root = Path(value).expanduser()
        if not root.is_absolute():
            raise TrajectoryConfigError(
                "dataset root environment value must be absolute"
            )
        return root.resolve()

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "upstream_root": self.upstream_root_relative,
            "expected_revision": self.expected_revision,
            "selection": self.selection_relative,
            "selection_sha256": self.selection_sha256,
            "selection_count": self.selection_count,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "trajectory_mode": "single",
            "selection_index": self.selection_index,
            "episode_uid": self.expected_episode_uid,
            "dataset_root_env": self.dataset_root_env,
            "outcome": self.outcome,
            "source_num_points": self.source_num_points,
        }
        if self.trajectory_count > 1:
            payload.update(
                {
                    "trajectory_mode": "contiguous_selection_prefix",
                    "selection_start_index": self.selection_index,
                    "trajectory_count": self.trajectory_count,
                    "first_episode_uid": self.expected_episode_uid,
                }
            )
            payload.pop("selection_index")
            payload.pop("episode_uid")
        if self.frame_coverage_policy != "selection_manifest":
            payload["frame_coverage_policy"] = self.frame_coverage_policy
        if self.adapter != "interprior":
            payload["adapter"] = self.adapter
        if self.selection_root != "upstream":
            payload["selection_root"] = self.selection_root
        return payload


@dataclass(frozen=True, slots=True)
class TimingSpec:
    physics_hz: int
    control_hz: int
    flow_anchor_offsets_steps: tuple[int, ...]
    settle_physics_steps: int

    def __post_init__(self) -> None:
        if (
            self.physics_hz <= 0
            or self.control_hz <= 0
            or self.physics_hz % self.control_hz
        ):
            raise TrajectoryConfigError("physics_hz must be divisible by control_hz")
        if self.physics_hz % TRAJECTORY_REFERENCE_SOURCE_CONTROL_HZ:
            raise TrajectoryConfigError(
                "physics_hz must be divisible by the 60 Hz reference source"
            )
        if (
            self.control_hz not in (30, TRAJECTORY_REFERENCE_SOURCE_CONTROL_HZ)
            or TRAJECTORY_REFERENCE_SOURCE_CONTROL_HZ % self.control_hz
        ):
            raise TrajectoryConfigError(
                "trajectory control_hz must be 60 Hz or the supported 30 Hz "
                "source-downsample ablation"
            )
        offsets = tuple(self.flow_anchor_offsets_steps)
        schedules = {
            60: (
                (0, 30, 60, 90, 120),
                (1, 30, 60, 90, 120),
                (1, 15, 30, 45, 60),
            ),
            30: ((1, 15, 30, 45, 60),),
        }
        if offsets not in schedules[self.control_hz]:
            raise TrajectoryConfigError(
                "flow anchors disagree with the supported control-rate/horizon "
                f"schedules for {self.control_hz} Hz"
            )
        if self.settle_physics_steps != 0:
            raise TrajectoryConfigError("first version forbids hidden settle steps")
        object.__setattr__(self, "flow_anchor_offsets_steps", offsets)

    @property
    def source_control_hz(self) -> int:
        """Sampling rate of the immutable lqr_test source arrays."""

        return TRAJECTORY_REFERENCE_SOURCE_CONTROL_HZ

    @property
    def source_frame_stride(self) -> int:
        """Stride used to keep reference wall time fixed at lower control rates."""

        return self.source_control_hz // self.control_hz

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "physics_hz": self.physics_hz,
            "control_hz": self.control_hz,
            "episode_end": "last_recorded_state",
            "episode_length": "derived_from_reset_start_and_pre_roll",
            "flow_anchor_offsets_steps": list(self.flow_anchor_offsets_steps),
            "settle_physics_steps": self.settle_physics_steps,
        }
        if self.source_frame_stride != 1:
            payload["reference_source_control_hz"] = self.source_control_hz
            payload["reference_source_frame_stride"] = self.source_frame_stride
            payload["reference_wall_time_policy"] = (
                "downsample_source_without_slow_motion"
            )
        if self.flow_anchor_offsets_steps[0] == 0:
            # Preserve historical manifests and checkpoint hashes exactly.
            payload["flow_anchor_spacing_s"] = 0.5
        else:
            payload["flow_anchor_times_s"] = [
                offset / self.control_hz
                for offset in self.flow_anchor_offsets_steps
            ]
            payload["first_anchor_semantic"] = "next_state_reward_target"
        return payload


@dataclass(frozen=True, slots=True)
class StartIndexSpec:
    mode: Literal["fixed", "uniform_valid", "object_lift_phase_weighted"]
    fixed_index: int | None
    minimum: int | None
    maximum: int | None
    maximum_trajectory_fraction: float
    trajectory_assignment: Literal["random_per_reset", "fixed_cyclic"] = (
        "random_per_reset"
    )
    pre_lift_reset_probability: float = 0.6
    lift_height_threshold_m: float = 0.02

    def __post_init__(self) -> None:
        if self.mode not in (
            "fixed",
            "uniform_valid",
            "object_lift_phase_weighted",
        ):
            raise TrajectoryConfigError("unsupported start-index mode")
        for name in ("fixed_index", "minimum", "maximum"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise TrajectoryConfigError(
                    f"start_index.{name} must be an integer >= 0 or null"
                )
        if self.mode == "fixed" and self.fixed_index is None:
            raise TrajectoryConfigError(
                "fixed start-index mode requires fixed_index >= 0"
            )
        if self.mode != "fixed" and self.fixed_index is not None:
            raise TrajectoryConfigError(
                f"{self.mode} mode must set fixed_index: null"
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise TrajectoryConfigError("start-index minimum exceeds maximum")
        if (
            isinstance(self.maximum_trajectory_fraction, bool)
            or not isinstance(self.maximum_trajectory_fraction, (int, float))
            or not isfinite(self.maximum_trajectory_fraction)
            or not 0.0 < self.maximum_trajectory_fraction <= 1.0
        ):
            raise TrajectoryConfigError(
                "start_index.maximum_trajectory_fraction must be in (0,1]"
            )
        if self.trajectory_assignment not in ("random_per_reset", "fixed_cyclic"):
            raise TrajectoryConfigError("unsupported trajectory assignment")
        if (
            isinstance(self.pre_lift_reset_probability, bool)
            or not isinstance(self.pre_lift_reset_probability, (int, float))
            or not isfinite(self.pre_lift_reset_probability)
            or not 0.0 < self.pre_lift_reset_probability < 1.0
        ):
            raise TrajectoryConfigError(
                "start_index.pre_lift_reset_probability must be in (0,1)"
            )
        if (
            isinstance(self.lift_height_threshold_m, bool)
            or not isinstance(self.lift_height_threshold_m, (int, float))
            or not isfinite(self.lift_height_threshold_m)
            or self.lift_height_threshold_m <= 0.0
        ):
            raise TrajectoryConfigError(
                "start_index.lift_height_threshold_m must be positive"
            )

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode,
            "fixed_index": self.fixed_index,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "maximum_trajectory_fraction": self.maximum_trajectory_fraction,
            "parallel_environment_start_sampling": (
                "independent_index_per_environment_reset"
            ),
        }
        if self.trajectory_assignment != "random_per_reset":
            payload["trajectory_assignment"] = self.trajectory_assignment
            payload["parallel_environment_trajectory_sampling"] = (
                "environment_id_mod_trajectory_count"
            )
        if self.mode == "object_lift_phase_weighted":
            payload["phase_boundary"] = {
                "signal": "object_root_height_above_materialized_frame_0",
                "lift_height_threshold_m": self.lift_height_threshold_m,
                "per_trajectory": True,
                "pre_lift_interval": "[legal_min, first_lift_index)",
                "post_lift_interval": "[first_lift_index, legal_max]",
            }
            payload["phase_sampling_probability"] = {
                "pre_lift": self.pre_lift_reset_probability,
                "post_lift": 1.0 - self.pre_lift_reset_probability,
                "within_phase": "uniform",
            }
        return payload


@dataclass(frozen=True, slots=True)
class TrajectoryDirectConfig:
    schema_version: int
    seed: int
    environment_version: str
    data: TrajectoryDataSpec
    timing: TimingSpec
    start_index: StartIndexSpec
    perturbation: RandomActionPerturbationSpec
    direct_action: DirectActionSpec
    observation: TrajectoryObservationSpec
    reward: TrajectoryTrackingRewardSpec
    termination: TrajectoryTrackingTerminationSpec
    ppo: PPOConfig
    experiment_plugin: PPOBenchmarkLineageSpec | None = None
    scene: TrajectorySceneRuntimeSpec = field(
        default_factory=TrajectorySceneRuntimeSpec
    )
    task_success: TrajectoryTaskSuccessSpec = field(
        default_factory=TrajectoryTaskSuccessSpec
    )
    action_smoothness: ActionSmoothnessSpec = field(default_factory=ActionSmoothnessSpec)
    joint_acceleration: JointAccelerationSpec = field(default_factory=JointAccelerationSpec)

    def __post_init__(self) -> None:
        if (
            self.schema_version not in (14, 17, 18)
            or self.seed < 0
            or not self.environment_version
        ):
            raise TrajectoryConfigError(
                "unsupported schema, seed, or environment version"
            )
        if self.direct_action.mode == "arm_target_increment":
            if self.timing.control_hz != 60:
                raise TrajectoryConfigError("arm target increment requires 60 Hz control")
            if not self.observation.include_previous_policy_action:
                raise TrajectoryConfigError("arm target increment requires previous target observation")
        uses_new_contract = (
            self.data.trajectory_count > 1
            or not self.observation.include_reference_robot_state
            or self.observation.include_previous_policy_action
            or self.observation.include_current_link7_pose
            or self.observation.object_target_representation
            != "legacy_reference_flow"
            or self.observation.reference_robot_target_representation != "none"
            or bool(self.observation.current_reference_robot_target_features)
            or self.observation.robot_joint_temporal_context
            or self.reward.object_flow_reward_weight > 0.0
            or not self.reward.robot_tracking_reward_enabled
            or self.reward.joint_velocity_penalty_weight > 0.0
            or self.start_index.mode == "object_lift_phase_weighted"
            or self.termination.object_departure_position_error_threshold_m is not None
            or self.termination.robot_tracking_reset_mode != "maniptrans_keypoints"
            or self.task_success.mode != "robot_endpoint"
            or self.action_smoothness.enabled
            or self.joint_acceleration.enabled
            or self.experiment_plugin is not None
            or self.scene.object_physics_mode != "dynamic"
        )
        if self.schema_version == 14 and uses_new_contract:
            raise TrajectoryConfigError(
                "multi-trajectory and pluggable reward/outcome features require schema 17"
            )
        if self.experiment_plugin is not None:
            if self.ppo.learner_profile != "benchmark":
                raise TrajectoryConfigError("alternate learner must clear benchmark experiment_plugin lineage")
            if self.schema_version != 18:
                raise TrajectoryConfigError(
                    "the frozen PPO benchmark lineage requires trajectory schema 18"
                )
            self._validate_ppo_benchmark_lineage()
        if (
            self.observation.flow_anchor_offsets_steps
            != self.timing.flow_anchor_offsets_steps
        ):
            raise TrajectoryConfigError("observation and timing flow anchors disagree")
        materialized_frame_count = self.materialized_frame_count
        legal_max = min(
            materialized_frame_count
            - 2
            - self.perturbation.pre_roll_policy_steps,
            floor(
                materialized_frame_count
                * self.start_index.maximum_trajectory_fraction
            ),
        )
        legal_min = 0 if (
            self.start_index.minimum == 0 or self.start_index.fixed_index == 0
        ) else 1
        if legal_max < legal_min:
            raise TrajectoryConfigError(
                "selected data prefix cannot fit pre-roll plus one PPO step"
            )
        if (
            self.start_index.fixed_index is not None
            and self.start_index.fixed_index > legal_max
        ):
            raise TrajectoryConfigError(
                "fixed start index violates the trajectory-end episode contract"
            )
        source_reward_scale = (
            self.reward.learner_scale_weight_sum / self.reward.maximum_reward
        )
        if not isclose(
            self.ppo.learner_reward_scale,
            source_reward_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise TrajectoryConfigError(
                "learner_reward_scale must recover the frozen base profile's "
                "raw scale from the public 0--100 score"
            )
        if self.ppo.learner_failure_penalty != 0.0:
            raise TrajectoryConfigError(
                "trajectory reward profiles must not add a terminal failure penalty"
            )

    def _validate_ppo_benchmark_lineage(self) -> None:
        assert self.experiment_plugin is not None
        observation = self.observation
        expected = {
            "rollout_steps_per_env": 32,
            "update_epochs": 5,
            "num_minibatches": 52,
            "actor_learning_rate": 2.0e-4,
            "critic_learning_rate": 2.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "value_loss_coef": 4.0,
            "entropy_coef": 0.0,
            "max_grad_norm": 1.0,
            "target_kl": 0.008,
            "actor_learning_rate_min": 1.0e-6,
            "actor_learning_rate_max": 1.0e-2,
            "actor_learning_rate_adaptation_factor": 1.5,
            "actor_lr_clip_fraction_threshold": 1.0,
            "arm_initial_std": TRAJECTORY_PPO_BENCHMARK_INITIAL_STD,
            "hand_initial_std": TRAJECTORY_PPO_BENCHMARK_INITIAL_STD,
        }
        mismatches = {
            name: (target, getattr(self.ppo, name))
            for name, target in expected.items()
            if not isclose(
                float(getattr(self.ppo, name)),
                float(target),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        }
        if (
            mismatches
            or self.ppo.actor_lr_schedule != "adaptive_kl"
            or not self.ppo.clip_value_loss
            or self.ppo.hidden_dims != TRAJECTORY_PPO_BENCHMARK_HIDDEN_DIMS
            or self.ppo.activation != "elu"
        ):
            raise TrajectoryConfigError(
                "promoted PPO benchmark settings disagree: "
                f"numeric={mismatches}"
            )
        if self.scene.object_physics_mode != "dynamic":
            raise TrajectoryConfigError(
                "the promoted benchmark requires dynamic object physics"
            )

    @property
    def materialized_frame_count(self) -> int:
        """Number of policy-rate frames produced by the source frame window."""

        return len(
            range(
                self.data.frame_start,
                self.data.frame_stop,
                self.timing.source_frame_stride,
            )
        )

    @property
    def uses_task_outcome_tracking(self) -> bool:
        return (
            self.task_success.mode in ("robot_object_trajectory", "object_trajectory")
            or self.termination.object_departure_position_error_threshold_m is not None
        )

    @property
    def compatibility_hash(self) -> str:
        encoded = json.dumps(
            self.to_manifest(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, object]:
        termination_manifest = self.termination.to_manifest()
        legacy = self.schema_version == 14
        if legacy:
            # Keep the historical 09/11 manifest shape so optional plugins do
            # not invalidate their checkpoint lineage.
            termination_manifest["active_for_reward_profile"] = False
        object_departure_reset = (
            self.termination.object_departure_reset_enabled
            and self.termination.object_departure_position_error_threshold_m is not None
        )
        episode_causes = ["reference_endpoint"]
        failure_precedence: list[str] = []
        if object_departure_reset:
            episode_causes.append("object_departure_position_error")
            failure_precedence.append("object_departure_position_error")
        if self.termination.robot_tracking_reset_mode == "link7_weighted_pose":
            robot_failure_cause = (
                "link7_weighted_pose_tracking_error_or_numerical_sanity"
            )
        else:
            robot_failure_cause = "maniptrans_keypoint_tracking_error"
        episode_causes.append(robot_failure_cause)
        failure_precedence.extend([robot_failure_cause, "reference_endpoint_result"])
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "task": "trajectory_residual",
            "environment_version": self.environment_version,
            "seed": self.seed,
            "data": self.data.to_manifest(),
            "timing": self.timing.to_manifest(),
            "start_index": self.start_index.to_manifest(),
            "perturbation": self.perturbation.to_manifest(),
            "direct_action": self.direct_action.to_manifest(),
            "recorded_action_usage": {
                "pre_training_nominal_replay_gate": True,
                "ppo_observation": False,
                "ppo_reward": False,
                "ppo_action_composition": False,
                "reset": self.perturbation.mode == "random_action",
            },
            "observation": self.observation.to_manifest(),
            "reward": self.reward.to_manifest(),
            "termination": termination_manifest,
            "episode_end": {
                "causes": episode_causes,
                "error_triggered_reset": True,
                "failure_precedence": failure_precedence,
            },
            "ppo": self.ppo.to_manifest(),
        }
        if not legacy:
            payload["task_success"] = self.task_success.to_manifest()
        if self.direct_action.mode == "arm_target_increment":
            payload["observation"]["previous_policy_action"] = {
                "semantic": "previous_executed_target_normalized_by_absolute_range",
                "reset": "normalized_actual_reset_joint_target",
                "source": "rollout.previous_joint_target",
            }
        if self.action_smoothness.enabled:
            payload["action_smoothness"] = self.action_smoothness.to_manifest()
        if self.joint_acceleration.enabled:
            payload["joint_acceleration"] = self.joint_acceleration.to_manifest()
        if self.experiment_plugin is not None:
            experiment_lineage = self.experiment_plugin.to_manifest()
            current_reference_features = (
                self.observation.current_reference_robot_target_features
            )
            if current_reference_features:
                observation_lock = dict(
                    cast(
                        Mapping[str, object],
                        experiment_lineage["observation_lock"],
                    )
                )
                observation_lock.update(
                    {
                        "semantic": (
                            "experiment_56_observation_D_base_plus_explicit_"
                            "current_reference_robot_target_extension"
                        ),
                        "base_feature_dim": 184,
                        "explicit_target_extension_added": True,
                        "current_reference_robot_target_features": list(
                            current_reference_features
                        ),
                    }
                )
                experiment_lineage["observation_lock"] = observation_lock
            payload["experiment_plugin"] = experiment_lineage
            payload["scene_runtime"] = self.scene.to_manifest()
        if include_hash:
            payload["compatibility_hash"] = self.compatibility_hash
        if self.ppo.learner_profile != "benchmark":
            payload["scene_runtime"] = self.scene.to_manifest()
        return payload




def _resolve_joint_acceleration(raw: object) -> JointAccelerationSpec:
    if raw is None:
        return JointAccelerationSpec()
    values = _section_with_optional(raw, "joint_acceleration", ("enabled",),
        ("coefficient", "arm_scale_rad_s2", "hand_scale_rad_s2",
         "calibration_required", "calibration_sha256"))
    defaults = JointAccelerationSpec()
    digest = values.get("calibration_sha256", "")
    if not isinstance(digest, str):
        raise TrajectoryConfigError("calibration_sha256 must be text")
    return JointAccelerationSpec(
        enabled=_boolean(values["enabled"], "joint_acceleration.enabled"),
        calibration_required=_boolean(values.get("calibration_required", False),
                                      "joint_acceleration.calibration_required"),
        calibration_sha256=digest,
        **{name: _number(values.get(name, getattr(defaults, name)), f"joint_acceleration.{name}")
           for name in ("coefficient", "arm_scale_rad_s2", "hand_scale_rad_s2")},
    )


def _resolve_action_smoothness(raw: object) -> ActionSmoothnessSpec:
    if raw is None:
        return ActionSmoothnessSpec()
    values = _section_with_optional(
        raw, "action_smoothness", ("enabled",),
        ("first_difference_weight", "second_difference_weight", "arm_weight", "hand_weight"),
    )
    defaults = ActionSmoothnessSpec()
    return ActionSmoothnessSpec(
        enabled=_boolean(values["enabled"], "action_smoothness.enabled"),
        **{name: _number(values.get(name, getattr(defaults, name)), f"action_smoothness.{name}")
           for name in ("first_difference_weight", "second_difference_weight", "arm_weight", "hand_weight")},
    )


def resolve_trajectory_direct_config(raw: object) -> TrajectoryDirectConfig:
    # Neutral public key; preserve the serialized learner marker and its math.
    if isinstance(raw, Mapping) and "learner_lineage" in raw:
        if "experiment_plugin" in raw:
            raise TrajectoryConfigError("learner_lineage and legacy experiment_plugin are mutually exclusive")
        raw = dict(raw)
        raw["experiment_plugin"] = raw.pop("learner_lineage")
    root = _section_with_optional(
        raw,
        "trajectory config",
        (
            "schema_version",
            "run",
            "data",
            "timing",
            "start_index",
            "perturbation",
            "direct_action",
            "observation",
            "reward",
            "termination",
            "ppo",
        ),
        ("task_success", "experiment_plugin", "scene", "action_smoothness", "joint_acceleration"),
    )
    run = _section(root["run"], "run", ("seed", "environment_version"))
    data = _section_with_optional(
        root["data"],
        "data",
        (
            "upstream_root",
            "expected_revision",
            "selection",
            "selection_sha256",
            "selection_count",
            "frame_start",
            "frame_stop",
            "selection_index",
            "expected_episode_uid",
            "dataset_root_env",
            "outcome",
            "source_num_points",
        ),
        ("trajectory_count", "frame_coverage_policy", "adapter", "selection_root"),
    )
    timing = _section(
        root["timing"],
        "timing",
        (
            "physics_hz",
            "control_hz",
            "flow_anchor_offsets_steps",
            "settle_physics_steps",
        ),
    )
    start = _section_with_optional(
        root["start_index"],
        "start_index",
        (
            "mode",
            "fixed_index",
            "minimum",
            "maximum",
            "maximum_trajectory_fraction",
        ),
        (
            "trajectory_assignment",
            "pre_lift_reset_probability",
            "lift_height_threshold_m",
        ),
    )
    perturbation = _section(
        root["perturbation"],
        "perturbation",
        (
            "mode",
            "pre_roll_policy_steps",
            "distribution",
            "active_joint_groups",
            "arm_delta_scale_rad",
            "hand_delta_scale_rad",
            "joint_limit_margin_rad",
        ),
    )
    direct_action = _section_with_optional(
        root["direct_action"],
        "direct_action",
        ("joint_limit_margin_rad",),
        ("mode", "arm_target_speed_rad_s"),
    )
    observation = _section_with_optional(
        root["observation"],
        "observation",
        (
            "num_object_points",
            "joint_velocity_scale_rad_s",
            "object_point_scale_m",
            "object_linear_velocity_scale_m_s",
            "object_angular_velocity_scale_rad_s",
            "fingertip_count",
        ),
        (
            "object_target_representation",
            "include_reference_robot_state",
            "include_previous_policy_action",
            "include_current_link7_pose",
            "link7_position_scale_m",
            "reference_robot_target_representation",
            "current_reference_robot_target_features",
            "robot_joint_temporal_context",
        ),
    )
    reward = _section_with_optional(
        root["reward"],
        "reward",
        (
            "combination",
            "maximum_reward",
            "object_flow_diagnostic_half_error_m",
            "arm_joint_position_half_reward_error_rad",
            "hand_joint_position_half_reward_error_rad",
            "maniptrans",
        ),
        (
            "object_flow_reward_weight",
            "joint_velocity_penalty_weight",
            "arm_joint_velocity_penalty_scale_rad_s",
            "hand_joint_velocity_penalty_scale_rad_s",
            "robot_tracking_reward_enabled",
        ),
    )
    termination = _section_with_optional(
        root["termination"],
        "termination",
        (
            "mean_absolute_joint_position_error_threshold_rad",
            "link7_position_error_threshold_m",
        ),
        (
            "object_departure_position_error_threshold_m",
            "object_departure_failure_warmup_steps",
            "object_departure_reset_enabled",
            "robot_tracking_reset_mode",
            "link7_weighted_pose_error_threshold_m",
            "link7_orientation_error_weight_m_per_rad",
            "link7_pose_failure_warmup_steps",
        ),
    )
    task_success_raw = root.get("task_success")
    task_success = (
        None
        if task_success_raw is None
        else _section(
            task_success_raw,
            "task_success",
            (
                "mode",
                "episode_mean_flow_error_threshold_m",
                "final_flow_error_threshold_m",
            ),
        )
    )
    experiment_plugin_raw = root.get("experiment_plugin")
    experiment_plugin = (
        None
        if experiment_plugin_raw is None
        else _section_with_optional(
            experiment_plugin_raw,
            "experiment_plugin",
            ("name", "learner_enabled", "task_runtime_enabled"),
            ("evaluation_mode",),
        )
    )
    scene_raw = root.get("scene")
    scene = (
        None
        if scene_raw is None
        else _section_with_optional(
            scene_raw,
            "scene",
            ("object_physics_mode",),
            ("contact_sensors_enabled", "table_collision"),
        )
    )
    ppo = _section_with_optional(
        root["ppo"],
        "ppo",
        (
            "rollout_steps_per_env",
            "update_epochs",
            "num_minibatches",
            "actor_learning_rate",
            "critic_learning_rate",
            "learner_reward_scale",
            "learner_failure_penalty",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "clip_value_loss",
            "value_loss_coef",
            "entropy_coef",
            "max_grad_norm",
            "target_kl",
            "actor_lr_schedule",
            "actor_learning_rate_min",
            "actor_learning_rate_max",
            "actor_learning_rate_adaptation_factor",
            "actor_lr_clip_fraction_threshold",
            "arm_initial_std",
            "hand_initial_std",
            "hidden_dims",
            "activation",
        ),
        ("learner_profile",),
    )

    offsets = tuple(
        _integer(value, f"timing.flow_anchor_offsets_steps[{index}]", allow_zero=True)
        for index, value in enumerate(timing["flow_anchor_offsets_steps"])
    )
    active_groups = tuple(perturbation["active_joint_groups"])
    hidden_dims = tuple(
        _integer(value, f"ppo.hidden_dims[{index}]")
        for index, value in enumerate(ppo["hidden_dims"])
    )
    return TrajectoryDirectConfig(
        schema_version=_integer(root["schema_version"], "schema_version"),
        seed=_integer(run["seed"], "run.seed", allow_zero=True),
        environment_version=_text(
            run["environment_version"], "run.environment_version"
        ),
        data=TrajectoryDataSpec(
            upstream_root_relative=_text(data["upstream_root"], "data.upstream_root"),
            expected_revision=_digest(
                data["expected_revision"], "data.expected_revision", length=40
            ),
            selection_relative=_text(data["selection"], "data.selection"),
            selection_sha256=_digest(data["selection_sha256"], "data.selection_sha256"),
            selection_count=_integer(data["selection_count"], "data.selection_count"),
            frame_start=_integer(
                data["frame_start"], "data.frame_start", allow_zero=True
            ),
            frame_stop=_integer(data["frame_stop"], "data.frame_stop"),
            selection_index=_integer(
                data["selection_index"], "data.selection_index", allow_zero=True
            ),
            expected_episode_uid=_digest(
                data["expected_episode_uid"], "data.expected_episode_uid"
            ),
            dataset_root_env=_text(data["dataset_root_env"], "data.dataset_root_env"),
            outcome=_text(data["outcome"], "data.outcome"),
            source_num_points=_integer(
                data["source_num_points"], "data.source_num_points"
            ),
            trajectory_count=_integer(
                data.get("trajectory_count", 1), "data.trajectory_count"
            ),
            adapter=_text(data.get("adapter", "interprior"), "data.adapter"),
            selection_root=_text(
                data.get("selection_root", "upstream"), "data.selection_root"
            ),
            frame_coverage_policy=_text(
                data.get("frame_coverage_policy", "selection_manifest"),
                "data.frame_coverage_policy",
            ),  # type: ignore[arg-type]
        ),
        timing=TimingSpec(
            physics_hz=_integer(timing["physics_hz"], "timing.physics_hz"),
            control_hz=_integer(timing["control_hz"], "timing.control_hz"),
            flow_anchor_offsets_steps=offsets,
            settle_physics_steps=_integer(
                timing["settle_physics_steps"],
                "timing.settle_physics_steps",
                allow_zero=True,
            ),
        ),
        start_index=StartIndexSpec(
            mode=_text(start["mode"], "start_index.mode"),  # type: ignore[arg-type]
            fixed_index=_optional_integer(
                start["fixed_index"],
                "start_index.fixed_index",
                allow_zero=True,
            ),
            minimum=_optional_integer(
                start["minimum"], "start_index.minimum", allow_zero=True
            ),
            maximum=_optional_integer(
                start["maximum"], "start_index.maximum", allow_zero=True
            ),
            maximum_trajectory_fraction=_number(
                start["maximum_trajectory_fraction"],
                "start_index.maximum_trajectory_fraction",
                positive=True,
            ),
            trajectory_assignment=_text(
                start.get("trajectory_assignment", "random_per_reset"),
                "start_index.trajectory_assignment",
            ),  # type: ignore[arg-type]
            pre_lift_reset_probability=_number(
                start.get("pre_lift_reset_probability", 0.6),
                "start_index.pre_lift_reset_probability",
                positive=True,
            ),
            lift_height_threshold_m=_number(
                start.get("lift_height_threshold_m", 0.02),
                "start_index.lift_height_threshold_m",
                positive=True,
            ),
        ),
        perturbation=RandomActionPerturbationSpec(
            mode=_text(
                perturbation["mode"], "perturbation.mode"
            ),  # type: ignore[arg-type]
            pre_roll_policy_steps=_integer(
                perturbation["pre_roll_policy_steps"],
                "perturbation.pre_roll_policy_steps",
                allow_zero=True,
            ),
            distribution=_text(
                perturbation["distribution"], "perturbation.distribution"
            ),  # type: ignore[arg-type]
            active_joint_groups=active_groups,  # type: ignore[arg-type]
            arm_delta_scale_rad=_number(
                perturbation["arm_delta_scale_rad"],
                "perturbation.arm_delta_scale_rad",
                positive=True,
            ),
            hand_delta_scale_rad=_number(
                perturbation["hand_delta_scale_rad"],
                "perturbation.hand_delta_scale_rad",
                positive=True,
            ),
            joint_limit_margin_rad=_number(
                perturbation["joint_limit_margin_rad"],
                "perturbation.joint_limit_margin_rad",
            ),
        ),
        direct_action=DirectActionSpec(
            mode=direct_action.get("mode", "absolute"),
            arm_target_speed_rad_s=(None if "arm_target_speed_rad_s" not in direct_action else
                _number(direct_action["arm_target_speed_rad_s"], "direct_action.arm_target_speed_rad_s", positive=True)),
            joint_limit_margin_rad=_number(
                direct_action["joint_limit_margin_rad"],
                "direct_action.joint_limit_margin_rad",
            ),
        ),
        observation=TrajectoryObservationSpec(
            num_object_points=_integer(
                observation["num_object_points"], "observation.num_object_points"
            ),
            flow_anchor_offsets_steps=offsets,
            joint_velocity_scale_rad_s=_number(
                observation["joint_velocity_scale_rad_s"],
                "observation.joint_velocity_scale_rad_s",
                positive=True,
            ),
            object_point_scale_m=_number(
                observation["object_point_scale_m"],
                "observation.object_point_scale_m",
                positive=True,
            ),
            object_linear_velocity_scale_m_s=_number(
                observation["object_linear_velocity_scale_m_s"],
                "observation.object_linear_velocity_scale_m_s",
                positive=True,
            ),
            object_angular_velocity_scale_rad_s=_number(
                observation["object_angular_velocity_scale_rad_s"],
                "observation.object_angular_velocity_scale_rad_s",
                positive=True,
            ),
            fingertip_count=_integer(
                observation["fingertip_count"], "observation.fingertip_count"
            ),
            include_reference_robot_state=_boolean(
                observation.get("include_reference_robot_state", True),
                "observation.include_reference_robot_state",
            ),
            include_previous_policy_action=_boolean(
                observation.get("include_previous_policy_action", False),
                "observation.include_previous_policy_action",
            ),
            include_current_link7_pose=_boolean(
                observation.get("include_current_link7_pose", False),
                "observation.include_current_link7_pose",
            ),
            link7_position_scale_m=_number(
                observation.get("link7_position_scale_m", 1.0),
                "observation.link7_position_scale_m",
                positive=True,
            ),
            object_target_representation=_text(
                observation.get(
                    "object_target_representation", "legacy_reference_flow"
                ),
                "observation.object_target_representation",
            ),  # type: ignore[arg-type]
            reference_robot_target_representation=_text(
                observation.get("reference_robot_target_representation", "none"),
                "observation.reference_robot_target_representation",
            ),  # type: ignore[arg-type]
            current_reference_robot_target_features=_text_tuple(
                observation.get("current_reference_robot_target_features", ()),
                "observation.current_reference_robot_target_features",
            ),  # type: ignore[arg-type]
            robot_joint_temporal_context=_boolean(
                observation.get("robot_joint_temporal_context", False),
                "observation.robot_joint_temporal_context",
            ),
        ),
        reward=TrajectoryTrackingRewardSpec(
            combination=_text(
                reward["combination"], "reward.combination"
            ),  # type: ignore[arg-type]
            maximum_reward=_number(
                reward["maximum_reward"],
                "reward.maximum_reward",
                positive=True,
            ),
            object_flow_diagnostic_half_error_m=_number(
                reward["object_flow_diagnostic_half_error_m"],
                "reward.object_flow_diagnostic_half_error_m",
                positive=True,
            ),
            arm_joint_position_half_reward_error_rad=_number(
                reward["arm_joint_position_half_reward_error_rad"],
                "reward.arm_joint_position_half_reward_error_rad",
                positive=True,
            ),
            hand_joint_position_half_reward_error_rad=_number(
                reward["hand_joint_position_half_reward_error_rad"],
                "reward.hand_joint_position_half_reward_error_rad",
                positive=True,
            ),
            maniptrans=_resolve_maniptrans_reward(reward["maniptrans"]),
            object_flow_reward_weight=_number(
                reward.get("object_flow_reward_weight", 0.0),
                "reward.object_flow_reward_weight",
            ),
            robot_tracking_reward_enabled=_boolean(
                reward.get("robot_tracking_reward_enabled", True),
                "reward.robot_tracking_reward_enabled",
            ),
            joint_velocity_penalty_weight=_number(
                reward.get("joint_velocity_penalty_weight", 0.0),
                "reward.joint_velocity_penalty_weight",
            ),
            arm_joint_velocity_penalty_scale_rad_s=_number(
                reward.get("arm_joint_velocity_penalty_scale_rad_s", 4.0),
                "reward.arm_joint_velocity_penalty_scale_rad_s",
                positive=True,
            ),
            hand_joint_velocity_penalty_scale_rad_s=_number(
                reward.get("hand_joint_velocity_penalty_scale_rad_s", 4.0),
                "reward.hand_joint_velocity_penalty_scale_rad_s",
                positive=True,
            ),
        ),
        termination=TrajectoryTrackingTerminationSpec(
            mean_absolute_joint_position_error_threshold_rad=_number(
                termination["mean_absolute_joint_position_error_threshold_rad"],
                "termination.mean_absolute_joint_position_error_threshold_rad",
                positive=True,
            ),
            link7_position_error_threshold_m=_number(
                termination["link7_position_error_threshold_m"],
                "termination.link7_position_error_threshold_m",
                positive=True,
            ),
            object_departure_position_error_threshold_m=(
                None
                if termination.get("object_departure_position_error_threshold_m")
                is None
                else _number(
                    termination["object_departure_position_error_threshold_m"],
                    "termination.object_departure_position_error_threshold_m",
                    positive=True,
                )
            ),
            object_departure_failure_warmup_steps=_integer(
                termination.get("object_departure_failure_warmup_steps", 0),
                "termination.object_departure_failure_warmup_steps",
                allow_zero=True,
            ),
            object_departure_reset_enabled=_boolean(
                termination.get("object_departure_reset_enabled", False),
                "termination.object_departure_reset_enabled",
            ),
            robot_tracking_reset_mode=_text(
                termination.get(
                    "robot_tracking_reset_mode", "maniptrans_keypoints"
                ),
                "termination.robot_tracking_reset_mode",
            ),  # type: ignore[arg-type]
            link7_weighted_pose_error_threshold_m=(
                None
                if termination.get("link7_weighted_pose_error_threshold_m") is None
                else _number(
                    termination["link7_weighted_pose_error_threshold_m"],
                    "termination.link7_weighted_pose_error_threshold_m",
                    positive=True,
                )
            ),
            link7_orientation_error_weight_m_per_rad=_number(
                termination.get(
                    "link7_orientation_error_weight_m_per_rad", 0.0
                ),
                "termination.link7_orientation_error_weight_m_per_rad",
            ),
            link7_pose_failure_warmup_steps=_integer(
                termination.get("link7_pose_failure_warmup_steps", 20),
                "termination.link7_pose_failure_warmup_steps",
                allow_zero=True,
            ),
        ),
        ppo=PPOConfig(
            learner_profile=_text(ppo.get("learner_profile", "benchmark"), "ppo.learner_profile"),
            rollout_steps_per_env=_integer(
                ppo["rollout_steps_per_env"],
                "ppo.rollout_steps_per_env",
            ),
            update_epochs=_integer(ppo["update_epochs"], "ppo.update_epochs"),
            num_minibatches=_integer(ppo["num_minibatches"], "ppo.num_minibatches"),
            actor_learning_rate=_number(
                ppo["actor_learning_rate"], "ppo.actor_learning_rate", positive=True
            ),
            critic_learning_rate=_number(
                ppo["critic_learning_rate"], "ppo.critic_learning_rate", positive=True
            ),
            learner_reward_scale=_number(
                ppo["learner_reward_scale"],
                "ppo.learner_reward_scale",
                positive=True,
            ),
            learner_failure_penalty=_number(
                ppo["learner_failure_penalty"],
                "ppo.learner_failure_penalty",
            ),
            gamma=_number(ppo["gamma"], "ppo.gamma"),
            gae_lambda=_number(ppo["gae_lambda"], "ppo.gae_lambda", positive=True),
            clip_coef=_number(ppo["clip_coef"], "ppo.clip_coef", positive=True),
            clip_value_loss=ppo["clip_value_loss"],
            value_loss_coef=_number(
                ppo["value_loss_coef"], "ppo.value_loss_coef", positive=True
            ),
            entropy_coef=_number(ppo["entropy_coef"], "ppo.entropy_coef"),
            max_grad_norm=_number(
                ppo["max_grad_norm"], "ppo.max_grad_norm", positive=True
            ),
            target_kl=_number(ppo["target_kl"], "ppo.target_kl", positive=True),
            actor_lr_schedule=_text(
                ppo["actor_lr_schedule"], "ppo.actor_lr_schedule"
            ),  # type: ignore[arg-type]
            actor_learning_rate_min=_number(
                ppo["actor_learning_rate_min"],
                "ppo.actor_learning_rate_min",
                positive=True,
            ),
            actor_learning_rate_max=_number(
                ppo["actor_learning_rate_max"],
                "ppo.actor_learning_rate_max",
                positive=True,
            ),
            actor_learning_rate_adaptation_factor=_number(
                ppo["actor_learning_rate_adaptation_factor"],
                "ppo.actor_learning_rate_adaptation_factor",
                positive=True,
            ),
            actor_lr_clip_fraction_threshold=_number(
                ppo["actor_lr_clip_fraction_threshold"],
                "ppo.actor_lr_clip_fraction_threshold",
                positive=True,
            ),
            arm_initial_std=_number(
                ppo["arm_initial_std"], "ppo.arm_initial_std", positive=True
            ),
            hand_initial_std=_number(
                ppo["hand_initial_std"], "ppo.hand_initial_std", positive=True
            ),
            hidden_dims=hidden_dims,
            activation=_text(ppo["activation"], "ppo.activation"),  # type: ignore[arg-type]
        ),
        experiment_plugin=(
            None
            if experiment_plugin is None
            else PPOBenchmarkLineageSpec(
                name=_text(
                    experiment_plugin["name"], "experiment_plugin.name"
                ),  # type: ignore[arg-type]
                learner_enabled=_boolean(
                    experiment_plugin["learner_enabled"],
                    "experiment_plugin.learner_enabled",
                ),
                task_runtime_enabled=_boolean(
                    experiment_plugin["task_runtime_enabled"],
                    "experiment_plugin.task_runtime_enabled",
                ),
                evaluation_mode=_text(
                    experiment_plugin.get(
                        "evaluation_mode", "configured_task"
                    ),
                    "experiment_plugin.evaluation_mode",
                ),  # type: ignore[arg-type]
            )
        ),
        scene=(
            TrajectorySceneRuntimeSpec()
            if scene is None
            else TrajectorySceneRuntimeSpec(
                object_physics_mode=_text(
                    scene["object_physics_mode"], "scene.object_physics_mode"
                ),  # type: ignore[arg-type]
                table_collision=_text(
                    scene.get("table_collision", "source"), "scene.table_collision"
                ),  # type: ignore[arg-type]
                contact_sensors_enabled=_boolean(
                    scene.get("contact_sensors_enabled", False),
                    "scene.contact_sensors_enabled",
                ),
            )
        ),
        action_smoothness=_resolve_action_smoothness(root.get("action_smoothness")),
        joint_acceleration=_resolve_joint_acceleration(root.get("joint_acceleration")),
        task_success=(
            TrajectoryTaskSuccessSpec()
            if task_success is None
            else TrajectoryTaskSuccessSpec(
                mode=_text(task_success["mode"], "task_success.mode"),  # type: ignore[arg-type]
                episode_mean_flow_error_threshold_m=_optional_positive_number(
                    task_success["episode_mean_flow_error_threshold_m"],
                    "task_success.episode_mean_flow_error_threshold_m",
                ),
                final_flow_error_threshold_m=_optional_positive_number(
                    task_success["final_flow_error_threshold_m"],
                    "task_success.final_flow_error_threshold_m",
                ),
            )
        ),
    )


def _validate_selection(config: TrajectoryDirectConfig, project_root: Path) -> None:
    path = config.data.selection_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != config.data.selection_sha256:
        raise TrajectoryConfigError(
            f"selection SHA-256 mismatch: expected={config.data.selection_sha256}, actual={actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TrajectoryConfigError("selection metadata disagrees with task config")
    selection_frame_start = payload.get("frame_start")
    selection_frame_stop = payload.get("frame_stop")
    if (
        payload.get("schema_version") != 1
        or payload.get("count") != config.data.selection_count
        or isinstance(selection_frame_start, bool)
        or not isinstance(selection_frame_start, int)
        or isinstance(selection_frame_stop, bool)
        or not isinstance(selection_frame_stop, int)
        or selection_frame_start < 0
        or selection_frame_stop <= selection_frame_start
        or selection_frame_start > config.data.frame_start
        or (
            selection_frame_stop < config.data.frame_stop
            and config.data.frame_coverage_policy != "source_valid_frames"
        )
    ):
        raise TrajectoryConfigError("selection metadata disagrees with task config")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != config.data.selection_count:
        raise TrajectoryConfigError("selection episode count is invalid")
    selected = episodes[config.data.selection_index]
    if (
        not isinstance(selected, Mapping)
        or selected.get("episode_uid") != config.data.expected_episode_uid
    ):
        raise TrajectoryConfigError("first selected trajectory identity changed")


def load_trajectory_direct_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> TrajectoryDirectConfig:
    project = Path(project_root).resolve()
    config_path = Path(path).resolve()
    if not config_path.is_relative_to(project):
        raise TrajectoryConfigError("trajectory config must stay inside the project")

    def deep_merge(base: object, overlay: object) -> object:
        if not isinstance(base, Mapping) or not isinstance(overlay, Mapping):
            return overlay
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else value
        return merged

    def load_one(current: Path, stack: tuple[Path, ...]) -> object:
        current = current.resolve()
        if not current.is_relative_to(project):
            raise TrajectoryConfigError("extends path escapes the project")
        if current in stack:
            chain = " -> ".join(str(item) for item in (*stack, current))
            raise TrajectoryConfigError(f"cyclic trajectory config extends: {chain}")
        try:
            raw = yaml.load(
                current.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
            )
        except yaml.YAMLError as error:
            raise TrajectoryConfigError(f"invalid YAML: {current}") from error
        if not isinstance(raw, Mapping):
            raise TrajectoryConfigError(
                f"trajectory config must be a mapping: {current}"
            )
        parent_value = raw.get("extends")
        overlay = {key: value for key, value in raw.items() if key != "extends"}
        if parent_value is None:
            return overlay
        if not isinstance(parent_value, str) or not parent_value.strip():
            raise TrajectoryConfigError("extends must be one non-empty relative path")
        parent_path = Path(parent_value)
        if parent_path.is_absolute() or ".." in parent_path.parts:
            raise TrajectoryConfigError("extends must be a safe relative path")
        base = load_one(current.parent / parent_path, (*stack, current))
        return deep_merge(base, overlay)

    raw = load_one(config_path, ())
    config = resolve_trajectory_direct_config(raw)
    _validate_selection(config, project)
    return config


__all__ = [
    "StartIndexSpec",
    "TimingSpec",
    "TrajectoryConfigError",
    "TrajectoryDataSpec",
    "TrajectoryDirectConfig",
    "load_trajectory_direct_config",
    "resolve_trajectory_direct_config",
]
