"""Backend-neutral lifecycle for trajectory-guided zero-base residual PPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from .actions import (
    CanonicalActionSpec,
    DirectActionApplication,
    DirectActionProjector,
    make_ordered_action_spec,
)
from .action_smoothness import ActionSmoothnessDiagnostics, ActionSmoothnessReward
from .joint_acceleration import JointAccelerationDiagnostics, JointAccelerationReward, JointAccelerationSpec
from ..configuration import TrajectoryDirectConfig
from .observations import (
    TrajectoryObservationAssembler,
    TrajectoryObservationInput,
)
from .reference_robot_observation import (
    current_reference_robot_target_requirements,
)
from ..utils.kinematics import (
    WUJI_KEYPOINT_COUNT,
    WUJI_KEYPOINTS_PER_FINGER,
    WUJI_TIP_KEYPOINT_INDICES,
)
from .evaluation_metrics import (
    MotionConditionedTrackingDiagnostics,
    motion_conditioned_tracking_diagnostics,
)
from .perturbations import (
    PerturbationPlan,
    PerturbationReceipt,
    RandomActionPerturbation,
)
from ..data.reference import (
    ReferenceCollection,
    ReferenceResetState,
    ReferenceStartSampler,
    ReferenceTrajectory,
    ReferenceTrajectoryBank,
    ReferenceEpisodeBatch,
    build_reference_episode,
    reference_object_lift_phase_boundaries,
)
from .rewards import (
    ManipTransTrackingDiagnostics,
    TrajectoryTrackingReward,
    trajectory_tracking_reward,
)
from .terminations import (
    JointPositionTracking,
    Link7PositionTracking,
    ObjectDepartureTracking,
    joint_position_tracking_error,
    link7_position_tracking_error,
    link7_weighted_pose_failure,
    object_departure_tracking,
)


@dataclass(frozen=True, slots=True)
class LiveState:
    joint_position: Tensor
    joint_velocity: Tensor
    link7_position_robot_base: Tensor
    object_pose_robot_base: Tensor
    object_linear_velocity_robot_base: Tensor
    object_angular_velocity_robot_base: Tensor
    object_surface_points_robot_base: Tensor
    fingertip_positions_robot_base: Tensor
    link7_orientation_robot_base: Tensor
    link7_linear_velocity_robot_base: Tensor
    link7_angular_velocity_robot_base: Tensor
    hand_keypoint_positions_robot_base: Tensor
    hand_keypoint_linear_velocities_robot_base: Tensor
    joint_effort: Tensor

    def validate(
        self,
        *,
        num_envs: int,
        num_object_points: int,
        fingertip_count: int,
        device: torch.device,
        hand_keypoint_count: int = 20,
    ) -> None:
        expected = {
            "joint_position": (num_envs, 27),
            "joint_velocity": (num_envs, 27),
            "link7_position_robot_base": (num_envs, 3),
            "object_pose_robot_base": (num_envs, 7),
            "object_linear_velocity_robot_base": (num_envs, 3),
            "object_angular_velocity_robot_base": (num_envs, 3),
            "object_surface_points_robot_base": (num_envs, num_object_points, 3),
            "fingertip_positions_robot_base": (num_envs, fingertip_count, 3),
            "link7_orientation_robot_base": (num_envs, 4),
            "link7_linear_velocity_robot_base": (num_envs, 3),
            "link7_angular_velocity_robot_base": (num_envs, 3),
            "hand_keypoint_positions_robot_base": (
                num_envs,
                hand_keypoint_count,
                3,
            ),
            "hand_keypoint_linear_velocities_robot_base": (
                num_envs,
                hand_keypoint_count,
                3,
            ),
            "joint_effort": (num_envs, 27),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != shape
                or value.dtype != torch.float32
                or value.device != device
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"backend {name} must be finite float32 {shape}")


_LIVE_STATE_TENSOR_NAMES = (
    "joint_position",
    "joint_velocity",
    "link7_position_robot_base",
    "object_pose_robot_base",
    "object_linear_velocity_robot_base",
    "object_angular_velocity_robot_base",
    "object_surface_points_robot_base",
    "fingertip_positions_robot_base",
    "link7_orientation_robot_base",
    "link7_linear_velocity_robot_base",
    "link7_angular_velocity_robot_base",
    "hand_keypoint_positions_robot_base",
    "hand_keypoint_linear_velocities_robot_base",
    "joint_effort",
)


ROLLOUT_STATE_SCHEMA_VERSION = 5
TASK_OUTCOME_ROLLOUT_STATE_SCHEMA_VERSION = 6
TEMPORAL_OBSERVATION_ROLLOUT_STATE_SCHEMA_VERSION = 7
SOFT_GRASP_PROGRESS_ROLLOUT_STATE_SCHEMA_VERSION = 8
ACTION_SMOOTHNESS_ROLLOUT_STATE_SCHEMA_VERSION = 9


class TrajectoryPhysicsBackend(Protocol):
    """The only simulator-facing surface required by the task core."""

    def restore(self, state: ReferenceResetState, *, env_ids: Tensor) -> LiveState: ...

    def step(self, canonical_joint_target: Tensor) -> LiveState: ...


@dataclass(frozen=True, slots=True)
class TaskResetResult:
    observation: Tensor
    episode: ReferenceEpisodeBatch
    perturbation: PerturbationReceipt


@dataclass(frozen=True, slots=True)
class TaskStepResult:
    next_observation: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    tracking: TrajectoryTrackingReward
    joint_position_tracking: JointPositionTracking
    link7_position_tracking: Link7PositionTracking
    object_departure_tracking: ObjectDepartureTracking
    motion_conditioned_tracking: MotionConditionedTrackingDiagnostics
    maniptrans_diagnostics: ManipTransTrackingDiagnostics | None
    object_departure_failure: Tensor
    maniptrans_failure: Tensor
    action: DirectActionApplication
    episode_steps: Tensor
    reference_cursors: Tensor
    valid_transition: Tensor
    active_after_step: Tensor
    reached_endpoint: Tensor
    episode_completed: Tensor
    episode_success: Tensor
    episode_object_departed: Tensor
    reset_started: Tensor
    pre_roll_mask: Tensor
    action_smoothness_diagnostics: ActionSmoothnessDiagnostics | None = None
    joint_acceleration_diagnostics: JointAccelerationDiagnostics | None = None


class TrajectoryDirectEnvCore:
    """Per-environment trajectory windows with asynchronous reset and pre-roll."""

    def __init__(
        self,
        *,
        config: TrajectoryDirectConfig,
        trajectory: ReferenceCollection,
        action_spec: CanonicalActionSpec,
        backend: TrajectoryPhysicsBackend,
        num_envs: int,
        episode_completion_mode: Literal[
            "configured", "reference_endpoint_only"
        ] = "configured",
        joint_acceleration_diagnostics: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if trajectory.control_hz != config.timing.control_hz:
            raise ValueError("trajectory and task control frequencies disagree")
        if trajectory.physics_hz != config.timing.physics_hz:
            raise ValueError("trajectory and task physics frequencies disagree")
        if trajectory.joint_names != action_spec.joint_names:
            raise ValueError("trajectory and runtime canonical joint orders disagree")
        if episode_completion_mode not in (
            "configured",
            "reference_endpoint_only",
        ):
            raise ValueError(
                f"unsupported episode completion mode: {episode_completion_mode!r}"
            )
        if trajectory.num_object_points != config.observation.num_object_points:
            raise ValueError(
                "trajectory point subset disagrees with observation config"
            )
        if not trajectory.hand_kinematic_reference_available:
            raise ValueError(
                "ManipTrans reward requires materialized Wuji hand reference kinematics"
            )
        if trajectory.hand_keypoint_count != WUJI_KEYPOINT_COUNT:
            raise ValueError(
                "ManipTrans reward requires the 25-point Wuji hand layout"
            )
        self.config = config
        self.trajectory = trajectory
        self.action_spec = action_spec
        self.backend = backend
        self.num_envs = num_envs
        self.device = trajectory.device
        self._object_assignment = None
        if getattr(trajectory, "object_uids", None) is not None:
            from ..data.object_assignment import ObjectTrajectoryAssignment

            environment_uids = getattr(getattr(backend, "scene", None), "object_uids", ())
            if len(environment_uids) != num_envs:
                raise ValueError("multi-object reference requires an explicit scene asset layout")
            self._object_assignment = ObjectTrajectoryAssignment(
                trajectory.object_uids, environment_uids, device=self.device,
            )
        acceleration_spec = config.joint_acceleration
        if joint_acceleration_diagnostics and not acceleration_spec.enabled:
            acceleration_spec = JointAccelerationSpec(enabled=True)
        self._joint_acceleration = (
            JointAccelerationReward(acceleration_spec, action_spec=action_spec,
                                    control_hz=config.timing.control_hz, device=self.device)
            if acceleration_spec.enabled else None
        )
        self.episode_completion_mode = episode_completion_mode
        self.trajectory_count = (
            trajectory.trajectory_count
            if isinstance(trajectory, ReferenceTrajectoryBank)
            else 1
        )
        if self.trajectory_count != config.data.trajectory_count:
            raise ValueError(
                "materialized trajectory count disagrees with data config"
            )

        start_generator = torch.Generator(device=self.device)
        start_generator.manual_seed(config.seed + 101)
        lift_phase_boundaries = (
            reference_object_lift_phase_boundaries(
                trajectory,
                height_threshold_m=(
                    config.start_index.lift_height_threshold_m
                ),
            )
            if config.start_index.mode == "object_lift_phase_weighted"
            else None
        )
        sampler_factory = getattr(trajectory, "make_start_sampler", ReferenceStartSampler)
        self.start_sampler = sampler_factory(
            frame_count=trajectory.frame_count,
            pre_roll_policy_steps=config.perturbation.pre_roll_policy_steps,
            maximum_trajectory_fraction=(
                config.start_index.maximum_trajectory_fraction
            ),
            mode=config.start_index.mode,
            fixed_index=config.start_index.fixed_index,
            minimum=config.start_index.minimum,
            maximum=config.start_index.maximum,
            generator=start_generator,
            device=self.device,
            phase_boundary_indices=lift_phase_boundaries,
            pre_lift_reset_probability=(
                config.start_index.pre_lift_reset_probability
            ),
        )
        perturbation_generator = torch.Generator(device=self.device)
        perturbation_generator.manual_seed(config.seed + 202)
        self.perturbation = RandomActionPerturbation(
            action_spec,
            config.perturbation,
            generator=perturbation_generator,
        )
        if config.direct_action.mode == "arm_target_increment":
            from .incremental_arm_action import ArmTargetIncrementProjector
            self.direct_action = ArmTargetIncrementProjector(action_spec, config.direct_action)
        else:
            self.direct_action = DirectActionProjector(action_spec, config.direct_action)
        self._action_smoothness = (
            ActionSmoothnessReward(config.action_smoothness, action_spec=action_spec,
                                   num_envs=num_envs, control_hz=config.timing.control_hz,
                                   device=self.device)
            if config.action_smoothness.enabled else None
        )
        self.observation_assembler = TrajectoryObservationAssembler(
            action_spec, config.observation
        )
        self._joint_history = None
        if config.observation.robot_joint_temporal_context:
            from .robot_temporal_observation import ActualJointHistory
            self._joint_history = ActualJointHistory(num_envs, self.device)
        self.episode: ReferenceEpisodeBatch | None = None
        self.live_state: LiveState | None = None
        self.last_executed_action: Tensor | None = None
        self._previous_policy_action = (
            torch.zeros(num_envs, 27, dtype=torch.float32, device=self.device)
            if config.observation.include_previous_policy_action
            else None
        )
        self.reference_cursors = torch.zeros(
            num_envs, dtype=torch.int64, device=self.device
        )
        self.episode_steps = torch.zeros_like(self.reference_cursors)
        self.episode_ids = torch.full_like(self.reference_cursors, -1)
        self._active_mask = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_reset_mask = torch.zeros_like(self._active_mask)
        self._pre_roll_stage = torch.full_like(self.reference_cursors, -1)
        self._pre_roll_actions = torch.zeros(
            num_envs,
            config.perturbation.pre_roll_policy_steps,
            27,
            dtype=torch.float32,
            device=self.device,
        )
        self._all_env_ids = torch.arange(
            num_envs, dtype=torch.int64, device=self.device
        )
        self.needs_reset = True
        self._maniptrans_simulation_frame_count = 0
        self._episode_flow_error_sum = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_flow_sample_count = torch.zeros_like(self.reference_cursors)
        self._episode_object_departed = torch.zeros_like(self._active_mask)

    @property
    def rollout_state_schema_version(self) -> int:
        if self._joint_history is not None:
            from .robot_temporal_observation import ROLLOUT_SCHEMA_VERSION
            return ROLLOUT_SCHEMA_VERSION
        if self._action_smoothness is not None:
            return ACTION_SMOOTHNESS_ROLLOUT_STATE_SCHEMA_VERSION
        if self.config.observation.include_previous_policy_action:
            return TEMPORAL_OBSERVATION_ROLLOUT_STATE_SCHEMA_VERSION
        return (
            TASK_OUTCOME_ROLLOUT_STATE_SCHEMA_VERSION
            if self.config.uses_task_outcome_tracking
            else ROLLOUT_STATE_SCHEMA_VERSION
        )

    def _generator_state_dict(self) -> dict[str, Tensor]:
        return {
            "start_sampler": self.start_sampler.generator.get_state().clone(),
            "perturbation": self.perturbation.generator.get_state().clone(),
        }

    @staticmethod
    def _validated_generator_state(state: object) -> dict[str, Tensor]:
        if not isinstance(state, dict) or set(state) != {
            "start_sampler",
            "perturbation",
        }:
            raise ValueError("reset RNG state must contain both task generators")
        validated: dict[str, Tensor] = {}
        for name in ("start_sampler", "perturbation"):
            value = state[name]
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.uint8
                or value.ndim != 1
                or value.numel() == 0
            ):
                raise ValueError(f"reset RNG state {name!r} is invalid")
            validated[name] = value.detach().cpu().clone()
        return validated

    def _load_generator_state(self, state: object) -> None:
        validated = self._validated_generator_state(state)
        self.start_sampler.generator.set_state(validated["start_sampler"])
        self.perturbation.generator.set_state(validated["perturbation"])

    def reset_rng_state_dict(self) -> dict[str, Tensor]:
        """Capture episode-boundary RNG state for exact checkpoint resume."""

        if not self.needs_reset:
            raise RuntimeError("reset RNG can only be saved at an episode boundary")
        return self._generator_state_dict()

    def load_reset_rng_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore reset RNG without restoring simulator state mid-episode."""

        if not self.needs_reset:
            raise RuntimeError("reset RNG can only be restored at an episode boundary")
        self._load_generator_state(state)

    def rollout_state_dict(self) -> dict[str, object]:
        """Capture a rollout boundary, including every asynchronous env phase."""

        payload: dict[str, object] = {
            "schema_version": self.rollout_state_schema_version,
            "phase": "needs_reset" if self.needs_reset else "active",
            "reset_rng_state": self._generator_state_dict(),
            "episode": None,
        }
        payload["maniptrans_simulation_frame_count"] = (
            self._maniptrans_simulation_frame_count
        )
        if self.needs_reset:
            return payload
        if (
            self.episode is None
            or self.live_state is None
            or self.last_executed_action is None
        ):
            raise RuntimeError("active trajectory task lacks rollout state")
        payload["episode"] = {
            "start_indices": self.episode.start_indices.detach().clone(),
            "ppo_start_cursors": self.episode.ppo_start_cursors.detach().clone(),
            "reference_cursors": self.reference_cursors.detach().clone(),
            "episode_steps": self.episode_steps.detach().clone(),
            "episode_ids": self.episode_ids.detach().clone(),
            "previous_joint_target": self.last_executed_action.detach().clone(),
            "active_mask": self._active_mask.detach().clone(),
            "pending_reset_mask": self._pending_reset_mask.detach().clone(),
            "pre_roll_stage": self._pre_roll_stage.detach().clone(),
            "pre_roll_actions": self._pre_roll_actions.detach().clone(),
            "live_state": {
                name: getattr(self.live_state, name).detach().clone()
                for name in _LIVE_STATE_TENSOR_NAMES
            },
        }
        if self.trajectory_count > 1:
            assert self.episode.trajectory_indices is not None
            episode_payload = payload["episode"]
            assert isinstance(episode_payload, dict)
            episode_payload["trajectory_indices"] = (
                self.episode.trajectory_indices.detach().clone()
            )
        if self.config.uses_task_outcome_tracking:
            episode_payload = payload["episode"]
            assert isinstance(episode_payload, dict)
            episode_payload.update(
                {
                    "episode_flow_error_sum": (
                        self._episode_flow_error_sum.detach().clone()
                    ),
                    "episode_flow_sample_count": (
                        self._episode_flow_sample_count.detach().clone()
                    ),
                    "episode_object_departed": (
                        self._episode_object_departed.detach().clone()
                    ),
                }
            )
        if self._previous_policy_action is not None:
            episode_payload = payload["episode"]
            assert isinstance(episode_payload, dict)
            episode_payload["previous_policy_action"] = (
                self._previous_policy_action.detach().clone()
            )
        if self._action_smoothness is not None:
            episode_payload = payload["episode"]
            assert isinstance(episode_payload, dict)
            episode_payload["action_smoothness"] = self._action_smoothness.state_dict()
        if self._joint_history is not None:
            episode_payload = payload["episode"]
            assert isinstance(episode_payload, dict)
            episode_payload["actual_joint_history"] = self._joint_history.state_dict()
        return payload

    def load_rollout_state_dict(self, state: object) -> None:
        """Restore a saved rollout boundary and its simulator-visible dynamics."""

        expected_state_keys = {
            "schema_version",
            "phase",
            "reset_rng_state",
            "episode",
        }
        expected_state_keys.add("maniptrans_simulation_frame_count")
        if not isinstance(state, dict) or set(state) != expected_state_keys:
            raise ValueError("rollout state has invalid keys")
        if state["schema_version"] != self.rollout_state_schema_version:
            raise ValueError("unsupported rollout-state schema")
        generators = self._validated_generator_state(state["reset_rng_state"])
        phase = state["phase"]
        episode_payload = state["episode"]
        frame_count = state["maniptrans_simulation_frame_count"]
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 0
        ):
            raise ValueError("ManipTrans rollout frame count is invalid")
        self._maniptrans_simulation_frame_count = frame_count
        if phase == "needs_reset":
            if episode_payload is not None:
                raise ValueError("reset-boundary rollout state must not contain an episode")
            self._load_generator_state(generators)
            self.episode = None
            self.live_state = None
            self.last_executed_action = None
            if self._joint_history is not None:
                self._joint_history.clear()
            if self._action_smoothness is not None:
                self._action_smoothness.clear()
            if self._previous_policy_action is not None:
                self._previous_policy_action.zero_()
            self.reference_cursors.zero_()
            self.episode_steps.zero_()
            self.episode_ids.fill_(-1)
            self._active_mask.zero_()
            self._pending_reset_mask.zero_()
            self._pre_roll_stage.fill_(-1)
            self._pre_roll_actions.zero_()
            self._episode_flow_error_sum.zero_()
            self._episode_flow_sample_count.zero_()
            self._episode_object_departed.zero_()
            self.needs_reset = True
            return
        if phase != "active" or not isinstance(episode_payload, dict):
            raise ValueError("rollout state phase must be needs_reset or active")
        expected_episode_keys = {
            "start_indices",
            "ppo_start_cursors",
            "reference_cursors",
            "episode_steps",
            "episode_ids",
            "previous_joint_target",
            "active_mask",
            "pending_reset_mask",
            "pre_roll_stage",
            "pre_roll_actions",
            "live_state",
        }
        if self.trajectory_count > 1:
            expected_episode_keys.add("trajectory_indices")
        if self.config.uses_task_outcome_tracking:
            expected_episode_keys.update(
                {
                    "episode_flow_error_sum",
                    "episode_flow_sample_count",
                    "episode_object_departed",
                }
            )
        if self.config.observation.include_previous_policy_action:
            expected_episode_keys.add("previous_policy_action")
        if self._action_smoothness is not None:
            expected_episode_keys.add("action_smoothness")
        if self._joint_history is not None:
            expected_episode_keys.add("actual_joint_history")
        if set(episode_payload) != expected_episode_keys:
            raise ValueError("active rollout episode has invalid keys")
        if self._joint_history is not None:
            self._joint_history.validate_state(episode_payload["actual_joint_history"])

        def device_tensor(name: str, *, dtype: torch.dtype) -> Tensor:
            value = episode_payload[name]
            if not isinstance(value, Tensor) or value.dtype != dtype:
                raise ValueError(f"rollout episode {name!r} has invalid dtype")
            return value.detach().to(device=self.device).clone()

        starts = device_tensor("start_indices", dtype=torch.int64)
        cursors = device_tensor("ppo_start_cursors", dtype=torch.int64)
        trajectory_indices = (
            device_tensor("trajectory_indices", dtype=torch.int64)
            if self.trajectory_count > 1
            else torch.zeros_like(starts)
        )
        if self._object_assignment is not None:
            self._object_assignment.validate(
                torch.arange(self.num_envs, device=self.device), trajectory_indices,
            )
        episode = ReferenceEpisodeBatch(
            trajectory=self.trajectory,
            start_indices=starts,
            ppo_start_cursors=cursors,
            flow_anchor_offsets_steps=self.config.timing.flow_anchor_offsets_steps,
            trajectory_indices=trajectory_indices,
        )
        if episode.batch_size != self.num_envs:
            raise ValueError("rollout episode batch size changed")
        expected_ppo_cursors = (
            starts + self.config.perturbation.pre_roll_policy_steps
        )
        if not torch.equal(cursors, expected_ppo_cursors):
            raise ValueError("rollout PPO cursors disagree with configured pre-roll")
        reference_cursors = device_tensor("reference_cursors", dtype=torch.int64)
        episode_steps = device_tensor("episode_steps", dtype=torch.int64)
        episode_ids = device_tensor("episode_ids", dtype=torch.int64)
        vector_shape = (self.num_envs,)
        if any(
            value.shape != vector_shape
            for value in (reference_cursors, episode_steps, episode_ids)
        ):
            raise ValueError("rollout per-environment progress has invalid shape")
        terminal = episode.terminal_cursors
        if (
            torch.logical_or(reference_cursors < starts, reference_cursors > terminal)
            .any()
            .item()
            or (episode_steps < 0).any().item()
            or (episode_ids < 0).any().item()
        ):
            raise ValueError("rollout per-environment progress is outside its episode")
        previous = device_tensor("previous_joint_target", dtype=torch.float32)
        if previous.shape != (self.num_envs, 27):
            raise ValueError("rollout previous action has invalid shape")
        self.action_spec.validate(
            previous,
            name="rollout_previous_joint_target",
            check_bounds=True,
        )
        previous_policy_action = (
            device_tensor("previous_policy_action", dtype=torch.float32)
            if self.config.observation.include_previous_policy_action
            else None
        )
        if previous_policy_action is not None:
            if previous_policy_action.shape != (self.num_envs, 27):
                raise ValueError("rollout previous policy action has invalid shape")
            tolerance = 32.0 * torch.finfo(torch.float32).eps
            if (
                torch.logical_or(
                    previous_policy_action < -1.0 - tolerance,
                    previous_policy_action > 1.0 + tolerance,
                )
                .any()
                .item()
            ):
                raise ValueError("rollout previous policy action violates [-1,1]")
        active_mask = device_tensor("active_mask", dtype=torch.bool)
        pending_reset_mask = device_tensor("pending_reset_mask", dtype=torch.bool)
        pre_roll_stage = device_tensor("pre_roll_stage", dtype=torch.int64)
        pre_roll_actions = device_tensor("pre_roll_actions", dtype=torch.float32)
        if self.config.uses_task_outcome_tracking:
            episode_flow_error_sum = device_tensor(
                "episode_flow_error_sum", dtype=torch.float32
            )
            episode_flow_sample_count = device_tensor(
                "episode_flow_sample_count", dtype=torch.int64
            )
            episode_object_departed = device_tensor(
                "episode_object_departed", dtype=torch.bool
            )
        else:
            episode_flow_error_sum = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            episode_flow_sample_count = torch.zeros_like(self.reference_cursors)
            episode_object_departed = torch.zeros_like(self._active_mask)
        if any(
            value.shape != vector_shape
            for value in (
                active_mask,
                pending_reset_mask,
                pre_roll_stage,
            )
        ):
            raise ValueError("rollout activity masks have invalid shape")
        if any(
            value.shape != vector_shape
            for value in (
                episode_flow_error_sum,
                episode_flow_sample_count,
                episode_object_departed,
            )
        ):
            raise ValueError("rollout task-outcome accumulators have invalid shape")
        if (
            (episode_flow_error_sum < 0.0).any().item()
            or not torch.isfinite(episode_flow_error_sum).all().item()
            or (episode_flow_sample_count < 0).any().item()
            or (episode_flow_sample_count > episode_steps).any().item()
        ):
            raise ValueError("rollout task-outcome accumulators are invalid")
        expected_pre_roll_shape = (
            self.num_envs,
            self.config.perturbation.pre_roll_policy_steps,
            27,
        )
        if pre_roll_actions.shape != expected_pre_roll_shape:
            raise ValueError("rollout pre-roll action queue has invalid shape")
        if not torch.isfinite(pre_roll_actions).all().item():
            raise ValueError("rollout pre-roll action queue is non-finite")
        pre_roll_mask = pre_roll_stage >= 0
        if (
            (pre_roll_stage >= self.config.perturbation.pre_roll_policy_steps)
            .any()
            .item()
            or (active_mask & (pending_reset_mask | pre_roll_mask)).any().item()
            or (pending_reset_mask & pre_roll_mask).any().item()
        ):
            raise ValueError("rollout asynchronous phases overlap")
        expected_regular_cursor = episode.ppo_start_cursors + episode_steps
        regular = ~pre_roll_mask
        if (
            (reference_cursors[regular] != expected_regular_cursor[regular])
            .any()
            .item()
        ):
            raise ValueError("rollout regular cursor and episode progress disagree")
        if pre_roll_mask.any().item():
            expected_pre_cursor = starts + pre_roll_stage.clamp_min(0)
            if (
                (reference_cursors[pre_roll_mask] != expected_pre_cursor[pre_roll_mask])
                .any()
                .item()
                or (episode_steps[pre_roll_mask] != 0).any().item()
            ):
                raise ValueError("rollout pre-roll cursor and stage disagree")
        if (active_mask & (reference_cursors >= terminal)).any().item():
            raise ValueError("an active rollout environment is already terminal")
        live_payload = episode_payload["live_state"]
        if (
            not isinstance(live_payload, dict)
            or set(live_payload) != set(_LIVE_STATE_TENSOR_NAMES)
        ):
            raise ValueError("rollout live state has invalid keys")
        live_values: dict[str, Tensor] = {}
        for name in live_payload:
            value = live_payload[name]
            if not isinstance(value, Tensor) or value.dtype != torch.float32:
                raise ValueError(f"rollout live state {name!r} has invalid dtype")
            live_values[name] = value.detach().to(device=self.device).clone()
        live = LiveState(**live_values)
        self._validate_live(live)

        root = episode._gather_trajectory_values("robot_root_pose_env_local")
        table = episode._gather_trajectory_values("table_root_pose_env_local")
        object_points = episode._gather_trajectory_values(
            "object_points_object_frame"
        )
        self.backend.restore(
            ReferenceResetState(
                frame_index=reference_cursors.detach().clone(),
                joint_position=live.joint_position,
                joint_velocity=live.joint_velocity,
                previous_joint_target=previous,
                object_pose_robot_base=live.object_pose_robot_base,
                object_linear_velocity_robot_base=(
                    live.object_linear_velocity_robot_base
                ),
                object_angular_velocity_robot_base=(
                    live.object_angular_velocity_robot_base
                ),
                object_points_object_frame=object_points,
                robot_root_pose_env_local=root,
                table_root_pose_env_local=table,
            ),
            env_ids=self._all_env_ids,
        )
        self._load_generator_state(generators)
        self.episode = episode
        self.live_state = live
        self.last_executed_action = previous
        if self._joint_history is not None:
            self._joint_history.load_state_dict(episode_payload["actual_joint_history"])
        if self._previous_policy_action is not None:
            assert previous_policy_action is not None
            self._previous_policy_action.copy_(previous_policy_action)
        if self._action_smoothness is not None:
            self._action_smoothness.load_state_dict(episode_payload["action_smoothness"])
        self._active_mask = active_mask
        self._pending_reset_mask = pending_reset_mask
        self._pre_roll_stage = pre_roll_stage
        self._pre_roll_actions = pre_roll_actions
        self.reference_cursors = reference_cursors
        self.episode_steps = episode_steps
        self.episode_ids = episode_ids
        self._episode_flow_error_sum = episode_flow_error_sum
        self._episode_flow_sample_count = episode_flow_sample_count
        self._episode_object_departed = episode_object_departed
        self.needs_reset = False

    def _validate_live(self, state: LiveState) -> None:
        if not isinstance(state, LiveState):
            raise TypeError("backend restore/step must return LiveState")
        state.validate(
            num_envs=self.num_envs,
            num_object_points=self.config.observation.num_object_points,
            fingertip_count=self.config.observation.fingertip_count,
            hand_keypoint_count=self.trajectory.hand_keypoint_count,
            device=self.device,
        )

    def _observation(self, state: LiveState) -> Tensor:
        episode = self.episode
        if episode is None:
            raise RuntimeError("task has no active reference episode")
        (
            include_current_reference_joint_target,
            include_current_reference_link7_target,
        ) = current_reference_robot_target_requirements(
            self.config.observation.current_reference_robot_target_features
        )
        reference = episode.observation_targets(
            self.reference_cursors,
            include_current_link7_pose=include_current_reference_link7_target,
        )
        progress = (
            self.episode_steps.to(dtype=torch.float32)
            / episode.episode_lengths.to(dtype=torch.float32)
        ).clamp_(0.0, 1.0)[:, None]
        include_reference_robot_errors = (
            self.config.observation.include_reference_robot_state
        )
        include_reference_robot_target = (
            self.config.observation.reference_robot_target_representation
            != "none"
        )
        link7_pose = (
            torch.cat(
                (
                    state.link7_position_robot_base,
                    state.link7_orientation_robot_base,
                ),
                dim=-1,
            )
            if self.config.observation.include_current_link7_pose
            else None
        )
        temporal_joint_positions = None
        if self._joint_history is not None:
            from .robot_temporal_observation import FUTURE_STEPS
            temporal_joint_positions = torch.cat((
                self._joint_history.features(),
                episode.joint_position_targets(self.reference_cursors, FUTURE_STEPS),
            ), dim=1)
        return self.observation_assembler.assemble(
            TrajectoryObservationInput(
                temporal_joint_positions=temporal_joint_positions,
                joint_position=state.joint_position,
                joint_velocity=state.joint_velocity,
                reference_joint_position=(
                    reference.joint_position
                    if (
                        include_reference_robot_errors
                        or include_current_reference_joint_target
                    )
                    else (
                        reference.next_joint_position
                        if include_reference_robot_target
                        else None
                    )
                ),
                reference_joint_velocity=(
                    reference.joint_velocity
                    if include_reference_robot_errors
                    else None
                ),
                live_object_points_robot_base=(state.object_surface_points_robot_base),
                reference_object_points_robot_base=(reference.object_surface_points),
                reference_object_flow=reference.reference_object_flow,
                object_pose_robot_base=state.object_pose_robot_base,
                object_linear_velocity_robot_base=(
                    state.object_linear_velocity_robot_base
                ),
                object_angular_velocity_robot_base=(
                    state.object_angular_velocity_robot_base
                ),
                fingertip_positions_robot_base=(state.fingertip_positions_robot_base),
                episode_progress=progress,
                previous_policy_action=(
                    self.direct_action.normalize_executed_target(self.last_executed_action)
                    if self.config.direct_action.mode == "arm_target_increment"
                    else self._previous_policy_action
                ),
                link7_pose_robot_base=link7_pose,
                reference_link7_pose_robot_base=(
                    reference.link7_pose_robot_base
                    if include_current_reference_link7_target
                    else (
                        reference.next_link7_pose_robot_base
                        if include_reference_robot_target
                        else None
                    )
                ),
            )
        )

    @torch.no_grad()
    def current_observation(self) -> Tensor:
        if self.needs_reset or self.live_state is None:
            raise RuntimeError("task has no active observation")
        return self._observation(self.live_state)

    def _replace_episode(
        self,
        env_ids: Tensor,
        replacement: ReferenceEpisodeBatch,
    ) -> None:
        if replacement.batch_size != env_ids.numel():
            raise ValueError("replacement episode batch and env_ids disagree")
        if self.episode is None:
            if not torch.equal(env_ids, self._all_env_ids):
                raise RuntimeError("the first reset must cover every environment")
            self.episode = replacement
            return
        starts = self.episode.start_indices.clone()
        starts.index_copy_(0, env_ids, replacement.start_indices)
        assert self.episode.trajectory_indices is not None
        assert replacement.trajectory_indices is not None
        trajectory_indices = self.episode.trajectory_indices.clone()
        trajectory_indices.index_copy_(
            0,
            env_ids,
            replacement.trajectory_indices,
        )
        self.episode = build_reference_episode(
            self.trajectory,
            start_indices=starts,
            pre_roll_policy_steps=self.config.perturbation.pre_roll_policy_steps,
            flow_anchor_offsets_steps=self.config.timing.flow_anchor_offsets_steps,
            trajectory_indices=trajectory_indices,
        )

    def _sample_trajectory_indices(
        self,
        env_ids: Tensor,
        explicit: Tensor | None,
    ) -> Tensor:
        count = int(env_ids.numel())
        if explicit is None:
            if self._object_assignment is not None:
                if self.config.start_index.trajectory_assignment == "fixed_cyclic":
                    result = torch.remainder(env_ids, self.trajectory_count)
                    self._object_assignment.validate(env_ids, result)
                    return result
                return self._object_assignment.sample(env_ids, generator=self.start_sampler.generator)
            if self.trajectory_count == 1:
                return torch.zeros(count, dtype=torch.int64, device=self.device)
            if self.config.start_index.trajectory_assignment == "fixed_cyclic":
                return torch.remainder(env_ids, self.trajectory_count)
            return torch.randint(
                0,
                self.trajectory_count,
                (count,),
                dtype=torch.int64,
                device=self.device,
                generator=self.start_sampler.generator,
            )
        if (
            not isinstance(explicit, Tensor)
            or explicit.shape != (count,)
            or explicit.dtype != torch.int64
            or explicit.device != self.device
            or (explicit < 0).any().item()
            or (explicit >= self.trajectory_count).any().item()
        ):
            raise ValueError(
                "explicit trajectory indices must be valid int64 [count]"
            )
        if self._object_assignment is not None:
            self._object_assignment.validate(env_ids, explicit)
        return explicit.detach().clone()

    def _reset_hold_target(self, joint_position: Tensor) -> Tensor:
        self.action_spec.validate(
            joint_position,
            name="reset_joint_position",
            check_bounds=False,
        )
        lower, upper = self.action_spec.bounds(joint_position)
        return torch.maximum(torch.minimum(joint_position, upper), lower)


    def _prepare_resets(
        self,
        env_ids: Tensor,
        *,
        start_indices: Tensor | None = None,
        trajectory_indices: Tensor | None = None,
        start_index_range: Literal["configured", "full_reference"] = "configured",
    ) -> tuple[ReferenceEpisodeBatch, PerturbationPlan, LiveState, Tensor]:
        if (
            env_ids.ndim != 1
            or env_ids.dtype != torch.int64
            or env_ids.device != self.device
            or env_ids.numel() == 0
        ):
            raise ValueError("reset env_ids must be non-empty int64 [M]")
        count = int(env_ids.numel())
        selected_trajectories = self._sample_trajectory_indices(
            env_ids,
            trajectory_indices,
        )
        starts = (
            self.start_sampler.sample(
                count,
                trajectory_indices=selected_trajectories,
            )
            if start_indices is None
            else self.start_sampler.validate_indices(
                start_indices,
                count=count,
                range_mode=start_index_range,
                trajectory_indices=selected_trajectories,
            )
        )
        replacement = build_reference_episode(
            self.trajectory,
            start_indices=starts,
            pre_roll_policy_steps=self.config.perturbation.pre_roll_policy_steps,
            flow_anchor_offsets_steps=self.config.timing.flow_anchor_offsets_steps,
            trajectory_indices=selected_trajectories,
        )
        reset_state = replacement.reset_state()
        hold_target = self._reset_hold_target(reset_state.joint_position)
        reset_state = replace(
            reset_state,
            previous_joint_target=hold_target,
        )
        restored_state = self.backend.restore(reset_state, env_ids=env_ids)
        self._validate_live(restored_state)
        if self._joint_history is not None:
            self._joint_history.reset(env_ids, restored_state.joint_position)
        reset_reference_points = replacement.step_targets(
            replacement.start_indices
        ).object_surface_points
        plan = self.perturbation.plan(replacement)
        self._replace_episode(env_ids, replacement)
        self.reference_cursors.index_copy_(0, env_ids, replacement.start_indices)
        self.episode_steps[env_ids] = 0
        self._episode_flow_error_sum[env_ids] = 0.0
        self._episode_flow_sample_count[env_ids] = 0
        self._episode_object_departed[env_ids] = False
        if self._action_smoothness is not None:
            self._action_smoothness.reset(env_ids)
        if self._previous_policy_action is not None:
            self._previous_policy_action[env_ids] = 0.0
        self.episode_ids[env_ids] += 1
        has_pre_roll = plan.pre_roll_policy_steps > 0
        self._active_mask[env_ids] = not has_pre_roll
        self._pending_reset_mask[env_ids] = False
        self._pre_roll_stage[env_ids] = 0 if has_pre_roll else -1
        self._pre_roll_actions.index_copy_(
            0,
            env_ids,
            plan.executed_pre_roll_actions,
        )
        return replacement, plan, restored_state, hold_target

    @torch.no_grad()
    def reset(
        self,
        *,
        start_indices: Tensor | None = None,
        trajectory_indices: Tensor | None = None,
        start_index_range: Literal["configured", "full_reference"] = "configured",
    ) -> TaskResetResult:
        """Reset all environments, optionally to an explicit deterministic batch."""

        # Training leaves start_indices unset so each environment samples its own
        # reset stream. Evaluation may provide one exact batch without consuming
        # that stream, which makes checkpoint comparisons order-independent.
        replacement, plan, state, hold_target = self._prepare_resets(
            self._all_env_ids,
            start_indices=start_indices,
            trajectory_indices=trajectory_indices,
            start_index_range=start_index_range,
        )
        for index in range(plan.pre_roll_policy_steps):
            state = self.backend.step(plan.executed_pre_roll_actions[:, index])
            self._maniptrans_simulation_frame_count += 1
            self._validate_live(state)
            if self._joint_history is not None:
                self._joint_history.advance(
                    state.joint_position, torch.ones_like(self._active_mask)
                )
            self.reference_cursors += 1
        receipt = self.perturbation.receipt(
            plan,
            post_joint_position=state.joint_position,
            post_object_pose_robot_base=state.object_pose_robot_base,
        )
        self.live_state = state
        self.last_executed_action = (
            plan.executed_pre_roll_actions[:, -1].clone()
            if plan.pre_roll_policy_steps > 0
            else hold_target.detach().clone()
        )
        self._active_mask.fill_(True)
        self._pending_reset_mask.zero_()
        self._pre_roll_stage.fill_(-1)
        self.episode_steps.zero_()
        self.needs_reset = False
        return TaskResetResult(
            observation=self._observation(state),
            episode=replacement,
            perturbation=receipt,
        )

    @property
    def episode_length_steps(self) -> int:
        if self.episode is None:
            raise RuntimeError("reset is required before reading episode length")
        return self.episode.shared_episode_length

    @property
    def remaining_episode_steps(self) -> int:
        if self.needs_reset or self.episode is None:
            raise RuntimeError("task has no active episode")
        remaining = self.episode.episode_lengths - self.episode_steps
        if remaining.ne(remaining[0]).any().item():
            raise RuntimeError("asynchronous environments have different horizons")
        return int(remaining[0].item())

    @property
    def episode_step(self) -> int:
        """Legacy scalar view, available only while all env progress is equal."""

        if self.episode_steps.ne(self.episode_steps[0]).any().item():
            raise RuntimeError("asynchronous environments have no shared episode_step")
        return int(self.episode_steps[0].item())

    @property
    def active_environment_mask(self) -> Tensor:
        """Logical environments that may still contribute a PPO transition."""

        if self.needs_reset:
            raise RuntimeError("task has no active episode")
        return self._active_mask.clone()

    @torch.no_grad()
    def normalized_nominal_action(self) -> Tensor:
        """Return the recorded action in policy coordinates for replay baselines."""

        if self.needs_reset or self.episode is None:
            raise RuntimeError("reset is required before reading nominal action")
        recorded_action = self.episode.recorded_actions(self.reference_cursors)
        if self.config.direct_action.mode == "arm_target_increment":
            return self.direct_action.normalize(recorded_action,
                previous_joint_target=self.last_executed_action)
        return self.direct_action.normalize(recorded_action)

    @torch.no_grad()
    def step(self, raw_action: Tensor, *, auto_reset: bool = True) -> TaskStepResult:
        if (
            self.needs_reset
            or self.episode is None
            or self.live_state is None
            or self.last_executed_action is None
        ):
            raise RuntimeError("reset is required before task.step")
        if (
            not isinstance(raw_action, Tensor)
            or raw_action.shape != (self.num_envs, 27)
            or raw_action.device != self.device
        ):
            raise ValueError(
                "raw zero-base residual must have shape [num_envs,27] "
                "on the task device"
            )
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset must be bool")
        reset_started = torch.zeros_like(self._active_mask)
        deferred_reset_ids = torch.empty(
            0, dtype=torch.int64, device=self.device
        )
        if auto_reset and self._pending_reset_mask.any().item():
            reset_ids = self._pending_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            if self.config.perturbation.pre_roll_policy_steps == 0:
                # The sampled action was produced from the terminal observation.
                # Advance other environments first, then restore this subset so
                # the next policy call sees the exact reset state without a
                # post-restore physics/reference pre-roll.
                deferred_reset_ids = reset_ids
            else:
                self._prepare_resets(reset_ids)
            reset_started[reset_ids] = True
        elif not auto_reset and self._pending_reset_mask.any().item():
            raise RuntimeError("pending training resets require auto_reset=True")

        state_before_step = self.live_state
        # Backends may reuse velocity buffers: snapshot before advancing physics.
        velocity_before = (state_before_step.joint_velocity.detach().clone()
                           if self._joint_acceleration is not None else None)
        pre_roll_mask = self._pre_roll_stage >= 0
        valid_transition = self._active_mask.clone()
        if (valid_transition & pre_roll_mask).any().item():
            raise RuntimeError("PPO and reset pre-roll phases overlap")
        if self.config.direct_action.mode == "arm_target_increment":
            action = self.direct_action.apply(raw_action,
                previous_joint_target=self.last_executed_action)
        else:
            action = self.direct_action.apply(raw_action)
        executed_action = torch.where(
            valid_transition[:, None],
            action.executed_action,
            self.last_executed_action,
        )
        if pre_roll_mask.any().item():
            rows = self._all_env_ids[pre_roll_mask]
            stages = self._pre_roll_stage[pre_roll_mask]
            executed_action[pre_roll_mask] = self._pre_roll_actions[rows, stages]
        action = replace(action, executed_action=executed_action)
        terminal_cursor = self.episode.terminal_cursors
        target_cursors = self.reference_cursors.clamp_max(terminal_cursor - 1)
        targets = self.episode.step_targets(target_cursors)
        state = self.backend.step(action.executed_action)
        self._maniptrans_simulation_frame_count += 1
        self._validate_live(state)
        if self._joint_history is not None:
            self._joint_history.advance(
                state.joint_position, valid_transition | pre_roll_mask
            )
        tracking, maniptrans_diagnostics = trajectory_tracking_reward(
            state.object_surface_points_robot_base,
            targets.next_object_surface_points,
            state.joint_position,
            targets.next_joint_position,
            state.joint_velocity,
            targets.next_joint_velocity,
            torch.cat(
                (
                    state.link7_position_robot_base,
                    state.link7_orientation_robot_base,
                ),
                dim=-1,
            ),
            targets.next_link7_pose_robot_base,
            state.link7_linear_velocity_robot_base,
            targets.next_link7_linear_velocity_robot_base,
            state.link7_angular_velocity_robot_base,
            targets.next_link7_angular_velocity_robot_base,
            state.hand_keypoint_positions_robot_base,
            targets.next_hand_keypoint_positions_robot_base,
            state.hand_keypoint_linear_velocities_robot_base,
            targets.next_hand_keypoint_linear_velocities_robot_base,
            state.joint_effort,
            self.episode_steps,
            self._maniptrans_simulation_frame_count,
            self.action_spec,
            spec=self.config.reward,
        )
        motion_conditioned_tracking = motion_conditioned_tracking_diagnostics(
            actual_object_points_before=(
                state_before_step.object_surface_points_robot_base
            ),
            actual_object_points_after=state.object_surface_points_robot_base,
            reference_object_points_before=targets.object_surface_points,
            reference_object_points_after=targets.next_object_surface_points,
            actual_robot_keypoints_before=(
                state_before_step.hand_keypoint_positions_robot_base
            ),
            actual_robot_keypoints_after=(
                state.hand_keypoint_positions_robot_base
            ),
            reference_robot_keypoints_before=(
                targets.hand_keypoint_positions_robot_base
            ),
            reference_robot_keypoints_after=(
                targets.next_hand_keypoint_positions_robot_base
            ),
        )
        joint_position_tracking = joint_position_tracking_error(
            state.joint_position,
            targets.next_joint_position,
            self.action_spec,
        )
        link7_position_tracking = link7_position_tracking_error(
            state.link7_position_robot_base,
            targets.next_link7_position_robot_base,
        )
        departure_tracking = object_departure_tracking(
            state.object_pose_robot_base,
            targets.next_object_pose_robot_base,
            self.episode_steps,
            self.config.termination,
        )
        object_departure_failure = valid_transition & departure_tracking.failed
        if self.config.termination.robot_tracking_reset_mode == (
            "link7_weighted_pose"
        ):
            robot_tracking_failed = link7_weighted_pose_failure(
                maniptrans_diagnostics.wrist_position_error_m,
                maniptrans_diagnostics.wrist_orientation_error_rad,
                maniptrans_diagnostics.numerical_sanity_failed,
                self.episode_steps,
                self.config.termination,
            ).failed
        else:
            robot_tracking_failed = maniptrans_diagnostics.failed
        maniptrans_failure = (
            valid_transition
            & robot_tracking_failed
            & ~object_departure_failure
        )
        if self.episode_completion_mode == "reference_endpoint_only":
            # Cross-reward evaluation must observe the same full reference suffix.
            # Keep continuous error diagnostics, but do not censor one candidate
            # with its training-time reward-specific failure/reset definition.
            maniptrans_failure = torch.zeros_like(valid_transition)
            object_departure_failure = torch.zeros_like(valid_transition)
        self._episode_flow_error_sum += torch.where(
            valid_transition,
            tracking.mean_point_error_m,
            torch.zeros_like(tracking.mean_point_error_m),
        )
        self._episode_flow_sample_count += valid_transition.to(dtype=torch.int64)
        self._episode_object_departed |= valid_transition & departure_tracking.detected
        self.reference_cursors[valid_transition] += 1
        self.episode_steps[valid_transition] += 1
        reached_endpoint = valid_transition & (
            self.reference_cursors == terminal_cursor
        )
        tracking_failure = object_departure_failure | maniptrans_failure
        episode_completed = reached_endpoint | tracking_failure
        terminated = episode_completed.clone()
        robot_endpoint_success = (
            reached_endpoint
            & ~tracking_failure
            & tracking.within_robot_success_threshold
        )
        if self.config.task_success.mode in (
            "robot_object_trajectory", "object_trajectory"
        ):
            mean_threshold = (
                self.config.task_success.episode_mean_flow_error_threshold_m
            )
            final_threshold = self.config.task_success.final_flow_error_threshold_m
            assert mean_threshold is not None and final_threshold is not None
            episode_mean_flow_error = self._episode_flow_error_sum / (
                self._episode_flow_sample_count.clamp_min(1).to(dtype=torch.float32)
            )
            episode_success = (
                reached_endpoint
                # A configured failure wins even when it coincides with the
                # endpoint. This is not an extra robot endpoint-quality gate;
                # independent_endpoint clears training failures above.
                & ~tracking_failure
                & (episode_mean_flow_error <= mean_threshold)
                & (tracking.mean_point_error_m <= final_threshold)
                & ~self._episode_object_departed
            )
            if self.config.task_success.mode == "robot_object_trajectory":
                # Preserve historical joint robot/object success exactly.
                episode_success &= robot_endpoint_success
        else:
            episode_success = robot_endpoint_success
        episode_object_departed = episode_completed & self._episode_object_departed
        self._active_mask[episode_completed] = False
        if auto_reset:
            self._pending_reset_mask[episode_completed] = True

        if pre_roll_mask.any().item():
            self.reference_cursors[pre_roll_mask] += 1
            last_pre_roll = pre_roll_mask & (
                self._pre_roll_stage
                == self.config.perturbation.pre_roll_policy_steps - 1
            )
            continuing_pre_roll = pre_roll_mask & ~last_pre_roll
            self._pre_roll_stage[continuing_pre_roll] += 1
            self._pre_roll_stage[last_pre_roll] = -1
            self._active_mask[last_pre_roll] = True

        smoothness = None
        reward = tracking.reward
        if self._action_smoothness is not None:
            reward_action = action.bounded_normalized_action
            if self.config.direct_action.mode == "arm_target_increment":
                reward_action = self.direct_action.normalize_executed_target(action.executed_action)
            smoothness = self._action_smoothness.evaluate(
                reward_action, state.joint_velocity, valid_transition)
            # Keep tracking diagnostics unmodified. Zero weights are an exact
            # bypass; shaping is additive and never renormalizes task rewards.
            if (self.config.action_smoothness.first_difference_weight > 0.0
                    or self.config.action_smoothness.second_difference_weight > 0.0):
                reward = reward - smoothness.penalty
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        acceleration = None
        if self._joint_acceleration is not None:
            assert velocity_before is not None
            acceleration = self._joint_acceleration.evaluate(
                velocity_before, state.joint_velocity, valid_transition)
            if self._joint_acceleration.spec.coefficient > 0:
                reward = reward - acceleration.penalty
        self.live_state = state
        self.last_executed_action = action.executed_action.detach().clone()
        if self._previous_policy_action is not None:
            self._previous_policy_action[valid_transition] = (
                action.bounded_normalized_action[valid_transition]
            )
        if deferred_reset_ids.numel() > 0:
            _, _, state, reset_hold_target = self._prepare_resets(
                deferred_reset_ids
            )
            self.live_state = state
            self.last_executed_action.index_copy_(
                0,
                deferred_reset_ids,
                reset_hold_target,
            )
        active_after_step = self._active_mask.clone()
        next_observation = self._observation(state)
        return TaskStepResult(
            next_observation=next_observation,
            reward=torch.where(
                valid_transition,
                reward,
                torch.zeros_like(tracking.reward),
            ),
            terminated=terminated,
            truncated=truncated,
            tracking=tracking,
            joint_position_tracking=joint_position_tracking,
            link7_position_tracking=link7_position_tracking,
            object_departure_tracking=departure_tracking,
            motion_conditioned_tracking=motion_conditioned_tracking,
            maniptrans_diagnostics=maniptrans_diagnostics,
            object_departure_failure=object_departure_failure,
            maniptrans_failure=maniptrans_failure,
            action=action,
            episode_steps=self.episode_steps.detach().clone(),
            reference_cursors=self.reference_cursors.detach().clone(),
            valid_transition=valid_transition,
            active_after_step=active_after_step.clone(),
            reached_endpoint=reached_endpoint,
            episode_completed=episode_completed,
            episode_success=episode_success,
            episode_object_departed=episode_object_departed,
            reset_started=reset_started,
            pre_roll_mask=pre_roll_mask,
            action_smoothness_diagnostics=smoothness,
            joint_acceleration_diagnostics=acceleration,
        )


_FINGERTIP_NAMES = tuple(f"right_finger{index}_link4" for index in range(1, 6))
_HAND_KEYPOINT_BODY_NAMES = tuple(
    f"right_finger{finger}_link{level}"
    for finger in range(1, 6)
    for level in (1, 2, 3, 4, 4)
)
_FINGERTIP_OFFSETS_M = (
    (-0.0011048, 0.00023218, 0.0283),
    (-0.00105, 0.0, 0.0274),
    (-0.0010491, 0.0, 0.0267),
    (-0.0010528, 0.0, 0.0267),
    (-0.0010446, 0.00001131, 0.0267),
)


def _quat_normalize(value: Tensor) -> Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )


def _quat_apply(quaternion: Tensor, vector: Tensor) -> Tensor:
    quaternion = _quat_normalize(quaternion)
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.cross(xyz, twice_cross, dim=-1)
    )


def _quat_multiply(left: Tensor, right: Tensor) -> Tensor:
    left = _quat_normalize(left)
    right = _quat_normalize(right)
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return _quat_normalize(
        torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
    )


def _compose_pose(parent: Tensor, child: Tensor) -> Tensor:
    return torch.cat(
        (
            parent[..., :3] + _quat_apply(parent[..., 3:], child[..., :3]),
            _quat_multiply(parent[..., 3:], child[..., 3:]),
        ),
        dim=-1,
    )


def _relative_pose(parent_world: Tensor, value_world: Tensor) -> Tensor:
    inverse_quaternion = torch.cat(
        (parent_world[..., 3:4], -parent_world[..., 4:]), dim=-1
    )
    return torch.cat(
        (
            _quat_apply(
                inverse_quaternion,
                value_world[..., :3] - parent_world[..., :3],
            ),
            _quat_multiply(inverse_quaternion, value_world[..., 3:]),
        ),
        dim=-1,
    )


def action_spec_from_isaac_robot(
    robot: Any,
    *,
    canonical_joint_names: tuple[str, ...],
) -> CanonicalActionSpec:
    """Resolve simulator limits by explicit name, never by tensor width alone."""

    simulator_names = tuple(robot.data.joint_names)
    if set(simulator_names) != set(canonical_joint_names) or len(simulator_names) != 27:
        raise ValueError("Isaac and reference joint-name sets disagree")
    canonical_from_lab = torch.tensor(
        [simulator_names.index(name) for name in canonical_joint_names],
        dtype=torch.int64,
        device=robot.data.joint_pos_limits.device,
    )
    limits = robot.data.joint_pos_limits[0].index_select(0, canonical_from_lab)
    return make_ordered_action_spec(
        joint_names=canonical_joint_names,
        position_lower_rad=tuple(float(value) for value in limits[:, 0].cpu().tolist()),
        position_upper_rad=tuple(float(value) for value in limits[:, 1].cpu().tolist()),
    )


def canonical_velocity_limits_from_isaac_robot(
    robot: Any,
    *,
    canonical_joint_names: tuple[str, ...],
) -> Tensor:
    """Resolve positive velocity limits in the declared canonical joint order."""

    simulator_names = tuple(robot.data.joint_names)
    if set(simulator_names) != set(canonical_joint_names) or len(simulator_names) != 27:
        raise ValueError("Isaac and reference joint-name sets disagree")
    canonical_from_lab = torch.tensor(
        [simulator_names.index(name) for name in canonical_joint_names],
        dtype=torch.int64,
        device=robot.data.joint_vel_limits.device,
    )
    limits = robot.data.joint_vel_limits[0].index_select(0, canonical_from_lab).abs()
    if limits.shape != (27,) or not torch.isfinite(limits).all().item() or (limits <= 0).any().item():
        raise ValueError("Isaac canonical joint velocity limits are invalid")
    return limits.to(dtype=torch.float32)


class IsaacTrajectoryBackend:
    """Duck-typed Isaac bridge for one restored state and one 60 Hz action.

    Importing this module does not import or launch Isaac. A task-local launcher
    owns scene construction and passes its simulator, scene, and asset handles.
    """

    def __init__(
        self,
        *,
        simulation_context: Any,
        scene: Any,
        robot: Any,
        table: Any,
        object_asset: Any,
        trajectory: ReferenceCollection,
        action_spec: CanonicalActionSpec,
        physics_dt: float,
        object_physics_mode: Literal["dynamic", "disabled"] = "dynamic",
        render: bool = False,
        health_check=None,
    ) -> None:
        if trajectory.joint_names != action_spec.joint_names:
            raise ValueError("Isaac backend action/reference joint orders disagree")
        if physics_dt <= 0 or abs(physics_dt - 1.0 / trajectory.physics_hz) > 1.0e-12:
            raise ValueError("Isaac physics_dt disagrees with trajectory physics_hz")
        self.sim = simulation_context
        self.scene = scene
        self.robot = robot
        self.table = table
        self.object = object_asset
        self.trajectory = trajectory
        self.action_spec = action_spec
        self.physics_dt = float(physics_dt)
        if object_physics_mode not in ("dynamic", "disabled"):
            raise ValueError("unsupported Isaac object physics mode")
        self.object_physics_mode = object_physics_mode
        if not isinstance(render, bool):
            raise TypeError("Isaac backend render must be a bool")
        self.render = render
        self.health_check = health_check or (lambda: None)
        self.decimation = trajectory.physics_hz // trajectory.control_hz
        self.device = trajectory.device
        self.num_envs = int(scene.env_origins.shape[0])
        if scene.env_origins.device != self.device:
            raise ValueError("Isaac scene and reference trajectory must share a device")

        simulator_names = tuple(robot.data.joint_names)
        if set(simulator_names) != set(action_spec.joint_names):
            raise ValueError("Isaac robot and canonical joint-name sets disagree")
        self._lab_from_canonical = torch.tensor(
            [action_spec.joint_names.index(name) for name in simulator_names],
            dtype=torch.int64,
            device=self.device,
        )
        self._canonical_from_lab = torch.tensor(
            [simulator_names.index(name) for name in action_spec.joint_names],
            dtype=torch.int64,
            device=self.device,
        )
        body_names = tuple(robot.data.body_names)
        self._requires_hand_kinematics = (
            trajectory.hand_kinematic_reference_available
        )
        required_bodies = (
            (*_HAND_KEYPOINT_BODY_NAMES, "link7")
            if self._requires_hand_kinematics
            else (*_FINGERTIP_NAMES, "link7")
        )
        missing = [name for name in required_bodies if name not in body_names]
        if missing:
            raise ValueError(f"Isaac robot lacks required bodies: {missing}")
        self._fingertip_ids = torch.tensor(
            [body_names.index(name) for name in _FINGERTIP_NAMES],
            dtype=torch.int64,
            device=self.device,
        )
        self._fingertip_offsets = torch.tensor(
            _FINGERTIP_OFFSETS_M, dtype=torch.float32, device=self.device
        )
        keypoint_names = (
            _HAND_KEYPOINT_BODY_NAMES
            if self._requires_hand_kinematics
            else tuple(name for name in _FINGERTIP_NAMES for _ in range(4))
        )
        self._hand_keypoint_ids = torch.tensor(
            [body_names.index(name) for name in keypoint_names],
            dtype=torch.int64,
            device=self.device,
        )
        keypoint_count = (
            WUJI_KEYPOINT_COUNT if self._requires_hand_kinematics else 20
        )
        keypoint_offsets = torch.zeros(
            keypoint_count, 3, dtype=torch.float32, device=self.device
        )
        if self._requires_hand_kinematics:
            keypoint_offsets[
                WUJI_KEYPOINTS_PER_FINGER - 1 :: WUJI_KEYPOINTS_PER_FINGER
            ] = self._fingertip_offsets
        else:
            keypoint_offsets.copy_(self._fingertip_offsets.repeat_interleave(4, dim=0))
        self._hand_keypoint_offsets = keypoint_offsets
        self._link7_id = body_names.index("link7")
        self._env_ids = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._target_lab = torch.zeros(
            self.num_envs, 27, dtype=torch.float32, device=self.device
        )
        self._object_points_object_frame = torch.zeros(
            self.num_envs,
            trajectory.num_object_points,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        try:
            self._contact_sensors = tuple(
                scene[f"finger{index}_object_contact"] for index in range(1, 6)
            )
        except KeyError:
            self._contact_sensors = ()

    def fingertip_net_contact_forces_w(self) -> Tensor:
        """Return net contact forces for the five Wuji fingertips."""

        if len(self._contact_sensors) != 5:
            raise RuntimeError("trajectory scene has no fingertip contact sensors")
        values = []
        for sensor in self._contact_sensors:
            matrix = sensor.data.net_forces_w
            if matrix is None or matrix.shape != (self.num_envs, 1, 3):
                raise RuntimeError(
                    "fingertip contact sensor net-force tensor has an unexpected shape"
                )
            values.append(matrix[:, 0].to(dtype=torch.float32))
        return torch.stack(values, dim=1)

    def _world_root_pose(self) -> Tensor:
        return torch.cat(
            (self.robot.data.root_pos_w, self.robot.data.root_quat_w), dim=-1
        )

    @torch.no_grad()
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
            raise ValueError("Isaac restore env_ids must be unique valid int64 [M]")
        count = int(env_ids.numel())
        if state.joint_position.shape != (count, 27):
            raise ValueError("Isaac restore state batch and env_ids disagree")
        if state.object_points_object_frame.shape != (
            count,
            self.trajectory.num_object_points,
            3,
        ):
            raise ValueError("Isaac restore object-point batch is invalid")
        q_lab = state.joint_position.index_select(1, self._lab_from_canonical)
        qd_lab = state.joint_velocity.index_select(1, self._lab_from_canonical)
        target_lab = state.previous_joint_target.index_select(
            1, self._lab_from_canonical
        )

        robot_world = state.robot_root_pose_env_local.clone()
        table_world = state.table_root_pose_env_local.clone()
        selected_origins = self.scene.env_origins.index_select(0, env_ids)
        robot_world[:, :3] += selected_origins
        table_world[:, :3] += selected_origins
        object_world = _compose_pose(robot_world, state.object_pose_robot_base)
        object_velocity_world = torch.cat(
            (
                _quat_apply(
                    robot_world[:, 3:],
                    state.object_linear_velocity_robot_base,
                ),
                _quat_apply(
                    robot_world[:, 3:],
                    state.object_angular_velocity_robot_base,
                ),
            ),
            dim=-1,
        )
        if self.object_physics_mode == "disabled":
            object_velocity_world.zero_()
        zeros = torch.zeros(count, 6, dtype=torch.float32, device=self.device)
        self.robot.write_root_pose_to_sim(robot_world, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(zeros, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_lab, qd_lab, env_ids=env_ids)
        self.robot.set_joint_position_target(target_lab, env_ids=env_ids)
        self.table.write_root_pose_to_sim(table_world, env_ids=env_ids)
        self.table.write_root_velocity_to_sim(zeros, env_ids=env_ids)
        self.object.write_root_pose_to_sim(object_world, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            object_velocity_world, env_ids=env_ids
        )
        self._target_lab.index_copy_(0, env_ids, target_lab)
        self._object_points_object_frame.index_copy_(
            0,
            env_ids,
            state.object_points_object_frame,
        )
        self.scene.write_data_to_sim()
        self.health_check()
        self.sim.forward()
        self.health_check()
        self.scene.update(0.0)
        state = self._live_state()
        self.health_check()
        return state

    @torch.no_grad()
    def step(self, canonical_joint_target: Tensor) -> LiveState:
        self.action_spec.validate(
            canonical_joint_target,
            name="canonical_joint_target",
            check_bounds=True,
        )
        if canonical_joint_target.shape[0] != self.num_envs:
            raise ValueError("Isaac command must cover every synchronous environment")
        self._target_lab.copy_(
            canonical_joint_target.index_select(1, self._lab_from_canonical)
        )
        for _ in range(self.decimation):
            self.robot.set_joint_position_target(
                self._target_lab, env_ids=self._env_ids
            )
            self.scene.write_data_to_sim()
            self.health_check()
            self.sim.step(render=self.render)
            self.health_check()
            self.scene.update(self.physics_dt)
        state = self._live_state()
        self.health_check()
        return state

    def _live_state(self) -> LiveState:
        joint_position = self.robot.data.joint_pos.index_select(
            1, self._canonical_from_lab
        ).to(dtype=torch.float32)
        joint_velocity = self.robot.data.joint_vel.index_select(
            1, self._canonical_from_lab
        ).to(dtype=torch.float32)
        robot_world = self._world_root_pose().to(dtype=torch.float32)
        object_world = torch.cat(
            (self.object.data.root_pos_w, self.object.data.root_quat_w), dim=-1
        ).to(dtype=torch.float32)
        object_pose_base = _relative_pose(robot_world, object_world)
        inverse_root = torch.cat((robot_world[:, 3:4], -robot_world[:, 4:]), dim=-1)
        linear_velocity = _quat_apply(
            inverse_root,
            self.object.data.root_lin_vel_w.to(dtype=torch.float32),
        )
        angular_velocity = _quat_apply(
            inverse_root,
            self.object.data.root_ang_vel_w.to(dtype=torch.float32),
        )
        points = self._object_points_object_frame
        surface = (
            _quat_apply(object_pose_base[:, None, 3:], points)
            + object_pose_base[:, None, :3]
        )

        body = self.robot.data.body_state_w[:, self._hand_keypoint_ids, :].to(
            dtype=torch.float32
        )
        rotated_offsets = _quat_apply(
            body[..., 3:7], self._hand_keypoint_offsets.unsqueeze(0)
        )
        keypoints_world = body[..., :3] + rotated_offsets
        keypoint_velocities_world = body[..., 7:10] + torch.cross(
            body[..., 10:13], rotated_offsets, dim=-1
        )
        keypoints_base = _quat_apply(
            inverse_root[:, None, :],
            keypoints_world - robot_world[:, None, :3],
        )
        keypoint_velocities_base = _quat_apply(
            inverse_root[:, None, :], keypoint_velocities_world
        )
        fingertip_stride = (
            WUJI_KEYPOINTS_PER_FINGER
            if self._requires_hand_kinematics
            else 4
        )
        fingertips_base = keypoints_base[:, fingertip_stride - 1 :: fingertip_stride]
        link7_state = self.robot.data.body_state_w[:, self._link7_id, :].to(
            dtype=torch.float32
        )
        link7_pose_base = _relative_pose(
            robot_world, torch.cat((link7_state[:, :3], link7_state[:, 3:7]), dim=-1)
        )
        link7_linear_velocity_base = _quat_apply(
            inverse_root, link7_state[:, 7:10]
        )
        link7_angular_velocity_base = _quat_apply(
            inverse_root, link7_state[:, 10:13]
        )
        applied_torque = getattr(self.robot.data, "applied_torque", None)
        if applied_torque is None:
            if self._requires_hand_kinematics:
                raise RuntimeError("ManipTrans reward requires Isaac applied_torque")
            joint_effort = torch.zeros_like(joint_position)
        else:
            joint_effort = applied_torque.index_select(
                1, self._canonical_from_lab
            ).to(dtype=torch.float32)
        return LiveState(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            link7_position_robot_base=link7_pose_base[:, :3],
            object_pose_robot_base=object_pose_base,
            object_linear_velocity_robot_base=linear_velocity,
            object_angular_velocity_robot_base=angular_velocity,
            object_surface_points_robot_base=surface.to(dtype=torch.float32),
            fingertip_positions_robot_base=fingertips_base.to(dtype=torch.float32),
            link7_orientation_robot_base=link7_pose_base[:, 3:],
            link7_linear_velocity_robot_base=link7_linear_velocity_base,
            link7_angular_velocity_robot_base=link7_angular_velocity_base,
            hand_keypoint_positions_robot_base=keypoints_base,
            hand_keypoint_linear_velocities_robot_base=(
                keypoint_velocities_base
            ),
            joint_effort=joint_effort,
        )


__all__ = [
    "IsaacTrajectoryBackend",
    "LiveState",
    "ROLLOUT_STATE_SCHEMA_VERSION",
    "TASK_OUTCOME_ROLLOUT_STATE_SCHEMA_VERSION",
    "TaskResetResult",
    "TaskStepResult",
    "TrajectoryPhysicsBackend",
    "TrajectoryDirectEnvCore",
    "action_spec_from_isaac_robot",
    "canonical_velocity_limits_from_isaac_robot",
]
