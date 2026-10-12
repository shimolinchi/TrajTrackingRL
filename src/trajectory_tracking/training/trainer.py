"""Compact PPO orchestration for asynchronous trajectory environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import torch

from ..environment.core import TaskStepResult, TrajectoryDirectEnvCore
from ..environment.action_smoothness import ActionSmoothnessMetricAccumulator
from ..environment.joint_acceleration import JointAccelerationAccumulator
from .ppo import PPOConfig, PPOTrainer, PPOUpdateStats, RolloutBuffer


@dataclass(frozen=True, slots=True)
class OverfitUpdateStats:
    ppo: PPOUpdateStats
    rollout_steps: int
    valid_transition_count: int
    robot_success_transition_count: int
    completed_episode_count: int
    episode_success_count: int
    endpoint_miss_episode_count: int
    object_departure_failure_episode_count: int
    maniptrans_failure_episode_count: int
    object_drop_or_escape_episode_count: int
    mean_reward: float
    mean_learner_reward: float
    mean_object_flow_quality: float
    mean_robot_joint_position_quality: float
    mean_joint_velocity_penalty: float
    mean_flow_error_m: float
    dynamic_object_flow_error_m: float | None
    object_motion_normalized_error: float | None
    object_motion_tracking_score: float | None
    flow_error_std_m: float
    max_flow_error_m: float
    final_flow_error_m: float
    episode_success_rate: float | None
    mean_arm_joint_position_rmse_rad: float
    mean_hand_joint_position_rmse_rad: float
    mean_joint_position_absolute_error_rad: float
    mean_link7_position_error_m: float
    mean_object_position_error_m: float
    dynamic_robot_keypoint_error_m: float | None
    robot_motion_normalized_error: float | None
    robot_motion_tracking_score: float | None
    object_reference_moving_fraction: float
    robot_reference_moving_fraction: float
    iteration_success_rate: float
    action_projection_fraction: float
    reset_overhead_fraction: float
    start_index_min: int
    start_index_mean: float
    start_index_max: int
    reference_cursor_min: int
    reference_cursor_mean: float
    reference_cursor_max: int
    episode_progress_mean: float
    action_smoothness_metrics: dict[str, float | int | None] | None = None
    joint_acceleration_metrics: dict[str, float | int] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationStats:
    episode_length_steps: int
    valid_transition_count: int
    robot_success_transition_count: int
    completed_episode_count: int
    episode_success_count: int
    endpoint_miss_episode_count: int
    object_departure_failure_episode_count: int
    maniptrans_failure_episode_count: int
    object_drop_or_escape_episode_count: int
    mean_reward: float
    mean_object_flow_quality: float
    mean_robot_joint_position_quality: float
    mean_joint_velocity_penalty: float
    mean_flow_error_m: float
    dynamic_object_flow_error_m: float | None
    object_motion_normalized_error: float | None
    object_motion_tracking_score: float | None
    flow_error_std_m: float
    max_flow_error_m: float
    final_flow_error_m: float
    episode_success_rate: float
    mean_arm_joint_position_rmse_rad: float
    mean_hand_joint_position_rmse_rad: float
    mean_joint_position_absolute_error_rad: float
    mean_link7_position_error_m: float
    mean_object_position_error_m: float
    dynamic_robot_keypoint_error_m: float | None
    robot_motion_normalized_error: float | None
    robot_motion_tracking_score: float | None
    object_reference_moving_fraction: float
    robot_reference_moving_fraction: float
    robot_success_fraction: float
    action_projection_fraction: float
    perturbation_mean_abs_delta_rad: float
    start_index_min: int
    start_index_mean: float
    start_index_max: int
    action_smoothness_metrics: dict[str, float | int | None] | None = None
    joint_acceleration_metrics: dict[str, float | int] | None = None


@dataclass(slots=True)
class _MetricAccumulator:
    action_smoothness: ActionSmoothnessMetricAccumulator
    joint_acceleration: JointAccelerationAccumulator
    reward: torch.Tensor
    object_flow_quality: torch.Tensor
    robot_joint_position_quality: torch.Tensor
    joint_velocity_penalty: torch.Tensor
    flow_error: torch.Tensor
    flow_error_squared: torch.Tensor
    flow_error_max: torch.Tensor
    arm_joint_position_rmse: torch.Tensor
    hand_joint_position_rmse: torch.Tensor
    joint_position_absolute_error: torch.Tensor
    link7_position_error: torch.Tensor
    object_position_error: torch.Tensor
    dynamic_object_flow_error_weighted: torch.Tensor
    object_reference_motion: torch.Tensor
    object_motion_error: torch.Tensor
    object_moving_samples: torch.Tensor
    dynamic_robot_keypoint_error_weighted: torch.Tensor
    robot_reference_motion: torch.Tensor
    robot_motion_error: torch.Tensor
    robot_moving_samples: torch.Tensor
    robot_success_steps: torch.Tensor
    projection: torch.Tensor
    final_flow_error: torch.Tensor
    last_flow_error: torch.Tensor
    valid_samples: torch.Tensor
    completed_episodes: torch.Tensor
    successful_episodes: torch.Tensor
    object_departure_failed_episodes: torch.Tensor
    maniptrans_failed_episodes: torch.Tensor
    object_departed_episodes: torch.Tensor
    reset_overhead_slots: torch.Tensor
    total_slots: torch.Tensor

    @classmethod
    def create(cls) -> _MetricAccumulator:
        zero = lambda: torch.zeros((), dtype=torch.float64)
        return cls(
            action_smoothness=ActionSmoothnessMetricAccumulator(),
            joint_acceleration=JointAccelerationAccumulator(),
            reward=zero(),
            object_flow_quality=zero(),
            robot_joint_position_quality=zero(),
            joint_velocity_penalty=zero(),
            flow_error=zero(),
            flow_error_squared=zero(),
            flow_error_max=zero(),
            arm_joint_position_rmse=zero(),
            hand_joint_position_rmse=zero(),
            joint_position_absolute_error=zero(),
            link7_position_error=zero(),
            object_position_error=zero(),
            dynamic_object_flow_error_weighted=zero(),
            object_reference_motion=zero(),
            object_motion_error=zero(),
            object_moving_samples=zero(),
            dynamic_robot_keypoint_error_weighted=zero(),
            robot_reference_motion=zero(),
            robot_motion_error=zero(),
            robot_moving_samples=zero(),
            robot_success_steps=zero(),
            projection=zero(),
            final_flow_error=zero(),
            last_flow_error=zero(),
            valid_samples=zero(),
            completed_episodes=zero(),
            successful_episodes=zero(),
            object_departure_failed_episodes=zero(),
            maniptrans_failed_episodes=zero(),
            object_departed_episodes=zero(),
            reset_overhead_slots=zero(),
            total_slots=zero(),
        )

    def add(self, transition: TaskStepResult) -> None:
        if transition.joint_acceleration_diagnostics is not None:
            self.joint_acceleration.add(transition.joint_acceleration_diagnostics,
                                        transition.tracking.reward)
        if transition.action_smoothness_diagnostics is not None:
            self.action_smoothness.add(transition.action_smoothness_diagnostics,
                                       transition.tracking.reward)
        tracking = transition.tracking
        action = transition.action
        valid = transition.valid_transition
        valid_float = valid.to(dtype=torch.float64)

        def masked_sum(value: torch.Tensor) -> torch.Tensor:
            return (value.double() * valid_float).sum().cpu()

        self.reward += masked_sum(transition.reward)
        self.object_flow_quality += masked_sum(tracking.object_flow_quality)
        self.robot_joint_position_quality += masked_sum(
            tracking.robot_joint_position_quality
        )
        self.joint_velocity_penalty += masked_sum(
            tracking.joint_velocity_penalty
        )
        self.flow_error += masked_sum(tracking.mean_point_error_m)
        self.flow_error_squared += masked_sum(tracking.mean_point_error_m.square())
        self.arm_joint_position_rmse += masked_sum(tracking.arm_joint_position_rmse_rad)
        self.hand_joint_position_rmse += masked_sum(
            tracking.hand_joint_position_rmse_rad
        )
        self.joint_position_absolute_error += masked_sum(
            transition.joint_position_tracking.mean_absolute_error_rad
        )
        self.link7_position_error += masked_sum(
            transition.link7_position_tracking.position_error_m
        )
        self.object_position_error += masked_sum(
            transition.object_departure_tracking.position_error_m
        )
        motion = transition.motion_conditioned_tracking
        object_moving = valid & motion.object_reference_moving
        object_motion_weight = torch.where(
            object_moving,
            motion.reference_object_motion_m,
            torch.zeros_like(motion.reference_object_motion_m),
        )
        self.dynamic_object_flow_error_weighted += (
            motion.object_flow_error_m.double()
            * object_motion_weight.double()
        ).sum().cpu()
        self.object_reference_motion += object_motion_weight.double().sum().cpu()
        self.object_motion_error += (
            motion.object_motion_error_m.double()
            * object_moving.to(dtype=torch.float64)
        ).sum().cpu()
        self.object_moving_samples += object_moving.double().sum().cpu()

        robot_moving = valid & motion.robot_reference_moving
        robot_motion_weight = torch.where(
            robot_moving,
            motion.reference_robot_keypoint_motion_m,
            torch.zeros_like(motion.reference_robot_keypoint_motion_m),
        )
        self.dynamic_robot_keypoint_error_weighted += (
            motion.robot_keypoint_position_error_m.double()
            * robot_motion_weight.double()
        ).sum().cpu()
        self.robot_reference_motion += robot_motion_weight.double().sum().cpu()
        self.robot_motion_error += (
            motion.robot_keypoint_motion_error_m.double()
            * robot_moving.to(dtype=torch.float64)
        ).sum().cpu()
        self.robot_moving_samples += robot_moving.double().sum().cpu()
        self.robot_success_steps += (
            (tracking.within_robot_success_threshold & valid).double().sum().cpu()
        )
        self.projection += (
            (action.projection_mask.any(dim=-1) & valid).double().sum().cpu()
        )
        self.valid_samples += valid.double().sum().cpu()
        if valid.any().item():
            valid_errors = tracking.mean_point_error_m[valid].double()
            self.flow_error_max = torch.maximum(
                self.flow_error_max,
                valid_errors.max().cpu(),
            )
            self.last_flow_error = valid_errors.mean().cpu()
        completed = transition.episode_completed
        self.completed_episodes += completed.double().sum().cpu()
        self.successful_episodes += transition.episode_success.double().sum().cpu()
        self.object_departure_failed_episodes += (
            transition.object_departure_failure.double().sum().cpu()
        )
        self.maniptrans_failed_episodes += (
            transition.maniptrans_failure.double().sum().cpu()
        )
        self.object_departed_episodes += (
            transition.episode_object_departed.double().sum().cpu()
        )
        if completed.any().item():
            self.final_flow_error += (
                tracking.mean_point_error_m[completed].double().sum().cpu()
            )
        self.reset_overhead_slots += (~valid).double().sum().cpu()
        self.total_slots += torch.tensor(valid.numel(), dtype=torch.float64)

    def flow_error_std(self) -> float:
        samples = float(self.valid_samples)
        if samples <= 0:
            raise RuntimeError("metrics contain no valid transitions")
        mean = self.flow_error / samples
        variance = (self.flow_error_squared / samples - mean.square()).clamp_min(0.0)
        return float(variance.sqrt())


def _motion_conditioned_metrics(
    metrics: _MetricAccumulator,
    *,
    valid_samples: float,
) -> dict[str, float | None]:
    """Finalize static-prefix-resistant metrics without inventing zero scores."""

    def normalized_metrics(
        *,
        weighted_position_error: torch.Tensor,
        reference_motion: torch.Tensor,
        motion_error: torch.Tensor,
    ) -> tuple[float | None, float | None, float | None]:
        denominator = float(reference_motion)
        if denominator <= 0.0:
            return None, None, None
        dynamic_position_error = float(weighted_position_error / reference_motion)
        normalized_motion_error = float(motion_error / reference_motion)
        tracking_score = max(0.0, 1.0 - normalized_motion_error)
        return dynamic_position_error, normalized_motion_error, tracking_score

    (
        dynamic_object_flow_error,
        object_motion_normalized_error,
        object_motion_tracking_score,
    ) = normalized_metrics(
        weighted_position_error=metrics.dynamic_object_flow_error_weighted,
        reference_motion=metrics.object_reference_motion,
        motion_error=metrics.object_motion_error,
    )
    (
        dynamic_robot_keypoint_error,
        robot_motion_normalized_error,
        robot_motion_tracking_score,
    ) = normalized_metrics(
        weighted_position_error=metrics.dynamic_robot_keypoint_error_weighted,
        reference_motion=metrics.robot_reference_motion,
        motion_error=metrics.robot_motion_error,
    )
    return {
        "dynamic_object_flow_error_m": dynamic_object_flow_error,
        "object_motion_normalized_error": object_motion_normalized_error,
        "object_motion_tracking_score": object_motion_tracking_score,
        "dynamic_robot_keypoint_error_m": dynamic_robot_keypoint_error,
        "robot_motion_normalized_error": robot_motion_normalized_error,
        "robot_motion_tracking_score": robot_motion_tracking_score,
        "object_reference_moving_fraction": float(
            metrics.object_moving_samples / valid_samples
        ),
        "robot_reference_moving_fraction": float(
            metrics.robot_moving_samples / valid_samples
        ),
    }


def _check_transition_contract(transition: TaskStepResult) -> None:
    if transition.truncated.any().item():
        raise RuntimeError("reference trajectory episodes must not be truncated")
    expected_completion = (
        transition.reached_endpoint
        | transition.object_departure_failure
        | transition.maniptrans_failure
    )
    if not torch.equal(transition.episode_completed, expected_completion):
        raise RuntimeError("task completion disagrees with endpoint/failure causes")
    if not torch.equal(transition.terminated, transition.episode_completed):
        raise RuntimeError("PPO termination disagrees with episode completion")
    if (transition.episode_completed & ~transition.valid_transition).any().item():
        raise RuntimeError("an invalid reset slot cannot complete an episode")
    if (transition.episode_success & ~transition.episode_completed).any().item():
        raise RuntimeError("success is only defined for completed environments")
    if (transition.maniptrans_failure & ~transition.valid_transition).any().item():
        raise RuntimeError("an invalid reset slot cannot fail ManipTrans imitation")
    if (
        transition.object_departure_failure & ~transition.valid_transition
    ).any().item():
        raise RuntimeError("an invalid reset slot cannot fail object tracking")
    if (
        transition.object_departure_failure & transition.maniptrans_failure
    ).any().item():
        raise RuntimeError("failure precedence must assign one terminating cause")
    if (transition.maniptrans_failure & transition.episode_success).any().item():
        raise RuntimeError("ManipTrans failure cannot also be episode success")
    if (transition.pre_roll_mask & transition.valid_transition).any().item():
        raise RuntimeError("reset pre-roll was incorrectly included in PPO")
    if (transition.reset_started & transition.valid_transition).any().item():
        raise RuntimeError("a reset slot was incorrectly included in PPO")


def learner_reward_for_transition(
    transition: TaskStepResult,
    config: PPOConfig,
) -> torch.Tensor:
    """Map the public tracking score to PPO reward and penalize failed ends."""

    episode_failed = transition.episode_completed & ~transition.episode_success
    return (
        transition.reward * config.learner_reward_scale
        - episode_failed.to(dtype=transition.reward.dtype)
        * config.learner_failure_penalty
    )


def _start_statistics(env: TrajectoryDirectEnvCore) -> tuple[int, float, int]:
    if env.episode is None:
        raise RuntimeError("environment has no sampled start indices")
    starts = env.episode.start_indices
    return int(starts.min()), float(starts.float().mean()), int(starts.max())


def _episode_outcome_counts(
    metrics: _MetricAccumulator,
) -> tuple[int, int, int, int, int]:
    completed = int(metrics.completed_episodes)
    successful = int(metrics.successful_episodes)
    object_departure_failed = int(metrics.object_departure_failed_episodes)
    maniptrans_failed = int(metrics.maniptrans_failed_episodes)
    endpoint_missed = (
        completed - successful - object_departure_failed - maniptrans_failed
    )
    if endpoint_missed < 0:
        raise RuntimeError("episode outcome counts exceed completed episodes")
    return (
        completed,
        successful,
        endpoint_missed,
        object_departure_failed,
        maniptrans_failed,
    )


@torch.no_grad()
def evaluate_episode(
    env: TrajectoryDirectEnvCore,
    policy: Callable[[torch.Tensor], torch.Tensor],
    *,
    start_indices: torch.Tensor | None = None,
    trajectory_indices: torch.Tensor | None = None,
    observation_callback: Callable[[TrajectoryDirectEnvCore], None] | None = None,
    metric_context=None,
) -> EvaluationStats:
    """Evaluate exactly one episode per environment under an optional schedule."""

    if observation_callback is not None and not callable(observation_callback):
        raise TypeError("observation_callback must be callable or None")
    reset = (
        env.reset()
        if start_indices is None and trajectory_indices is None
        else env.reset(
            start_indices=start_indices,
            trajectory_indices=trajectory_indices,
        )
    )
    if observation_callback is not None:
        observation_callback(env)
    observation = reset.observation
    start_min, start_mean, start_max = _start_statistics(env)
    metrics = _MetricAccumulator.create()
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps = 0
    while not completed.all().item():
        transition = env.step(policy(observation), auto_reset=False)
        _check_transition_contract(transition)
        if (completed & transition.episode_completed).any().item():
            raise RuntimeError("evaluation completed one environment more than once")
        completed |= transition.episode_completed
        metrics.add(transition)
        steps += 1
        if steps > env.trajectory.frame_count:
            raise RuntimeError("evaluation exceeded the reference trajectory horizon")
        if observation_callback is not None:
            observation_callback(env)
        observation = transition.next_observation
    samples = float(metrics.valid_samples)
    completed_count = float(metrics.completed_episodes)
    if samples <= 0 or completed_count != env.num_envs:
        raise RuntimeError("evaluation completed without one result per environment")
    if metric_context is not None:
        from .parallel.metrics import merge_rollout_metrics
        merge_rollout_metrics(metrics, metric_context)
        samples = float(metrics.valid_samples)
        completed_count = float(metrics.completed_episodes)
        start_min = int(metric_context.reduce(torch.tensor(start_min), "min"))
        start_max = int(metric_context.reduce(torch.tensor(start_max), "max"))
        start_mean = float(metric_context.reduce(torch.tensor(start_mean * env.num_envs,
                                                             dtype=torch.float64))) / (
            env.num_envs * metric_context.world_size)
        steps = int(metric_context.reduce(torch.tensor(steps), "max"))
    (
        completed,
        successful,
        endpoint_missed,
        object_departure_failed,
        maniptrans_failed,
    ) = _episode_outcome_counts(metrics)
    composition = reset.perturbation.plan.action_composition
    perturbation_delta = (
        torch.zeros((), dtype=torch.float64)
        if composition is None
        else composition.applied_delta_rad.abs().double().mean().cpu()
    )
    if metric_context is not None:
        perturbation_delta = metric_context.reduce(perturbation_delta) / metric_context.world_size
    return EvaluationStats(
        action_smoothness_metrics=metrics.action_smoothness.result(),
        joint_acceleration_metrics=metrics.joint_acceleration.result(),
        episode_length_steps=steps,
        valid_transition_count=int(metrics.valid_samples),
        robot_success_transition_count=int(metrics.robot_success_steps),
        completed_episode_count=completed,
        episode_success_count=successful,
        endpoint_miss_episode_count=endpoint_missed,
        object_departure_failure_episode_count=object_departure_failed,
        maniptrans_failure_episode_count=maniptrans_failed,
        object_drop_or_escape_episode_count=int(metrics.object_departed_episodes),
        mean_reward=float(metrics.reward / samples),
        mean_object_flow_quality=float(metrics.object_flow_quality / samples),
        mean_robot_joint_position_quality=float(
            metrics.robot_joint_position_quality / samples
        ),
        mean_joint_velocity_penalty=float(
            metrics.joint_velocity_penalty / samples
        ),
        mean_flow_error_m=float(metrics.flow_error / samples),
        flow_error_std_m=metrics.flow_error_std(),
        max_flow_error_m=float(metrics.flow_error_max),
        final_flow_error_m=float(metrics.final_flow_error / completed_count),
        episode_success_rate=float(metrics.successful_episodes / completed_count),
        mean_arm_joint_position_rmse_rad=float(
            metrics.arm_joint_position_rmse / samples
        ),
        mean_hand_joint_position_rmse_rad=float(
            metrics.hand_joint_position_rmse / samples
        ),
        mean_joint_position_absolute_error_rad=float(
            metrics.joint_position_absolute_error / samples
        ),
        mean_link7_position_error_m=float(metrics.link7_position_error / samples),
        mean_object_position_error_m=float(metrics.object_position_error / samples),
        robot_success_fraction=float(metrics.robot_success_steps / samples),
        action_projection_fraction=float(metrics.projection / samples),
        perturbation_mean_abs_delta_rad=float(perturbation_delta),
        start_index_min=start_min,
        start_index_mean=start_mean,
        start_index_max=start_max,
        **_motion_conditioned_metrics(metrics, valid_samples=samples),
    )


class PPOOverfitRunner:
    """Collect fixed-size rollouts while environments reset independently."""

    def __init__(
        self,
        env: TrajectoryDirectEnvCore,
        trainer: PPOTrainer,
        *,
        observation_callback: Callable[[TrajectoryDirectEnvCore], None] | None = None,
    ) -> None:
        if trainer.model.observation_dim != env.config.observation.feature_dim:
            raise ValueError("trainer and task observation widths disagree")
        if observation_callback is not None and not callable(observation_callback):
            raise TypeError("observation_callback must be callable or None")
        self.env = env
        self.trainer = trainer
        self.observation_callback = observation_callback

    def run_update(self) -> OverfitUpdateStats:
        started = time.perf_counter()
        execution = getattr(self.trainer, "execution", None)
        if self.env.needs_reset:
            observation = self.env.reset().observation
            if self.observation_callback is not None:
                self.observation_callback(self.env)
        else:
            observation = self.env.current_observation()
        rollout_steps = self.trainer.config.rollout_steps_per_env
        buffer = RolloutBuffer(
            steps=rollout_steps,
            num_envs=self.env.num_envs,
            observation_dim=self.trainer.model.observation_dim,
            action_dim=self.trainer.model.action_dim,
            device=self.env.device,
            environment_major=False,
        )
        metrics = _MetricAccumulator.create()
        for _ in range(rollout_steps):
            raw_action, pre_tanh, log_prob, value = self.trainer.act(observation)
            transition = self.env.step(raw_action, auto_reset=True)
            _check_transition_contract(transition)
            learner_reward = learner_reward_for_transition(
                transition,
                self.trainer.config,
            )
            buffer.add(
                observation=observation,
                pre_tanh_action=pre_tanh,
                old_log_prob=log_prob,
                old_value=value,
                # Metrics retain the public 0--100 tracking score.  PPO uses
                # a scaled score plus an explicit failed-terminal penalty.
                reward=learner_reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                valid=transition.valid_transition,
            )
            observation = transition.next_observation
            metrics.add(transition)
            if self.observation_callback is not None:
                self.observation_callback(self.env)
        last_value = torch.where(
            self.env.active_environment_mask,
            self.trainer.value(observation),
            torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device),
        )
        batch = buffer.finish(
            last_value=last_value,
            gamma=self.trainer.config.gamma,
            gae_lambda=self.trainer.config.gae_lambda,
            **({"allow_empty": True} if execution is not None else {}),
        )
        collected = time.perf_counter()
        observer = getattr(self, "memory_observer", None)
        if observer is not None:
            observer("rollout_complete", local_valid_n=batch.batch_size)
        if execution is not None:
            execution.set_rollout_order(buffer.valid[:buffer.cursor],
                                       environment_major=buffer.environment_major)
        update = self.trainer.update(batch)
        self.last_timing = {"rollout_seconds": collected - started,
                            "ppo_seconds": time.perf_counter() - collected}
        if execution is not None:
            from .parallel.metrics import merge_rollout_metrics
            merge_rollout_metrics(metrics, execution.context)
        samples = float(metrics.valid_samples)
        if samples <= 0:
            raise RuntimeError("PPO rollout completed without valid transitions")
        completed = float(metrics.completed_episodes)
        (
            completed_count,
            successful_count,
            endpoint_miss_count,
            object_departure_failure_count,
            maniptrans_failure_count,
        ) = _episode_outcome_counts(metrics)
        start_min, start_mean, start_max = _start_statistics(self.env)
        cursors = self.env.reference_cursors
        assert self.env.episode is not None
        progress = (
            self.env.episode_steps.float() / self.env.episode.episode_lengths.float()
        ).clamp(0.0, 1.0)
        final_error = (
            metrics.final_flow_error / completed
            if completed > 0
            else metrics.last_flow_error
        )
        return OverfitUpdateStats(
            ppo=update,
            action_smoothness_metrics=metrics.action_smoothness.result(),
            joint_acceleration_metrics=metrics.joint_acceleration.result(),
            rollout_steps=buffer.cursor,
            valid_transition_count=int(metrics.valid_samples),
            robot_success_transition_count=int(metrics.robot_success_steps),
            completed_episode_count=completed_count,
            episode_success_count=successful_count,
            endpoint_miss_episode_count=endpoint_miss_count,
            object_departure_failure_episode_count=(
                object_departure_failure_count
            ),
            maniptrans_failure_episode_count=maniptrans_failure_count,
            object_drop_or_escape_episode_count=int(
                metrics.object_departed_episodes
            ),
            mean_reward=float(metrics.reward / samples),
            mean_learner_reward=float(
                metrics.reward / samples * self.trainer.config.learner_reward_scale
                - (completed_count - successful_count)
                / samples
                * self.trainer.config.learner_failure_penalty
            ),
            mean_object_flow_quality=float(metrics.object_flow_quality / samples),
            mean_robot_joint_position_quality=float(
                metrics.robot_joint_position_quality / samples
            ),
            mean_joint_velocity_penalty=float(
                metrics.joint_velocity_penalty / samples
            ),
            mean_flow_error_m=float(metrics.flow_error / samples),
            flow_error_std_m=metrics.flow_error_std(),
            max_flow_error_m=float(metrics.flow_error_max),
            final_flow_error_m=float(final_error),
            episode_success_rate=(
                float(metrics.successful_episodes / completed)
                if completed > 0
                else None
            ),
            mean_arm_joint_position_rmse_rad=float(
                metrics.arm_joint_position_rmse / samples
            ),
            mean_hand_joint_position_rmse_rad=float(
                metrics.hand_joint_position_rmse / samples
            ),
            mean_joint_position_absolute_error_rad=float(
                metrics.joint_position_absolute_error / samples
            ),
            mean_link7_position_error_m=float(metrics.link7_position_error / samples),
            mean_object_position_error_m=float(
                metrics.object_position_error / samples
            ),
            iteration_success_rate=float(metrics.robot_success_steps / samples),
            action_projection_fraction=float(metrics.projection / samples),
            reset_overhead_fraction=float(
                metrics.reset_overhead_slots / metrics.total_slots
            ),
            start_index_min=start_min,
            start_index_mean=start_mean,
            start_index_max=start_max,
            reference_cursor_min=int(cursors.min()),
            reference_cursor_mean=float(cursors.float().mean()),
            reference_cursor_max=int(cursors.max()),
            episode_progress_mean=float(progress.mean()),
            **_motion_conditioned_metrics(metrics, valid_samples=samples),
        )


__all__ = [
    "EvaluationStats",
    "OverfitUpdateStats",
    "PPOOverfitRunner",
    "learner_reward_for_transition",
    "evaluate_episode",
]
