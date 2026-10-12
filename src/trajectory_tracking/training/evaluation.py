"""Reward-independent, per-metric trajectory-policy evaluation."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from math import isfinite
from typing import Callable, Sequence

import torch
from torch import Tensor

from ..environment.core import TrajectoryDirectEnvCore
from ..environment.joint_acceleration import JointAccelerationAccumulator


@dataclass(frozen=True, slots=True)
class IndependentMetric:
    """One independently ranked metric; lower values are always better."""

    key: str
    stats_field: str
    label: str
    unit: str
    display_scale: float
    decimal_places: int

    def __post_init__(self) -> None:
        if not self.key or not self.stats_field or not self.label or not self.unit:
            raise ValueError("independent metric text fields must be non-empty")
        if not isfinite(self.display_scale) or self.display_scale <= 0.0:
            raise ValueError("independent metric display scale must be positive")
        if self.decimal_places < 0:
            raise ValueError("independent metric decimal places must be non-negative")

    def raw_value(self, stats: IndependentEpisodeStats) -> float:
        value = float(getattr(stats, self.stats_field))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"invalid value for metric {self.key}: {value}")
        return value

    def display_value(self, stats: IndependentEpisodeStats) -> float:
        return self.display_scale * self.raw_value(stats)

    def format_value(self, stats: IndependentEpisodeStats) -> str:
        return f"{self.display_value(stats):.{self.decimal_places}f}"

    def to_manifest(self) -> dict[str, object]:
        return {
            "key": self.key,
            "stats_field": self.stats_field,
            "label": self.label,
            "unit": self.unit,
            "display_scale": self.display_scale,
            "decimal_places": self.decimal_places,
            "direction": "lower_is_better",
        }


INDEPENDENT_METRICS = (
    IndependentMetric(
        key="link7_mean_position_error",
        stats_field="mean_link7_position_error_m",
        label="机械臂末端平均位置误差",
        unit="mm",
        display_scale=1000.0,
        decimal_places=1,
    ),
    IndependentMetric(
        key="link7_max_position_error",
        stats_field="mean_episode_max_link7_position_error_m",
        label="机械臂末端最大位置误差",
        unit="mm",
        display_scale=1000.0,
        decimal_places=1,
    ),
    IndependentMetric(
        key="hand_joint_position_rmse",
        stats_field="mean_hand_position_rmse_rad",
        label="手指关节位置 RMSE",
        unit="rad",
        display_scale=1.0,
        decimal_places=4,
    ),
)


@dataclass(frozen=True, slots=True)
class IndependentEpisodeStats:
    """Robot metrics plus fairness metadata and optional acceleration diagnostics."""

    evaluation_num_envs: int
    valid_transition_count: int
    start_indices_sha256: str
    start_index_min: int
    start_index_mean: float
    start_index_max: int
    completion_rate: float
    mean_link7_position_error_m: float
    mean_episode_max_link7_position_error_m: float
    mean_hand_position_rmse_rad: float
    joint_acceleration_metrics: dict[str, float | int] | None = None

    def __post_init__(self) -> None:
        if self.evaluation_num_envs <= 0 or self.valid_transition_count <= 0:
            raise ValueError("evaluation counts must be positive")
        if len(self.start_indices_sha256) != 64:
            raise ValueError("start-index SHA-256 must contain 64 characters")
        if not 0.0 <= self.completion_rate <= 1.0:
            raise ValueError("completion rate must be in [0,1]")
        for metric in INDEPENDENT_METRICS:
            metric.raw_value(self)


@dataclass(frozen=True, slots=True)
class IndependentEnvironmentDiagnostic:
    """Per-environment audit row excluded from cross-checkpoint ranking."""

    environment_index: int
    trajectory_index: int
    start_index: int
    valid_transition_count: int
    episode_success: bool
    object_departed: bool
    mean_object_flow_error_m: float
    final_object_flow_error_m: float
    mean_link7_position_error_m: float
    max_link7_position_error_m: float
    mean_hand_position_rmse_rad: float
    object_lifted_2cm: bool
    first_object_lift_step: int | None
    maximum_object_lift_m: float

    def __post_init__(self) -> None:
        if self.environment_index < 0 or self.trajectory_index < 0:
            raise ValueError("environment and trajectory indices must be non-negative")
        if self.start_index < 0 or self.valid_transition_count <= 0:
            raise ValueError("episode index/count diagnostics are invalid")
        for name in (
            "mean_object_flow_error_m",
            "final_object_flow_error_m",
            "mean_link7_position_error_m",
            "max_link7_position_error_m",
            "mean_hand_position_rmse_rad",
            "maximum_object_lift_m",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"invalid environment diagnostic {name}: {value}")
        if self.object_lifted_2cm != (self.maximum_object_lift_m >= 0.02):
            raise ValueError("2 cm lift flag disagrees with maximum object lift")
        if self.object_lifted_2cm:
            if self.first_object_lift_step is None or self.first_object_lift_step <= 0:
                raise ValueError("lifted object requires a positive first-lift step")
        elif self.first_object_lift_step is not None:
            raise ValueError("non-lifted object cannot have a first-lift step")


@dataclass(frozen=True, slots=True)
class IndependentCandidateResult:
    label: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_update: int
    training_num_envs: int
    evaluation_seconds: float
    metrics: IndependentEpisodeStats

    def __post_init__(self) -> None:
        if (
            not self.label
            or not self.checkpoint_path
            or len(self.checkpoint_sha256) != 64
        ):
            raise ValueError("evaluation candidate identity is invalid")
        if self.checkpoint_update < 0 or self.training_num_envs <= 0:
            raise ValueError("evaluation candidate checkpoint metadata is invalid")
        if not isfinite(self.evaluation_seconds) or self.evaluation_seconds < 0.0:
            raise ValueError("evaluation candidate time is invalid")


@dataclass(frozen=True, slots=True)
class MetricRankedCandidate:
    rank: int
    result: IndependentCandidateResult
    raw_value: float


@dataclass(frozen=True, slots=True)
class IndependentMetricRanking:
    metric: IndependentMetric
    candidates: tuple[MetricRankedCandidate, ...]


def independent_metrics_manifest() -> list[dict[str, object]]:
    return [metric.to_manifest() for metric in INDEPENDENT_METRICS]


def policy_evaluation_contract(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Select fields that can affect deterministic policy rollout semantics.

    Reward, termination, optimizer, and learner-return fields are deliberately
    excluded because independent endpoint evaluation neither ranks nor censors
    a policy with them. Observation/action/dynamics identity and actor-network
    construction remain strict.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("policy evaluation manifest must be a mapping")

    def required(name: str) -> object:
        if name not in manifest:
            raise ValueError(f"policy evaluation manifest lacks {name}")
        return manifest[name]

    ppo = required("ppo")
    if not isinstance(ppo, Mapping):
        raise ValueError("policy evaluation manifest ppo must be a mapping")
    actor_fields = (
        "algorithm",
        "implementation_version",
        "activation",
        "hidden_dims",
    )
    missing_actor_fields = [name for name in actor_fields if name not in ppo]
    if missing_actor_fields:
        raise ValueError(
            "policy evaluation manifest lacks actor fields: " f"{missing_actor_fields}"
        )
    return {
        "schema_version": required("schema_version"),
        "task": required("task"),
        "environment_version": required("environment_version"),
        "seed": required("seed"),
        "data": required("data"),
        "timing": required("timing"),
        "start_index": required("start_index"),
        "perturbation": required("perturbation"),
        "direct_action": required("direct_action"),
        "recorded_action_usage": required("recorded_action_usage"),
        "observation": required("observation"),
        "actor": {name: ppo[name] for name in actor_fields},
    }


def verify_historical_policy_evaluation_lineage(
    *,
    current_manifest: Mapping[str, object],
    training_manifest: Mapping[str, object],
) -> str:
    """Verify a frozen training manifest before relaxing only its full hash.

    Returns the exact historical compatibility hash that the checkpoint must
    contain. Any policy-rollout contract difference fails before simulator
    startup; only fields excluded by :func:`policy_evaluation_contract` may
    differ.
    """

    training_hash = training_manifest.get("compatibility_hash")
    if (
        not isinstance(training_hash, str)
        or len(training_hash) != 64
        or any(character not in "0123456789abcdef" for character in training_hash)
    ):
        raise ValueError(
            "historical training manifest requires a lowercase SHA-256 "
            "compatibility_hash"
        )
    current_contract = policy_evaluation_contract(current_manifest)
    training_contract = policy_evaluation_contract(training_manifest)
    mismatches = tuple(
        name
        for name in current_contract
        if current_contract[name] != training_contract[name]
    )
    if mismatches:
        raise ValueError(
            "historical checkpoint policy-evaluation contract mismatch: "
            f"{mismatches}"
        )
    return training_hash


def _start_indices_sha256(start_indices: Tensor) -> str:
    value = start_indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return sha256(value.numpy().tobytes()).hexdigest()


def _masked_add_(destination: Tensor, value: Tensor, mask: Tensor) -> None:
    destination.add_(
        torch.where(mask, value, torch.zeros_like(value)).to(torch.float64)
    )


@torch.no_grad()
def evaluate_independent_metrics(
    env: TrajectoryDirectEnvCore,
    policy: Callable[[Tensor], Tensor],
    *,
    start_indices: Tensor | None = None,
    trajectory_indices: Tensor | None = None,
    observation_callback: Callable[[TrajectoryDirectEnvCore], None] | None = None,
    rollout_start_callback: Callable[[], None] | None = None,
    metric_environment_count: int | None = None,
    diagnostics_callback: (
        Callable[[tuple[IndependentEnvironmentDiagnostic, ...]], None] | None
    ) = None,
) -> IndependentEpisodeStats:
    """Run one full reference suffix per environment and aggregate three metrics.

    Mean errors are averaged over valid transitions within an episode first,
    then averaged equally across episodes. Maximum link7 error is computed
    within each episode first and likewise averaged across episodes. No
    training reward, success threshold, terminal penalty, object state, or
    weighted score participates in robot-metric ranking. Optional acceleration
    diagnostics include unpenalized reward solely for explicit calibration.
    """

    if env.episode_completion_mode != "reference_endpoint_only":
        raise ValueError(
            "independent evaluation requires "
            "episode_completion_mode=reference_endpoint_only"
        )
    if not callable(policy):
        raise TypeError("independent evaluation policy must be callable")
    metric_count = env.num_envs if metric_environment_count is None else metric_environment_count
    if not 1 <= metric_count <= env.num_envs:
        raise ValueError("metric environment prefix must fit the physical batch")
    acceleration_metrics = JointAccelerationAccumulator(percentiles=True)
    if observation_callback is not None and not callable(observation_callback):
        raise TypeError("observation callback must be callable or None")
    if rollout_start_callback is not None and not callable(rollout_start_callback):
        raise TypeError("rollout start callback must be callable or None")
    if diagnostics_callback is not None and not callable(diagnostics_callback):
        raise TypeError("diagnostics callback must be callable or None")

    reset = env.reset(
        start_indices=start_indices,
        trajectory_indices=trajectory_indices,
        start_index_range="full_reference",
    )
    if observation_callback is not None:
        observation_callback(env)
    if rollout_start_callback is not None:
        rollout_start_callback()
    observation = reset.observation
    initial_state = env.live_state
    if initial_state is None:
        raise RuntimeError("evaluation environment lacks reset live state")
    initial_object_z = initial_state.object_pose_robot_base[:, 2].detach().clone()
    starts = reset.episode.start_indices.detach().clone()
    trajectories = reset.episode.trajectory_indices
    if trajectories is None:
        trajectories = torch.zeros_like(starts)
    else:
        trajectories = trajectories.detach().clone()
    count = env.num_envs
    device = env.device
    completed = torch.zeros(count, dtype=torch.bool, device=device)
    step_count = torch.zeros(count, dtype=torch.float64, device=device)
    hand_error_sum = torch.zeros(count, dtype=torch.float64, device=device)
    link7_error_sum = torch.zeros(count, dtype=torch.float64, device=device)
    link7_error_max = torch.zeros(count, dtype=torch.float64, device=device)
    flow_error_sum = torch.zeros(count, dtype=torch.float64, device=device)
    final_flow_error = torch.zeros(count, dtype=torch.float64, device=device)
    episode_success = torch.zeros(count, dtype=torch.bool, device=device)
    object_departed = torch.zeros(count, dtype=torch.bool, device=device)
    maximum_object_lift = torch.zeros(count, dtype=torch.float64, device=device)
    first_object_lift_step = torch.full(
        (count,),
        -1,
        dtype=torch.int64,
        device=device,
    )

    hand_indices = env.action_spec.groups[1].indices
    global_steps = 0
    while not completed.all().item():
        episode = env.episode
        if episode is None:
            raise RuntimeError("evaluation environment lost its reference episode")
        # Completed rows remain in the synchronized batch while longer suffixes
        # finish. Clamp those rows to the final legal transition; ``valid`` masks
        # them below. ReferenceEpisodeBatch performs the per-row trajectory-bank
        # gather, unlike indexing the collection tensor as a single trajectory.
        target_cursors = env.reference_cursors.clamp_max(
            episode.terminal_cursors - 1
        )
        reference_targets = episode.step_targets(target_cursors)
        transition = env.step(policy(observation), auto_reset=False)
        if transition.joint_acceleration_diagnostics is not None:
            acceleration_metrics.add(transition.joint_acceleration_diagnostics.prefix(metric_count),
                                     transition.tracking.reward[:metric_count])
        valid = transition.valid_transition
        if (valid & completed).any().item():
            raise RuntimeError(
                "completed environment produced another valid transition"
            )
        state = env.live_state
        if state is None:
            raise RuntimeError("evaluation environment lost its live state")
        q_error = state.joint_position - reference_targets.next_joint_position
        hand_error = q_error[:, hand_indices].square().mean(-1).sqrt()
        link7_error = torch.linalg.vector_norm(
            state.link7_position_robot_base
            - reference_targets.next_link7_position_robot_base,
            dim=-1,
        )
        if (
            not torch.isfinite(hand_error).all().item()
            or not torch.isfinite(link7_error).all().item()
        ):
            raise RuntimeError(
                "independent evaluation metric contains non-finite values"
            )

        step_count.add_(valid.to(torch.float64))
        object_lift = state.object_pose_robot_base[:, 2] - initial_object_z
        maximum_object_lift = torch.maximum(
            maximum_object_lift,
            torch.where(valid, object_lift, torch.zeros_like(object_lift)).to(
                torch.float64
            ),
        )
        first_lift = valid & (first_object_lift_step < 0) & (object_lift >= 0.02)
        first_object_lift_step = torch.where(
            first_lift,
            step_count.to(torch.int64),
            first_object_lift_step,
        )
        _masked_add_(hand_error_sum, hand_error, valid)
        _masked_add_(link7_error_sum, link7_error, valid)
        _masked_add_(flow_error_sum, transition.tracking.mean_point_error_m, valid)
        link7_error_max = torch.maximum(
            link7_error_max,
            torch.where(valid, link7_error, torch.zeros_like(link7_error)).to(
                torch.float64
            ),
        )
        final_flow_error = torch.where(
            transition.episode_completed,
            transition.tracking.mean_point_error_m.to(torch.float64),
            final_flow_error,
        )
        episode_success |= transition.episode_success
        object_departed |= transition.episode_object_departed

        completed |= transition.episode_completed
        observation = transition.next_observation
        if observation_callback is not None:
            observation_callback(env)
        global_steps += 1
        if global_steps > env.trajectory.frame_count:
            raise RuntimeError("independent evaluation exceeded the reference horizon")

    if (step_count <= 0.0).any().item():
        raise RuntimeError("independent evaluation produced an empty episode")

    hand_error_mean = hand_error_sum / step_count
    link7_error_mean = link7_error_sum / step_count
    flow_error_mean = flow_error_sum / step_count

    if diagnostics_callback is not None:
        scalar_rows = zip(
            trajectories.detach().cpu().tolist(),
            starts.detach().cpu().tolist(),
            step_count.detach().cpu().tolist(),
            episode_success.detach().cpu().tolist(),
            object_departed.detach().cpu().tolist(),
            flow_error_mean.detach().cpu().tolist(),
            final_flow_error.detach().cpu().tolist(),
            link7_error_mean.detach().cpu().tolist(),
            link7_error_max.detach().cpu().tolist(),
            hand_error_mean.detach().cpu().tolist(),
            maximum_object_lift.detach().cpu().tolist(),
            first_object_lift_step.detach().cpu().tolist(),
        )
        diagnostics_callback(
            tuple(
                IndependentEnvironmentDiagnostic(
                    environment_index=environment_index,
                    trajectory_index=int(trajectory_index),
                    start_index=int(start_index),
                    valid_transition_count=int(valid_count),
                    episode_success=bool(success),
                    object_departed=bool(departed),
                    mean_object_flow_error_m=float(mean_flow_error),
                    final_object_flow_error_m=float(final_error),
                    mean_link7_position_error_m=float(mean_link7_error),
                    max_link7_position_error_m=float(max_link7_error),
                    mean_hand_position_rmse_rad=float(mean_hand_error),
                    object_lifted_2cm=float(maximum_lift) >= 0.02,
                    first_object_lift_step=(
                        None if first_lift_step < 0 else int(first_lift_step)
                    ),
                    maximum_object_lift_m=float(maximum_lift),
                )
                for environment_index, (
                    trajectory_index,
                    start_index,
                    valid_count,
                    success,
                    departed,
                    mean_flow_error,
                    final_error,
                    mean_link7_error,
                    max_link7_error,
                    mean_hand_error,
                    maximum_lift,
                    first_lift_step,
                ) in enumerate(scalar_rows)
            )
        )

    def average(value: Tensor) -> float:
        return float(value.mean().cpu())

    return IndependentEpisodeStats(
        joint_acceleration_metrics=acceleration_metrics.result(),
        evaluation_num_envs=count,
        valid_transition_count=int(step_count.sum().cpu()),
        start_indices_sha256=_start_indices_sha256(starts),
        start_index_min=int(starts.min().cpu()),
        start_index_mean=float(starts.double().mean().cpu()),
        start_index_max=int(starts.max().cpu()),
        completion_rate=average(completed.to(torch.float64)),
        mean_link7_position_error_m=average(link7_error_mean),
        mean_episode_max_link7_position_error_m=average(link7_error_max),
        mean_hand_position_rmse_rad=average(hand_error_mean),
    )


def rank_each_metric(
    candidates: Sequence[IndependentCandidateResult],
) -> tuple[IndependentMetricRanking, ...]:
    """Return one independent ordering per metric and no overall winner."""

    values = tuple(candidates)
    if len(values) < 2:
        raise ValueError("comparison requires at least two candidates")
    labels = [item.label for item in values]
    if len(labels) != len(set(labels)):
        raise ValueError("comparison candidate labels must be unique")
    starts = {item.metrics.start_indices_sha256 for item in values}
    if len(starts) != 1:
        raise ValueError("comparison candidates did not use identical start indices")

    rankings: list[IndependentMetricRanking] = []
    for metric in INDEPENDENT_METRICS:
        ordered = sorted(
            values,
            key=lambda item: (metric.raw_value(item.metrics), item.label),
        )
        rankings.append(
            IndependentMetricRanking(
                metric=metric,
                candidates=tuple(
                    MetricRankedCandidate(
                        rank=index,
                        result=result,
                        raw_value=metric.raw_value(result.metrics),
                    )
                    for index, result in enumerate(ordered, start=1)
                ),
            )
        )
    return tuple(rankings)


def _comparison_rows(
    candidates: Sequence[IndependentCandidateResult],
) -> list[list[str]]:
    return [
        [
            result.label,
            str(result.checkpoint_update),
            *(metric.format_value(result.metrics) for metric in INDEPENDENT_METRICS),
        ]
        for result in candidates
    ]


_COMPARISON_HEADERS = (
    "Candidate",
    "Update",
    *(f"{metric.label} ({metric.unit})" for metric in INDEPENDENT_METRICS),
)


def format_comparison_table(
    candidates: Sequence[IndependentCandidateResult],
) -> str:
    rows = _comparison_rows(candidates)
    widths = [len(header) for header in _COMPARISON_HEADERS]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def line(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        (line(_COMPARISON_HEADERS), separator, *(line(row) for row in rows))
    )


def format_comparison_csv(
    candidates: Sequence[IndependentCandidateResult],
) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(_COMPARISON_HEADERS)
    writer.writerows(_comparison_rows(candidates))
    return stream.getvalue()


def format_metric_rankings_markdown(
    rankings: Sequence[IndependentMetricRanking],
) -> str:
    sections = [
        "# Trajectory residual independent metric comparison\n",
        "No combined score or overall winner is computed. Lower is better for every table.\n",
    ]
    for ranking in rankings:
        metric = ranking.metric
        sections.extend(
            (
                f"## {metric.label}\n",
                f"| Rank | Candidate | Update | Value ({metric.unit}) |\n",
                "|---:|---|---:|---:|\n",
            )
        )
        sections.extend(
            f"| {item.rank} | {item.result.label} | "
            f"{item.result.checkpoint_update} | "
            f"{metric.format_value(item.result.metrics)} |\n"
            for item in ranking.candidates
        )
        sections.append("\n")
    return "".join(sections)


def metric_rankings_manifest(
    rankings: Sequence[IndependentMetricRanking],
) -> dict[str, list[dict[str, object]]]:
    return {
        ranking.metric.key: [
            {
                "rank": item.rank,
                "label": item.result.label,
                "checkpoint_update": item.result.checkpoint_update,
                "value": ranking.metric.display_value(item.result.metrics),
                "unit": ranking.metric.unit,
            }
            for item in ranking.candidates
        ]
        for ranking in rankings
    }


__all__ = [
    "INDEPENDENT_METRICS",
    "IndependentCandidateResult",
    "IndependentEnvironmentDiagnostic",
    "IndependentEpisodeStats",
    "IndependentMetric",
    "IndependentMetricRanking",
    "MetricRankedCandidate",
    "evaluate_independent_metrics",
    "format_comparison_csv",
    "format_comparison_table",
    "format_metric_rankings_markdown",
    "independent_metrics_manifest",
    "metric_rankings_manifest",
    "policy_evaluation_contract",
    "rank_each_metric",
    "verify_historical_policy_evaluation_lineage",
]
