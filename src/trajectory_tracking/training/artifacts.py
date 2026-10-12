"""Small reproducible run-artifact and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4

import torch

from trajectory_tracking.compatibility.checkpoint import load_checkpoint as load_archive

from ..configuration import TrajectoryDirectConfig
from ..environment.actions import CanonicalActionSpec
from ..environment.action_smoothness import SMOOTHNESS_METRIC_NAMES
from .ppo import PPOTrainer
from .run_control import TrainingControlSpec


CHECKPOINT_SCHEMA_VERSION = 16
WANDB_METRIC_VIEW_VERSION = "trajectory_compact_v9_action_smoothness"
_TRAINING_TIME_WANDB_GROUPS = frozenset(
    {
        "trajres-training-scale",
        "trajres-multi-trajectory-scale",
        # Compatibility with speed runs launched before the group taxonomy.
        "trajectory-residual-stage1-speed-scaling",
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointResumeState:
    update: int
    consecutive_successful_evaluations: int
    environment_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class CheckpointPolicyState:
    """Checkpoint metadata returned by inference-only policy loading."""

    update: int
    training_num_envs: int


def stable_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(project_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_lineage(project_root: str | Path) -> dict[str, object]:
    """Record the outer commit, dirty patch identity, and runtime source identity."""

    root = Path(project_root).resolve()
    diff = subprocess.run(
        ("git", "-C", str(root), "diff", "--binary", "HEAD"),
        check=True,
        capture_output=True,
    ).stdout
    submodule_rows = _git(root, "submodule", "status", "--recursive").splitlines()
    submodules: dict[str, str] = {}
    for row in submodule_rows:
        fields = row.lstrip("-+U ").split()
        if len(fields) >= 2:
            submodules[fields[1]] = fields[0]
    # New experiment modules/configs may intentionally remain uncommitted.
    # A tracked diff alone would omit the very implementation being compared.
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--",
                     "src/trajectory_tracking", "configs/trajectory_tracking",
                     "scripts/launch_train", "tools/visualization", "tools/viewers")
    untracked_source_sha256 = {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in untracked.splitlines() if (root / relative).is_file()
    }
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(diff),
        "tracked_diff_sha256": sha256(diff).hexdigest(),
        "untracked_source_sha256": untracked_source_sha256,
        "submodules": submodules,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.partial"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunArtifacts:
    """Own one immutable run directory and append-only JSONL metrics."""

    def __init__(self, output_root: str | Path, experiment_name: str) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_name):
            raise ValueError("experiment_name must be a lowercase filesystem slug")
        root = Path(output_root).expanduser().resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / f"{timestamp}_{experiment_name}"
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{timestamp}_{experiment_name}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.checkpoint_dir = run_dir / "checkpoints"

    def write_manifest(self, payload: dict[str, object]) -> None:
        _atomic_json(self.run_dir / "run_manifest.json", payload)

    def write_resolved_config(self, config: TrajectoryDirectConfig) -> None:
        _atomic_json(self.run_dir / "resolved_config.json", config.to_manifest())

    def append_metrics(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, payload: dict[str, object]) -> None:
        _atomic_json(self.run_dir / "summary.json", payload)

    def checkpoint_path(self, update: int) -> Path:
        if update < 0:
            raise ValueError("checkpoint update cannot be negative")
        return self.checkpoint_dir / f"update_{update:06d}.pt"


@dataclass(frozen=True, slots=True)
class WandbTrackingSpec:
    """Runtime-only W&B settings; local JSONL remains authoritative."""

    enabled: bool = False
    mode: str = "offline"
    project: str = "interprior-bc2resrl"
    entity: str | None = None
    group: str | None = "trajectory-residual-ppo"
    run_name: str | None = None
    resume_run_id: str | None = None
    resume_next_history_step: int = 0

    def __post_init__(self) -> None:
        if (type(self.resume_next_history_step) is not int or self.resume_next_history_step < 0):
            raise ValueError("resume history step must be a nonnegative integer")
        if not isinstance(self.enabled, bool):
            raise TypeError("wandb enabled must be a boolean")
        if self.mode not in ("online", "offline"):
            raise ValueError("wandb mode must be online or offline")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("wandb project must be a non-empty string")
        if self.resume_run_id is not None and (not self.enabled or self.mode != "online"):
            raise ValueError("W&B identity resume requires enabled online tracking")
        for name in ("entity", "group", "run_name", "resume_run_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"wandb {name} must be None or a non-empty string")

    def to_manifest(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "group": self.group,
            "run_name": self.run_name,
            **({"resume_run_id": self.resume_run_id} if self.resume_run_id is not None else {}),
        }


_TIME_WANDB_FIELDS = (
    ("training_elapsed_seconds", "time/training_elapsed_seconds"),
    ("iteration_seconds", "time/iteration_seconds"),
)

_UPDATE_WANDB_FIELDS = (
    ("mean_reward", "train/tracking_reward"),
    ("mean_learner_reward", "train/learner_reward"),
    ("iteration_success_rate", "train/robot_success_rate"),
    ("episode_success_rate", "train/episode_success_rate"),
    ("episode_progress_mean", "train/episode_progress"),
    ("completed_episode_count", "train/completed_episodes"),
    ("mean_arm_joint_position_rmse_rad", "train/arm_rmse_rad"),
    ("mean_hand_joint_position_rmse_rad", "train/hand_rmse_rad"),
    ("mean_link7_position_error_m", "train/link7_error_m"),
    ("mean_flow_error_m", "train/object_flow_error_m"),
    ("dynamic_object_flow_error_m", "train/dynamic_object_flow_error_m"),
    ("object_motion_normalized_error", "train/object_motion_normalized_error"),
    ("object_motion_tracking_score", "train/object_motion_tracking_score"),
    (
        "dynamic_robot_keypoint_error_m",
        "train/dynamic_robot_keypoint_error_m",
    ),
    ("robot_motion_normalized_error", "train/robot_motion_normalized_error"),
    ("robot_motion_tracking_score", "train/robot_motion_tracking_score"),
    (
        "object_reference_moving_fraction",
        "train/object_reference_moving_fraction",
    ),
    (
        "robot_reference_moving_fraction",
        "train/robot_reference_moving_fraction",
    ),
    ("mean_object_position_error_m", "train/object_root_error_m"),
    ("mean_joint_velocity_penalty", "train/joint_velocity_penalty"),
    ("ppo_approximate_kl", "ppo/approximate_kl"),
    ("ppo_clip_fraction", "ppo/clip_fraction"),
    ("ppo_actor_learning_rate", "ppo/actor_learning_rate"),
    ("ppo_value_loss", "ppo/value_loss"),
    ("ppo_explained_variance", "ppo/explained_variance"),
    ("ppo_early_stopped", "ppo/early_stopped"),
)

_EVALUATION_WANDB_FIELDS = (
    ("mean_reward", "evaluation/tracking_reward"),
    ("robot_success_fraction", "evaluation/robot_success_rate"),
    ("episode_success_rate", "evaluation/episode_success_rate"),
    ("episode_length_steps", "evaluation/episode_length_steps"),
    ("mean_arm_joint_position_rmse_rad", "evaluation/arm_rmse_rad"),
    ("mean_hand_joint_position_rmse_rad", "evaluation/hand_rmse_rad"),
    ("mean_link7_position_error_m", "evaluation/link7_error_m"),
    ("mean_flow_error_m", "evaluation/object_flow_error_m"),
    (
        "dynamic_object_flow_error_m",
        "evaluation/dynamic_object_flow_error_m",
    ),
    (
        "object_motion_normalized_error",
        "evaluation/object_motion_normalized_error",
    ),
    (
        "object_motion_tracking_score",
        "evaluation/object_motion_tracking_score",
    ),
    (
        "dynamic_robot_keypoint_error_m",
        "evaluation/dynamic_robot_keypoint_error_m",
    ),
    (
        "robot_motion_normalized_error",
        "evaluation/robot_motion_normalized_error",
    ),
    (
        "robot_motion_tracking_score",
        "evaluation/robot_motion_tracking_score",
    ),
    (
        "object_reference_moving_fraction",
        "evaluation/object_reference_moving_fraction",
    ),
    (
        "robot_reference_moving_fraction",
        "evaluation/robot_reference_moving_fraction",
    ),
    ("mean_object_position_error_m", "evaluation/object_root_error_m"),
    ("mean_joint_velocity_penalty", "evaluation/joint_velocity_penalty"),
)


def _copy_wandb_fields(
    payload: Mapping[str, object],
    fields: tuple[tuple[str, str], ...],
    output: dict[str, bool | int | float],
) -> None:
    for source, destination in fields:
        value = payload.get(source)
        if isinstance(value, bool):
            output[destination] = value
        elif isinstance(value, int):
            output[destination] = value
        elif isinstance(value, Real):
            numeric = float(value)
            if isfinite(numeric):
                output[destination] = numeric


def _add_wandb_outcome_rates(
    payload: Mapping[str, object],
    *,
    prefix: str,
    output: dict[str, bool | int | float],
) -> None:
    completed = payload.get("completed_episode_count")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed <= 0:
        return
    robot_failure_destination = (
        "link7_pose_reset_rate"
        if payload.get("robot_tracking_reset_mode") == "link7_weighted_pose"
        else "maniptrans_failure_rate"
    )
    outcomes = (
        ("endpoint_miss_episode_count", "endpoint_miss_rate"),
        (
            "object_departure_failure_episode_count",
            "object_departure_failure_rate",
        ),
        ("maniptrans_failure_episode_count", robot_failure_destination),
        ("object_drop_or_escape_episode_count", "object_drop_or_escape_rate"),
    )
    for source, destination in outcomes:
        count = payload.get(source)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            continue
        output[f"{prefix}/{destination}"] = count / completed


class TrajectoryWandbMirror:
    """Best-effort W&B mirror for already-persisted trajectory run events."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        spec: WandbTrackingSpec,
        wandb_module: Any | None = None,
    ) -> None:
        if not isinstance(spec, WandbTrackingSpec):
            raise TypeError("spec must be a WandbTrackingSpec")
        self.spec = spec
        self.run_dir = Path(run_dir).resolve(strict=True)
        self._run: Any | None = None
        self._error: str | None = None
        self._logging_disabled = False
        self._finished = False
        self._history_step = 0
        self._current_update = 0
        self._terminal_status: str | None = None
        self._terminal_stop_reason: str | None = None
        if not spec.enabled:
            return
        try:
            if wandb_module is None:
                import wandb as wandb_module

            settings = wandb_module.Settings(
                console="off",
                disable_code=True,
                x_disable_stats=True,
                x_save_requirements=False,
            )
            self._run = wandb_module.init(
                **({"id": spec.resume_run_id, "resume": "must"}
                   if spec.resume_run_id is not None else {}),
                project=spec.project,
                entity=spec.entity,
                group=spec.group,
                name=spec.run_name or self.run_dir.name,
                mode=spec.mode,
                dir=str(self.run_dir),
                config={
                    "task": "trajectory_tracking",
                    "metric_view_version": WANDB_METRIC_VIEW_VERSION,
                    "tracking": spec.to_manifest(),
                },
                settings=settings,
            )
            if self._run is None:
                raise RuntimeError("wandb.init returned no run")
            if spec.resume_run_id is not None:
                if self.run_id != spec.resume_run_id or not self._run.resumed:
                    raise RuntimeError("W&B must resume the exact existing run")
                # SDK run.step can be zero even after a successful online resume.
                # A bounded, authoritative read prevents silently discarded history.
                remote = wandb_module.Api(timeout=30).run(
                    f"{self._run.entity}/{self._run.project}/{self.run_id}")
                last_step = remote.lastHistoryStep
                if isinstance(last_step, bool) or not isinstance(last_step, int) or last_step < -1:
                    raise ValueError("invalid W&B server history step")
                self._history_step = max(last_step + 1, int(getattr(self._run, "step", 0)),
                                         int(getattr(self._run, "starting_step", 0)),
                                         spec.resume_next_history_step)
                self._run.summary["training/status"] = "running"
                _atomic_json(self.run_dir / "wandb_resume.json", dict(
                    run_id=self.run_id, next_history_step=self._history_step,
                    server_last_history_step=last_step))
            wandb_module.define_metric("global/update")
            wandb_module.define_metric("time/training_elapsed_seconds")
            primary_step_metric = (
                "time/training_elapsed_seconds"
                if spec.group in _TRAINING_TIME_WANDB_GROUPS
                else "global/update"
            )
            wandb_module.define_metric(
                "time/iteration_seconds",
                step_metric=primary_step_metric,
            )
            for namespace in ("train/*", "ppo/*", "evaluation/*", "train_evaluation/*", "test/*"):
                wandb_module.define_metric(
                    namespace,
                    step_metric=primary_step_metric,
                )
        except Exception as error:
            # Explicit resume is an identity contract, never silently create a new
            # run or continue with a history counter known to be unverified.
            if spec.resume_run_id is not None:
                if self._run is not None:
                    try:
                        self._run.finish(exit_code=1)
                    except Exception:
                        pass
                raise RuntimeError("W&B resume identity/history verification failed") from error
            self._run = None
            self._logging_disabled = True
            self._record_error("initialization", error)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def run_id(self) -> str | None:
        value = None if self._run is None else getattr(self._run, "id", None)
        return value if isinstance(value, str) and value else None

    @property
    def run_url(self) -> str | None:
        value = None if self._run is None else getattr(self._run, "url", None)
        return value if isinstance(value, str) and value else None

    def _record_error(self, stage: str, error: Exception) -> None:
        detail = f"{stage}: {type(error).__name__}: {error}"
        self._error = detail if self._error is None else f"{self._error}; {detail}"
        print(
            f"W&B {stage} failed; continuing with local metrics.jsonl: {detail}",
            flush=True,
        )

    def bind_manifest(self, manifest: dict[str, object]) -> None:
        """Attach reproducibility metadata without making it a run dependency."""

        if self._run is None:
            return
        try:
            self._run.config.update(
                {"reproducibility_manifest": manifest},
                allow_val_change=True,
            )
        except Exception as error:
            self._record_error("config logging", error)

    def _event_values(
        self,
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        event = payload.get("event")
        if not isinstance(event, str) or not event:
            raise ValueError("W&B event payload requires a non-empty event")
        update = payload.get("update", self._current_update)
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError("W&B event update must be a non-negative integer")
        self._current_update = update
        values: dict[str, object] = {
            "global/update": update,
        }
        numeric: dict[str, bool | int | float] = {}
        _copy_wandb_fields(payload, _TIME_WANDB_FIELDS, numeric)
        if event == "update":
            _copy_wandb_fields(payload, _UPDATE_WANDB_FIELDS, numeric)
            _add_wandb_outcome_rates(payload, prefix="train", output=numeric)
        elif event == "evaluation":
            mode = payload.get("mode")
            if not isinstance(mode, str) or not mode.startswith("C_"):
                return None
            _copy_wandb_fields(payload, _EVALUATION_WANDB_FIELDS, numeric)
            _add_wandb_outcome_rates(
                payload,
                prefix="evaluation",
                output=numeric,
            )
            if payload.get("dataset_split") == "train":
                numeric.update({key.replace("evaluation/", "train_evaluation/", 1): value
                                for key, value in tuple(numeric.items())
                                if key.startswith("evaluation/")})
        elif event == "test_evaluation":
            from .heldout import TEST_FIELDS

            _copy_wandb_fields(payload, tuple((name, f"test/{name}") for name in TEST_FIELDS), numeric)
        else:
            return None
        parallel = payload.get("parallel_metrics")
        if isinstance(parallel, Mapping):
            _copy_wandb_fields(parallel, tuple((name, f"parallel/{name}") for name in (
                "world_size", "rollout_seconds", "ppo_seconds", "valid_transitions_per_second",
                "evaluation_fraction",
            )), numeric)
        smoothness = payload.get("action_smoothness_metrics")
        acceleration = payload.get("joint_acceleration_metrics")
        if isinstance(acceleration, Mapping):
            prefix = {"update": "train", "test_evaluation": "test"}.get(event, "evaluation")
            _copy_wandb_fields(acceleration, tuple(
                (name, f"{prefix}/joint_acceleration/{name}") for name in (
                    "valid_transition_count", "arm_rms_rad_s2", "hand_rms_rad_s2",
                    "arm_abs_p95_rad_s2", "hand_abs_p95_rad_s2", "mean_normalized_huber",
                    "mean_penalty_public", "mean_unpenalized_tracking_reward")
            ), numeric)
        if isinstance(smoothness, Mapping):
            prefix = "train" if event == "update" else "evaluation"
            _copy_wandb_fields(smoothness, tuple(
                (name, f"{prefix}/{name}") for name in SMOOTHNESS_METRIC_NAMES
            ), numeric)
        values.update(numeric)
        return values

    def log_event(self, payload: dict[str, object]) -> None:
        """Mirror one event only after its local JSONL append has succeeded."""

        if not isinstance(payload, dict):
            raise TypeError("W&B event payload must be a dictionary")
        event = payload.get("event")
        if event in ("complete", "stop"):
            status = payload.get("status")
            self._terminal_status = status if isinstance(status, str) else str(event)
            stop_reason = payload.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                self._terminal_stop_reason = stop_reason
        elif event == "error":
            self._terminal_status = "error"
        if self._run is None or self._logging_disabled or self._finished:
            return
        try:
            values = self._event_values(payload)
            if values is None:
                return
            self._run.log(values, step=self._history_step)
            self._history_step += 1
            _atomic_json(self.run_dir / "wandb_history.json", dict(
                run_id=self.run_id, next_history_step=self._history_step))
        except Exception as error:
            self._logging_disabled = True
            self._record_error("metric logging", error)

    def finish(self, *, exit_code: int) -> None:
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("exit_code must be an integer")
        if self._finished:
            return
        self._finished = True
        if self._run is None:
            return
        try:
            self._run.summary["training/status"] = self._terminal_status or (
                "process_completed" if exit_code == 0 else "process_failed"
            )
            self._run.summary["training/final_update"] = self._current_update
            self._run.summary["training/process_exit_code"] = exit_code
            if self._terminal_stop_reason is not None:
                self._run.summary["training/stop_reason"] = self._terminal_stop_reason
            if self._error is not None:
                self._run.summary["tracking/wandb_error"] = self._error
            self._run.finish(exit_code=0 if exit_code == 0 else 1)
        except Exception as error:
            self._record_error("finalization", error)


def save_ppo_checkpoint(
    path: str | Path,
    *,
    trainer: PPOTrainer,
    config: TrajectoryDirectConfig,
    action_spec: CanonicalActionSpec,
    trajectory_id: str,
    scene_manifest: dict[str, object],
    adapter_version: str,
    update: int,
    num_envs: int,
    control: TrainingControlSpec,
    consecutive_successful_evaluations: int,
    environment_state: dict[str, object],
    parallel_manifest: dict[str, object] | None = None,
) -> Path:
    """Atomically save PPO and every compatibility boundary needed to restore."""

    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("checkpoint update must be a non-negative integer")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("checkpoint num_envs must be a positive integer")
    streak = consecutive_successful_evaluations
    if (
        isinstance(streak, bool)
        or not isinstance(streak, int)
        or streak < 0
        or streak > control.success_patience
    ):
        raise ValueError("checkpoint success streak is invalid")
    rollout_state = _validated_environment_state(
        environment_state,
    )
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": "trajectory_residual",
        "update": update,
        "trajectory_id": trajectory_id,
        "adapter_version": adapter_version,
        "environment_version": config.environment_version,
        "config_compatibility_hash": config.compatibility_hash,
        "action_spec_hash": stable_json_hash(action_spec.to_manifest()),
        "scene_manifest_hash": stable_json_hash(scene_manifest),
        "training": {
            "num_envs": num_envs,
            "max_updates_at_save": control.max_updates,
            "evaluation_gate": control.gate_manifest(),
            "consecutive_successful_evaluations": streak,
            "environment_state": rollout_state,
        },
        "trainer": trainer.state_dict(),
    }
    if parallel_manifest is not None:
        payload["parallel"] = parallel_manifest
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.partial"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _load_compatible_ppo_checkpoint_payload(
    path: str | Path,
    *,
    config: TrajectoryDirectConfig,
    action_spec: CanonicalActionSpec,
    trajectory_id: str,
    scene_manifest: dict[str, object],
    adapter_version: str,
    checkpoint_config_compatibility_hash: str | None = None,
    checkpoint_adapter_version: str | None = None,
) -> tuple[dict[str, object], int]:
    """Read and validate checkpoint boundaries shared by resume and evaluation."""

    expected_config_hash = (
        config.compatibility_hash
        if checkpoint_config_compatibility_hash is None
        else checkpoint_config_compatibility_hash
    )
    if (
        not isinstance(expected_config_hash, str)
        or len(expected_config_hash) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_config_hash
        )
    ):
        raise ValueError(
            "checkpoint config compatibility hash must be lowercase SHA-256"
        )
    expected_adapter_version = (
        adapter_version
        if checkpoint_adapter_version is None
        else checkpoint_adapter_version
    )
    if not isinstance(expected_adapter_version, str) or not expected_adapter_version:
        raise ValueError("checkpoint adapter version must be non-empty")

    # Keep serialized RNG and environment state on CPU. Module/optimizer loading
    # moves parameter-owned tensors to the trainer device, while Generator
    # ``set_state`` requires a CPU ByteTensor even for a CUDA generator.
    payload = load_archive(Path(path), map_location="cpu")
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": "trajectory_residual",
        "trajectory_id": trajectory_id,
        "adapter_version": expected_adapter_version,
        "environment_version": config.environment_version,
        "config_compatibility_hash": expected_config_hash,
        "action_spec_hash": stable_json_hash(action_spec.to_manifest()),
        "scene_manifest_hash": stable_json_hash(scene_manifest),
    }
    if not isinstance(payload, dict):
        raise ValueError("trajectory checkpoint payload must be a mapping")
    mismatches = {
        name: (expected_value, payload.get(name))
        for name, expected_value in expected.items()
        if payload.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"trajectory checkpoint compatibility mismatch: {mismatches}")
    update = payload.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("trajectory checkpoint update counter is invalid")
    return payload, update


def load_ppo_policy_checkpoint(
    path: str | Path,
    *,
    trainer: PPOTrainer,
    config: TrajectoryDirectConfig,
    action_spec: CanonicalActionSpec,
    trajectory_id: str,
    scene_manifest: dict[str, object],
    adapter_version: str,
    verified_training_config_compatibility_hash: str | None = None,
    verified_training_adapter_version: str | None = None,
) -> CheckpointPolicyState:
    """Load only deterministic-policy weights for a fresh evaluation batch.

    Evaluation deliberately does not restore the optimizer, PPO random streams,
    success-gate state, or an in-progress training episode.  It therefore may
    use a different number of parallel environments while preserving all task,
    data, action, scene, model, and observation compatibility boundaries. The
    optional historical hash is accepted only after the caller has verified a
    frozen training manifest against the policy-evaluation contract; resume
    remains strictly bound to the current full config hash.
    """

    payload, update = _load_compatible_ppo_checkpoint_payload(
        path,
        config=config,
        action_spec=action_spec,
        trajectory_id=trajectory_id,
        scene_manifest=scene_manifest,
        adapter_version=adapter_version,
        checkpoint_config_compatibility_hash=(
            verified_training_config_compatibility_hash
        ),
        checkpoint_adapter_version=verified_training_adapter_version,
    )
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("trajectory checkpoint lacks training metadata")
    training_num_envs = training.get("num_envs")
    if (
        isinstance(training_num_envs, bool)
        or not isinstance(training_num_envs, int)
        or training_num_envs <= 0
    ):
        raise ValueError("trajectory checkpoint training num_envs is invalid")
    state = payload.get("trainer")
    if not isinstance(state, dict):
        raise ValueError("trajectory checkpoint lacks trainer state")
    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("trajectory checkpoint lacks PPO model state")
    trainer.model.load_state_dict(model_state)  # type: ignore[arg-type]
    trainer.model.eval()
    return CheckpointPolicyState(
        update=update,
        training_num_envs=training_num_envs,
    )


def load_ppo_checkpoint(
    path: str | Path,
    *,
    trainer: PPOTrainer,
    config: TrajectoryDirectConfig,
    action_spec: CanonicalActionSpec,
    trajectory_id: str,
    scene_manifest: dict[str, object],
    adapter_version: str,
    num_envs: int,
    control: TrainingControlSpec,
) -> CheckpointResumeState:
    """Restore only when task, data, action, scene, and config are identical."""

    payload, update = _load_compatible_ppo_checkpoint_payload(
        path,
        config=config,
        action_spec=action_spec,
        trajectory_id=trajectory_id,
        scene_manifest=scene_manifest,
        adapter_version=adapter_version,
    )
    if "parallel" in payload:
        raise ValueError("distributed exact resume requires the complete checkpoint directory")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("trajectory checkpoint lacks exact-resume training state")
    if training.get("num_envs") != num_envs:
        raise ValueError(
            "trajectory checkpoint num_envs mismatch: "
            f"expected {num_envs}, got {training.get('num_envs')}"
        )
    expected_gate = control.gate_manifest()
    if training.get("evaluation_gate") != expected_gate:
        raise ValueError(
            "trajectory checkpoint evaluation gate mismatch: "
            f"expected {expected_gate}, got {training.get('evaluation_gate')}"
        )
    streak = training.get("consecutive_successful_evaluations")
    if (
        isinstance(streak, bool)
        or not isinstance(streak, int)
        or streak < 0
        or streak > control.success_patience
    ):
        raise ValueError("trajectory checkpoint success streak is invalid")
    environment_state = _validated_environment_state(
        training.get("environment_state"),
    )
    state = payload.get("trainer")
    if not isinstance(state, dict):
        raise ValueError("trajectory checkpoint lacks trainer state")
    trainer.load_state_dict(state)
    return CheckpointResumeState(
        update=update,
        consecutive_successful_evaluations=streak,
        environment_state=environment_state,
    )


def _validated_environment_state(
    value: object,
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "phase",
        "reset_rng_state",
        "episode",
        # The historical field name is retained in the rollout schema, but the
        # monotonically increasing task clock is shared by every reward profile.
        "maniptrans_simulation_frame_count",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("checkpoint environment state has invalid keys")

    def clone(item: object, *, path: str) -> object:
        if item is None or isinstance(item, str):
            return item
        if isinstance(item, bool) or (not isinstance(item, (int, dict, torch.Tensor))):
            raise ValueError(f"checkpoint environment state {path} is invalid")
        if isinstance(item, int):
            return item
        if isinstance(item, torch.Tensor):
            empty_zero_pre_roll_queue = (
                path == "environment.episode.pre_roll_actions"
                and item.dtype == torch.float32
                and item.ndim == 3
                and item.shape[0] > 0
                and item.shape[1:] == (0, 27)
            )
            invalid_empty = item.numel() == 0 and not empty_zero_pre_roll_queue
            invalid_dtype = item.dtype not in (
                torch.uint8,
                torch.bool,
                torch.int64,
                torch.float32,
            )
            if invalid_empty or invalid_dtype:
                raise ValueError(
                    f"checkpoint environment tensor {path} has invalid metadata"
                )
            if item.is_floating_point() and not torch.isfinite(item).all().item():
                raise ValueError(f"checkpoint environment tensor {path} is non-finite")
            return item.detach().cpu().clone()
        if any(not isinstance(key, str) for key in item):
            raise ValueError(
                f"checkpoint environment mapping {path} has non-string keys"
            )
        return {key: clone(child, path=f"{path}.{key}") for key, child in item.items()}

    result = clone(value, path="environment")
    assert isinstance(result, dict)
    return result


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointPolicyState",
    "CheckpointResumeState",
    "RunArtifacts",
    "TrajectoryWandbMirror",
    "WANDB_METRIC_VIEW_VERSION",
    "WandbTrackingSpec",
    "load_ppo_checkpoint",
    "load_ppo_policy_checkpoint",
    "repository_lineage",
    "save_ppo_checkpoint",
    "stable_json_hash",
]
