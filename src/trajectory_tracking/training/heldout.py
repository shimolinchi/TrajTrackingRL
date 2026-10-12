"""Opt-in full-start test evaluation, isolated from PPO and its stopping gate."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ..environment.core import TrajectoryDirectEnvCore
from ..data.heldout_terminal import terminal_audit
from .evaluation import evaluate_independent_metrics

TEST_FIELDS = (
    "episode_success_rate", "episode_success_count", "episode_count",
    "object_flow_error_m", "final_object_flow_error_m", "object_departure_rate",
    "ever_lifted_2cm_rate", "link7_error_m", "hand_rmse_rad", "evaluation_seconds",
)


def test_summary(diagnostics, test_count, *, require_exact_count=False):
    """Score each test trajectory exactly once, never the physical filler slots."""
    if require_exact_count and len(diagnostics) != test_count:
        raise ValueError("complete held-out aggregation requires exactly test_count rows")
    rows = diagnostics[:test_count]
    if (len(rows) != test_count or test_count <= 0
            or [r.trajectory_index for r in rows] != list(range(test_count))
            or any(r.start_index != 0 for r in rows)):
        raise ValueError("test requires one local-zero episode per held-out trajectory")

    def mean(field):
        return sum(getattr(row, field) for row in rows) / test_count

    return {
        "episode_success_rate": mean("episode_success"),
        "episode_success_count": sum(r.episode_success for r in rows),
        "episode_count": test_count,
        "object_flow_error_m": mean("mean_object_flow_error_m"),
        "final_object_flow_error_m": mean("final_object_flow_error_m"),
        "object_departure_rate": mean("object_departed"),
        "ever_lifted_2cm_rate": mean("object_lifted_2cm"),
        "link7_error_m": mean("mean_link7_position_error_m"),
        "hand_rmse_rad": mean("mean_hand_position_rmse_rad"),
    }


def evaluate_heldout(*, runner, test_config, test_trajectory, update, output_dir,
                     observation_callback=None):
    """Reuse the scene at a saved rollout boundary; restore training even on error.

    Scene capacity stays unchanged. Only slots [0,test_count) count towards the
    test denominator; extra slots repeat test rows, never training trajectories.
    This avoids a second simulator process or any backend/action changes.
    """
    n = runner.env.num_envs
    count = test_config.data.trajectory_count
    if n < count:
        raise ValueError("held-out evaluation requires scene capacity >= test count")
    saved = runner.env.rollout_state_dict()
    was_training = runner.trainer.model.training
    heldout_protocol, terminal_metadata = terminal_audit(test_trajectory)
    started = time.perf_counter()
    diagnostics = []
    try:
        env = TrajectoryDirectEnvCore(
            config=test_config, trajectory=test_trajectory,
            action_spec=runner.env.action_spec, backend=runner.env.backend,
            num_envs=n, episode_completion_mode="reference_endpoint_only",
        )
        stats = evaluate_independent_metrics(
            env, runner.trainer.deterministic_action,
            start_indices=torch.zeros(n, dtype=torch.long, device=env.device),
            trajectory_indices=torch.arange(n, device=env.device) % count,
            observation_callback=observation_callback,
            diagnostics_callback=diagnostics.extend,
            metric_environment_count=count,
        )
    finally:
        runner.env.load_rollout_state_dict(saved)
        runner.trainer.model.train(was_training)
        if observation_callback is not None:
            observation_callback(runner.env)
    summary = test_summary(diagnostics, count)
    if stats.joint_acceleration_metrics is not None:
        summary["joint_acceleration_metrics"] = stats.joint_acceleration_metrics
    payload = {
        "event": "test_evaluation", "dataset_split": "test", "update": update,
        "protocol": heldout_protocol,
        **terminal_metadata,
        "selection_sha256": test_config.data.selection_sha256,
        "scored_environment_indices": [0, count],
        "physical_environment_count": n,
        "used_for_early_stop": False,
        "evaluation_seconds": time.perf_counter() - started,
        **summary,
    }
    directory = Path(output_dir) / "heldout_test"
    directory.mkdir(exist_ok=True)
    path = directory / f"update_{update:06d}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({**payload, "episodes": [asdict(r) for r in diagnostics[:count]]},
                  stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    payload["diagnostics_path"] = str(path)
    return payload
