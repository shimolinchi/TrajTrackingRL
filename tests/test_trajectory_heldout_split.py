"""Whole-episode leakage, frozen baseline and read-only test rollout contracts."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from trajectory_tracking.data.preparation.split import grouped_partition
from trajectory_tracking.configuration import load_trajectory_direct_config
from trajectory_tracking.data.heldout_split import heldout_test_config
from trajectory_tracking.data.variable_length import VariableLengthReferenceBank
from trajectory_tracking.environment.core import TrajectoryDirectEnvCore
from trajectory_tracking.training.heldout import evaluate_heldout, test_summary as summarize
from trajectory_tracking.training.artifacts import TrajectoryWandbMirror
from trajectory_tracking.utils.synthetic import (
    LinearRecoveryBackend, synthetic_action_spec, synthetic_reference_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
from parallel_fixtures import recipe_config

def config():
    return recipe_config()


class HeldoutSplitTest(unittest.TestCase):
    def test_grouping_is_deterministic_and_transitively_prevents_leakage(self):
        rows = [{"episode_uid": str(i)} for i in range(20)]
        ids = {str(i): [f"unique:{i}"] for i in range(20)}
        ids["0"].append("same_attempt")
        ids["1"] += ["same_attempt", "same_flow"]
        ids["2"].append("same_flow")
        a = grouped_partition(rows, ids, seed=42, test_fraction=.2)
        b = grouped_partition(list(reversed(rows)), ids, seed=42, test_fraction=.2)
        self.assertEqual(a, b)
        train, test, count = a
        self.assertEqual(count, 18)
        self.assertFalse({x for r in train for x in ids[r["episode_uid"]]} &
                         {x for r in test for x in ids[r["episode_uid"]]})
        self.assertEqual(len(train) + len(test), 20)

    def test_test_scoring_excludes_filler_and_requires_zero_start(self):
        def row(i, success):
            return SimpleNamespace(trajectory_index=i, start_index=0,
                episode_success=success, mean_object_flow_error_m=.01,
                final_object_flow_error_m=.02, object_departed=False,
                object_lifted_2cm=True, mean_link7_position_error_m=.03,
                mean_hand_position_rmse_rad=.1)
        rows = [row(0, True), row(1, False), row(0, True)]
        self.assertEqual(summarize(rows, 2)["episode_success_rate"], .5)
        rows[0].start_index = 1
        with self.assertRaisesRegex(ValueError, "local-zero"):
            summarize(rows, 2)

    def test_wandb_test_metrics_cannot_overwrite_training_evaluation(self):
        mirror = object.__new__(TrajectoryWandbMirror)
        mirror._current_update = 0
        values = mirror._event_values({"event": "test_evaluation", "update": 500,
            "episode_success_rate": .3, "episode_success_count": 3, "episode_count": 10})
        self.assertEqual(values["test/episode_success_rate"], .3)
        self.assertNotIn("evaluation/episode_success_rate", values)
        self.assertNotIn("train/episode_success_rate", values)

    def test_test_rollout_restores_training_state_and_model_mode(self):
        c = config()
        c = replace(c, data=replace(c.data, trajectory_count=1),
                    start_index=replace(c.start_index, mode="fixed", fixed_index=1))
        action = synthetic_action_spec()
        training = TrajectoryDirectEnvCore(
            config=c, trajectory=synthetic_reference_trajectory(frame_count=8),
            action_spec=action, backend=LinearRecoveryBackend(num_envs=4, device=torch.device("cpu")),
            num_envs=4,
        )
        training.reset()
        before = training.current_observation().clone()
        cursors = training.reference_cursors.clone()
        model = torch.nn.Linear(1, 1).train()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        trainer = SimpleNamespace(model=model,
            deterministic_action=lambda obs: torch.zeros(obs.shape[0], 27))
        runner = SimpleNamespace(env=training, trainer=trainer)
        bank = VariableLengthReferenceBank(tuple(
            replace(synthetic_reference_trajectory(frame_count=n), trajectory_id=str(n)) for n in (8, 10)))
        bank = bank.with_terminal_frame_counts(
            [7, 9], protocol="heldout_local0_pre_place_descend_v1",
            metadata={"terminal_phase_excluded": "place_descend"},
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as output:
            result = evaluate_heldout(runner=runner, test_config=replace(c,
                data=replace(c.data, trajectory_count=2),
                start_index=replace(c.start_index, mode="uniform_valid", fixed_index=None)),
                test_trajectory=bank, update=0, output_dir=output)
            self.assertEqual(result["episode_count"], 2)
            self.assertEqual(result["physical_environment_count"], 4)
            payload = json.loads(Path(result["diagnostics_path"]).read_text())
            self.assertEqual(len(payload["episodes"]), 2)
            self.assertEqual(payload["protocol"], "heldout_local0_pre_place_descend_v1")
            self.assertEqual(payload["terminal_phase_excluded"], "place_descend")
        torch.testing.assert_close(training.current_observation(), before, rtol=0, atol=0)
        torch.testing.assert_close(training.reference_cursors, cursors, rtol=0, atol=0)
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
