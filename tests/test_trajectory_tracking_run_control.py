"""Absolute update-budget and deterministic success-gate contracts."""

from __future__ import annotations

from dataclasses import replace
import unittest

from trajectory_tracking.training.run_control import (
    ConsecutiveSuccessGate,
    NominalReplayGateSpec,
    TrainingControlSpec,
)


class TrajectoryTrainingControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = TrainingControlSpec(
            max_updates=10,
            evaluation_interval=4,
            target_success_rate=0.8,
            success_patience=2,
        )

    def test_periodic_and_final_updates_are_evaluated(self) -> None:
        evaluated = [
            update
            for update in range(1, self.spec.max_updates + 1)
            if self.spec.should_evaluate(update)
        ]

        self.assertEqual(evaluated, [4, 8, 10])

    def test_gate_requires_consecutive_deterministic_successes(self) -> None:
        gate = ConsecutiveSuccessGate(self.spec)

        self.assertFalse(gate.observe(0.9))
        self.assertEqual(gate.consecutive_successes, 1)
        self.assertFalse(gate.observe(0.7))
        self.assertEqual(gate.consecutive_successes, 0)
        self.assertFalse(gate.observe(0.8))
        self.assertTrue(gate.observe(1.0))
        self.assertTrue(gate.satisfied)

    def test_max_updates_may_change_without_changing_resume_gate(self) -> None:
        extended = replace(self.spec, max_updates=20)

        self.assertEqual(self.spec.gate_manifest(), extended.gate_manifest())
        self.assertNotEqual(self.spec.to_manifest(), extended.to_manifest())

    def test_invalid_controls_fail_fast(self) -> None:
        for field, value in (
            ("max_updates", 0),
            ("evaluation_interval", 0),
            ("target_success_rate", 1.1),
            ("success_patience", 0),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(self.spec, **{field: value})

    def test_nominal_replay_requires_mean_error_and_every_environment(self) -> None:
        gate = NominalReplayGateSpec(
            mean_error_threshold_m=0.03,
            success_rate_threshold=1.0,
        )

        self.assertTrue(
            gate.passes(mean_flow_error_m=0.002, episode_success_rate=1.0)
        )
        self.assertFalse(
            gate.passes(mean_flow_error_m=0.002, episode_success_rate=0.81)
        )
        self.assertFalse(
            gate.passes(mean_flow_error_m=0.04, episode_success_rate=1.0)
        )


if __name__ == "__main__":
    unittest.main()
