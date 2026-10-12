"""Producer identity and explicit no-replay contracts; no private data or GPU."""

import unittest
from types import SimpleNamespace

import numpy as np

from trajectory_tracking.data.tro_success_continuous import successful_candidate, validate_arrays


class ProducerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(episode_status="success", collection_status="completed",
                        trajectory_written=True, training_eligible=True)
        self.meta = dict(episode_status="success", collection_status="completed")

    def test_producer_success_completed_required(self):
        self.assertTrue(successful_candidate(self.row, self.meta))
        for key, value in (("episode_status", "fail"), ("collection_status", "hard_failure")):
            self.assertFalse(successful_candidate(self.row, dict(self.meta, **{key: value})))
            self.assertFalse(successful_candidate(dict(self.row, **{key: value}), self.meta))

    def test_missing_and_ineligible_record_rejected(self):
        self.assertFalse(successful_candidate(self.row, {}))
        self.assertFalse(successful_candidate(dict(self.row, training_eligible=False), self.meta))
        self.assertFalse(successful_candidate(dict(self.row, trajectory_written=False), self.meta))

    def test_continuous_timing_not_original_gap_replay(self):
        pose = np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(float)
        arrays = dict(time_s=np.arange(3) / 60, physics_step=np.arange(3) * 2,
                      source_frame_index=np.array([120, 121, 129]),
                      ee_current_pose_w=pose, object_root_current_pose_w=pose)
        validate_arrays(arrays, 3)  # Retained continuous timing, honest original gap.
        with self.assertRaises(ValueError):
            validate_arrays(dict(arrays, physics_step=np.array([0, 2, 18])), 3)
        with self.assertRaises(ValueError):
            validate_arrays(dict(arrays, time_s=np.array([0, np.nan, 2 / 60])), 3)
        with self.assertRaises(ValueError):
            validate_arrays(arrays, 4)

    def test_singleton_fillers_never_become_test_identities(self):
        from trajectory_tracking.data.partition import resident_evaluation_sources, evaluation_batches
        def source(uid):
            return SimpleNamespace(entry=SimpleNamespace(replay_audit={"object_uid": uid}))
        test = (source("a"), source("a"))
        singleton = source("singleton")
        bank, indices = resident_evaluation_sources(test, (source("a"), singleton), {"a", "singleton"})
        self.assertEqual(indices, (0, 1, -1))
        self.assertIs(bank[-1], singleton)
        batches = list(evaluation_batches(("a", "singleton"), ("a", "a", "singleton"), (0, 1)))
        self.assertEqual(len(batches), 2)
        self.assertEqual([indices[rows[slot]] for rows, scored in batches for slot in scored], [0, 1])



if __name__ == "__main__":
    unittest.main()
