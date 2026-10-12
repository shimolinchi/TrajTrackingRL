from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

from trajectory_tracking.training import coordinator as module


@pytest.mark.parametrize("test_only,expected", [
    (True, [(0, True), (2, True), (3, True)]),
    (False, [(0, True), (2, False), (2, True), (3, False), (3, True)]),
])
def test_periodic_test_protocol_never_evaluates_training_or_duplicates_tests(tmp_path, monkeypatch, test_only, expected):
    keys = ("start_index_min", "start_index_mean", "start_index_max", "reference_cursor_min",
            "reference_cursor_mean", "reference_cursor_max", "episode_progress_mean")
    Stats = make_dataclass("Stats", [(name, float) for name in keys] + [("ppo", dict), ("valid_transition_count", int)])
    stats = Stats(*([0.] * len(keys)), {}, 8)
    runner = SimpleNamespace(env=SimpleNamespace(needs_reset=False), run_update=lambda: stats, last_timing={})
    context = SimpleNamespace(world_size=1, reduce=lambda value, operation="sum": value)
    coordinator = module.SynchronousCoordinator(runner=runner, context=context,
        artifacts=SimpleNamespace(run_dir=tmp_path), tracker=None, contract={}, checkpoint_kwargs={},
        evaluate_train=lambda: None, evaluate_test=lambda: None, periodic_heldout=test_only)
    calls = []
    coordinator.evaluate = lambda update, heldout=False: calls.append((update, heldout))
    coordinator.emit = lambda payload: None
    monkeypatch.setattr(module, "save_checkpoint", lambda *args, **kwargs: tmp_path)
    coordinator.run(max_updates=3, evaluation_interval=2, test_interval=2, checkpoint_interval=3)
    assert calls == expected


def test_test_only_mode_requires_a_real_test_callback():
    with pytest.raises(ValueError, match="test callback"):
        module.SynchronousCoordinator(runner=None, context=None, artifacts=None, tracker=None,
            contract={}, checkpoint_kwargs={}, evaluate_train=lambda: None, periodic_heldout=True)
