"""CPU-only contracts for the explicit distributed source-label replay waiver."""

from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from trajectory_tracking.cli import distributed_train as cli
from trajectory_tracking.cli.producer_admission import producer_admission
from trajectory_tracking.data.tro_success_continuous import ADAPTER_NAME as ADAPTER, PROTOCOL, VERSION
from trajectory_tracking.training.coordinator import SynchronousCoordinator


@pytest.fixture
def selection(tmp_path):
    payload = dict(adapter_version=VERSION, admission_protocol=PROTOCOL,
        action_replay_performed=False, producer_success_required=True,
        dataset_metadata_sha256="a" * 64, source_hashes={"success_continuous/commit.json": "b" * 64},
        count=2, frame_start=0, frame_stop=600,
        split=dict(protocol="whole_episode_grouped_train_test_v1", train_count=1, test_count=1,
                   source_selection_sha256="c" * 64),
        episodes=[dict(episode_uid=str(i), original_collection_status="completed",
                       admission_protocol=PROTOCOL, action_replay_performed=False, frame_stop=600)
                  for i in range(2)])

    def freeze(value=payload):
        path = tmp_path / "selection.json"
        path.write_text(json.dumps(value))
        return SimpleNamespace(data=SimpleNamespace(adapter=ADAPTER, selection_count=2,
            selection_path=lambda root: path, selection_sha256=sha256(path.read_bytes()).hexdigest()))
    return payload, freeze


def test_disabled_does_not_access_data():
    assert producer_admission(None, None) is None


@pytest.mark.parametrize("flags,expected", [([], False), (["--source-label-admission"], True)])
def test_distributed_flag_is_opt_in_and_not_forwarded(monkeypatch, flags, expected):
    base = dict(num_envs=8192, max_updates=2, checkpoint_interval=1, evaluation_interval=2,
                heldout_test_interval=0, disable_success_early_stop=True, browser_enabled=False,
                browser_env_index=0, wandb_enabled=False)
    parse_base = Mock(return_value=SimpleNamespace(**base))
    monkeypatch.setattr(cli, "_parse_args", parse_base)
    monkeypatch.setattr(sys, "argv", ["distributed_train", *flags])
    assert cli._arguments().source_label_admission is expected
    parse_base.assert_called_once_with([])


def test_valid_frozen_proof_is_a_waiver_not_a_pass(selection, tmp_path):
    _, freeze = selection
    config = freeze()
    record = producer_admission(config, tmp_path, enabled=True)
    assert record["selection_sha256"] == config.data.selection_sha256
    assert record["nominal_gate"] == "NOT_EXECUTED"
    assert record["action_replay_performed"] is False
    assert record["learned_evaluation"] == "unchanged"
    assert not any("passed" in key or "success_rate" in key for key in record)


@pytest.mark.parametrize("adapter,synthetic", [("tro_replay", False), ("unknown", False), (ADAPTER, True)])
def test_unsupported_sources_cannot_waive_replay(selection, tmp_path, adapter, synthetic):
    _, freeze = selection
    config = freeze()
    config.data.adapter = adapter
    with pytest.raises(ValueError, match="real source adapter"):
        producer_admission(config, tmp_path, enabled=True, synthetic=synthetic)


@pytest.mark.parametrize("mutation", [
    lambda p: p.pop("producer_success_required"),
    lambda p: p.update(producer_success_required=False),
    lambda p: p.update(adapter_version="old"),
    lambda p: p.update(admission_protocol="replay_passed"),
    lambda p: p.update(action_replay_performed=True),
    lambda p: p.update(dataset_metadata_sha256="missing"),
    lambda p: p.update(source_hashes={}),
    lambda p: p.update(count=3),
    lambda p: p.pop("split"),
    lambda p: p["split"].update(test_count=0),
    lambda p: p["episodes"][0].update(original_collection_status="hard_failure"),
    lambda p: p["episodes"][0].pop("original_collection_status"),
    lambda p: p["episodes"][0].update(action_replay_performed=True),
    lambda p: p["episodes"][0].update(admission_protocol="unknown"),
    lambda p: p["episodes"][0].update(frame_stop=0),
])
def test_invalid_producer_proof_fails_closed(selection, tmp_path, mutation):
    payload, freeze = selection
    mutation(payload)
    with pytest.raises(ValueError, match="invalid producer admission proof"):
        producer_admission(freeze(), tmp_path, enabled=True)


def test_missing_or_corrupt_selection_fails_closed(selection, tmp_path):
    _, freeze = selection
    config = freeze()
    path = config.data.selection_path(tmp_path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        producer_admission(config, tmp_path, enabled=True)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        producer_admission(config, tmp_path, enabled=True)


@dataclass
class Perturbation:
    mode: str = "offset"
    pre_roll_policy_steps: int = 8


@dataclass
class Config:
    perturbation: Perturbation


def gate_args(split="test", enforcement="blocking"):
    return SimpleNamespace(resume_checkpoint=None, evaluation_data_split=split,
        gate_no_offset_mean_error_m=.03, gate_no_offset_success_rate=.96,
        nominal_replay_gate_mode=enforcement)


@pytest.mark.parametrize("split", ["test", "train"])
def test_default_preserves_nominal_callback_and_blocking_failure(split):
    args = gate_args(split)
    result = dict(event="evaluation", mean_flow_error_m=.2, episode_success_rate=.1)
    train, test = Mock(return_value=result), Mock(return_value=result)
    coordinator = SimpleNamespace(pause_recording=nullcontext, emit=Mock())
    with pytest.raises(RuntimeError, match="nominal replay gate failed"):
        cli._run_nominal_gate(args, Config(Perturbation()), coordinator,
                             lambda **kw: nullcontext(), train, test, None)
    selected, other = (test, train) if split == "test" else (train, test)
    assert selected.call_args.kwargs == {"nominal": True}
    other.assert_not_called()
    assert coordinator.emit.call_args.args[0]["nominal_replay_gate_passed"] is False
    if split == "train":
        assert train.call_args.args[0].perturbation == Perturbation("none", 0)


def test_diagnostic_failure_remains_nonblocking():
    coordinator = SimpleNamespace(pause_recording=nullcontext, emit=Mock())
    test = Mock(return_value=dict(event="evaluation", mean_flow_error_m=.2, episode_success_rate=.1))
    cli._run_nominal_gate(gate_args(enforcement="diagnostic"), Config(Perturbation()),
                         coordinator, lambda **kw: nullcontext(), Mock(), test, None)
    test.assert_called_once_with(nominal=True)
    assert coordinator.emit.call_args.args[0]["nominal_replay_gate_passed"] is False


def test_waiver_skips_nominal_but_keeps_real_coordinator_learned_u0_and_ppo(selection, tmp_path):
    _, freeze = selection
    record = producer_admission(freeze(), tmp_path, enabled=True)
    nominal = Mock(side_effect=AssertionError("nominal must not execute"))
    learned = Mock(return_value=dict(event="evaluation", episode_success_rate=.1))
    scope = Mock(side_effect=lambda **kw: nullcontext())
    runner = SimpleNamespace(
        env=SimpleNamespace(needs_reset=False, rollout_state_dict=lambda: {}, load_rollout_state_dict=lambda x: None),
        trainer=SimpleNamespace(model=torch.nn.Linear(1, 1)), observation_callback=None,
        run_update=Mock(side_effect=RuntimeError("PPO reached after learned evaluation")))
    context = SimpleNamespace(primary=True, barrier=lambda: None, reduce=lambda value, op: value)
    events = []
    coordinator = SynchronousCoordinator(runner=runner, context=context,
        artifacts=SimpleNamespace(append_metrics=events.append), tracker=None, contract={},
        checkpoint_kwargs={}, evaluate_train=nominal, evaluate_test=learned,
        periodic_heldout=True, evaluation_scope=scope)
    cli._run_nominal_gate(gate_args(), Config(Perturbation()), coordinator, scope, nominal, nominal, record)
    assert events == []
    scope.assert_not_called()
    with pytest.raises(RuntimeError, match="PPO reached"):
        coordinator.run(max_updates=1, evaluation_interval=1, test_interval=1, checkpoint_interval=1)
    learned.assert_called_once_with()
    nominal.assert_not_called()
    runner.run_update.assert_called_once_with()
    scope.assert_called_once_with(update=0, split="test", policy="learned")
    assert events[0]["update"] == 0
    assert "nominal_replay_gate_passed" not in events[0]


def test_invalid_producer_proof_rejected_before_cuda_context(selection, tmp_path, monkeypatch):
    payload, freeze = selection
    payload["producer_success_required"] = False
    config = freeze()
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_arguments", lambda: SimpleNamespace(expected_world_size=8,
        config=tmp_path / "config.yaml", source_label_admission=True, synthetic_smoke=False))
    monkeypatch.setattr(cli, "load_trajectory_direct_config", lambda *a, **kw: config)
    create_context = Mock(side_effect=AssertionError("GPU context must not be created"))
    monkeypatch.setattr(cli.ParallelContext, "from_torchrun", create_context)
    with pytest.raises(ValueError, match="producer success"):
        cli.main()
    create_context.assert_not_called()
