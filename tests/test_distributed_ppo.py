"""Real Gloo workers: global PPO equals a concatenated serial batch."""

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from parallel_fixtures import recipe_config
from trajectory_tracking.training.parallel import ParallelContext, PPOExecution
from trajectory_tracking.training.ppo import RolloutBatch, create_ppo_trainer

def _worker(rank, world, rendezvous, output, profile, empty_rank):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    context = ParallelContext(torch.device("cpu"), rank=rank, world_size=world)
    try:
        admission = {"source_hash": "a" * 64, "ids": [1, 3, 7]}
        assert context.broadcast(admission if rank == 0 else None) == admission
        config = recipe_config().ppo
        if profile == "benchmark":
            config = replace(config, hidden_dims=(16, 16), update_epochs=2, num_minibatches=7)
        kwargs = dict(observation_dim=191, action_dim=27, config=config,
                      device=torch.device("cpu"), seed=42)
        serial, parallel = create_ppo_trainer(**kwargs), create_ppo_trainer(**kwargs)
        torch.manual_seed(73)
        size = 1031 if profile == "maniptrans" else 19
        observations = torch.randn(size, 191) * 0.1
        _, raw, logp, value = serial.act(observations)
        batch = RolloutBatch(observations, raw, logp, value,
                             value + torch.randn(size) * 0.1, torch.randn(size))
        # Strongly unequal partitions, including zero samples and empty minibatches.
        counts = [0 if empty_rank else 3] + [0] * (world - 2)
        counts.append(size - sum(counts))
        start, stop = sum(counts[:rank]), sum(counts[:rank + 1])
        # Shards are not in original global environment order (asset grouping).
        indices = torch.arange(size).roll(5)[start:stop]
        local = replace(batch, **{f.name: getattr(batch, f.name)[indices] for f in fields(batch)})
        parallel.execution = PPOExecution(context)
        parallel.execution.sample_keys = indices
        context.synchronize_model(parallel.model)
        actual = parallel.update(local)
        # Single-process reference after collectives, using the identical global batch.
        expected = serial.update(batch)
        errors = {}
        for name, tensor in parallel.model.state_dict().items():
            reference = serial.model.state_dict()[name]
            errors[name] = float((tensor - reference).abs().max())
            torch.testing.assert_close(tensor, reference, atol=3e-5, rtol=3e-4)
        assert actual.minibatches == expected.minibatches
        assert actual.actor_learning_rate == expected.actor_learning_rate
        assert abs(actual.approximate_kl - expected.approximate_kl) < 2e-5
        assert abs(actual.explained_variance - expected.explained_variance) < 2e-5
        fingerprints = context.gather({k: v.tolist() for k, v in parallel.model.state_dict().items()})
        assert all(item == fingerprints[0] for item in fingerprints)
        if context.primary:
            torch.save(errors, output)
    finally:
        context.close()


@pytest.mark.parametrize("profile,empty_rank", [("benchmark", False), ("benchmark", True)])
def test_two_rank_serial_parity(tmp_path, profile, empty_rank):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    output = str(tmp_path / "errors.pt")
    mp.spawn(_worker, args=(2, rendezvous, output, profile, empty_rank), nprocs=2, join=True)
    assert Path(output).is_file()


@pytest.mark.parametrize("world_size", [7, 8])
def test_many_rank_serial_parity(tmp_path, world_size):
    mp.spawn(_worker, args=(world_size, (tmp_path / "rendezvous").as_uri(),
                            str(tmp_path / "errors.pt"), "benchmark", True), nprocs=world_size, join=True)


def test_single_context_does_not_initialize_process_group():
    context = ParallelContext(torch.device("cpu"))
    assert context.gather("value") == ["value"]
    assert context.broadcast("value") == "value"
    assert context.device_manifest()["torch_device"] == "cpu"
    assert context.device_manifest()["rank"] == 0
    assert not dist.is_initialized()
    assert context.reduce(torch.tensor(3.0)).item() == 3.0


def test_cuda_collectives_start_only_when_runtime_is_ready(monkeypatch):
    events = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(dist, "init_process_group", lambda backend, **kw: events.append(backend))
    monkeypatch.setattr(dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(str(device)))
    tensor_group = object()

    def new_group(**kwargs):
        events.append(kwargs["backend"])
        return tensor_group

    monkeypatch.setattr(dist, "new_group", new_group)
    context = ParallelContext.from_torchrun()
    assert events == ["gloo"]
    assert context.tensor_group is None
    context.initialize_training_collectives()
    assert events == ["gloo", "barrier", "cuda:1", "nccl"]
    assert context.tensor_group is tensor_group


def test_explicit_device_mapping_preserves_visibility_and_defers_cuda(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    events = []
    monkeypatch.setattr(dist, "init_process_group", lambda backend, **kw: events.append(backend))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: pytest.fail("CUDA initialized before Isaac"))
    context = ParallelContext.from_torchrun(cuda_device_indices=(1, 7))
    assert str(context.device) == "cuda:7"
    assert context.rank == 1
    assert events == ["gloo"]
    import os
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("indices,cpu", [((1,), False), ((1, 1), False), ((-1, 2), False),
                                          ((True, 2), False), ((1, 2), True)])
def test_invalid_device_mapping_fails_before_group_initialization(monkeypatch, indices, cpu):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **kw: pytest.fail("group initialized"))
    with pytest.raises(ValueError, match="device indices"):
        ParallelContext.from_torchrun(cpu=cpu, cuda_device_indices=indices)


@pytest.mark.parametrize("environment_major,expected", [
    (True, [6, 8, 1, 2]), (False, [2, 4, 10, 8]),
])
def test_rollout_keys_preserve_original_global_order(environment_major, expected):
    context = ParallelContext(torch.device("cpu"), world_size=2)
    execution = PPOExecution(context, global_environment_ids=(2, 0))
    valid = torch.tensor([[True, False], [False, True], [True, True]])
    execution.set_rollout_order(valid, environment_major=environment_major)
    assert execution.sample_keys.tolist() == expected
