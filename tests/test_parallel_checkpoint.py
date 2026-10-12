"""Completed rank bundles are resumable; incomplete/tampered bundles are not."""

from dataclasses import replace

import pytest
import torch

from parallel_fixtures import recipe_config
from trajectory_tracking.environment.core import TrajectoryDirectEnvCore
from trajectory_tracking.training.artifacts import load_ppo_checkpoint, load_ppo_policy_checkpoint
from trajectory_tracking.training.parallel import ParallelContext, PPOExecution
from trajectory_tracking.training.parallel.checkpoint import load_checkpoint, save_checkpoint
from trajectory_tracking.training.ppo import create_ppo_trainer
from trajectory_tracking.training.run_control import TrainingControlSpec
from trajectory_tracking.training.trainer import PPOOverfitRunner
from trajectory_tracking.utils.synthetic import (
    LinearRecoveryBackend, synthetic_action_spec, synthetic_reference_trajectory,
)

def make_runner():
    torch.set_num_threads(1)
    config = recipe_config()
    config = replace(config, experiment_plugin=None, data=replace(config.data, trajectory_count=1),
                     ppo=replace(config.ppo, hidden_dims=(16, 16), rollout_steps_per_env=4,
                                 update_epochs=2, num_minibatches=4))
    device = torch.device("cpu")
    action_spec = synthetic_action_spec()
    env = TrajectoryDirectEnvCore(config=config, trajectory=synthetic_reference_trajectory(frame_count=130),
                                  action_spec=action_spec, backend=LinearRecoveryBackend(num_envs=4, device=device),
                                  num_envs=4)
    trainer = create_ppo_trainer(observation_dim=config.observation.feature_dim, action_dim=27,
                                config=config.ppo, device=device, seed=42)
    context = ParallelContext(device)
    trainer.execution = PPOExecution(context)
    runner = PPOOverfitRunner(env, trainer)
    control = TrainingControlSpec(max_updates=5, evaluation_interval=1, target_success_rate=0.95,
                                  success_patience=3, success_early_stop_enabled=False)
    kwargs = dict(config=config, action_spec=action_spec, trajectory_id=env.trajectory.trajectory_id,
                  scene_manifest={"backend": "synthetic"}, adapter_version="test_adapter",
                  num_envs=4, control=control, consecutive_successful_evaluations=0)
    contract = {"world_size": 1, "config_hash": config.compatibility_hash, "global_num_envs": 4}
    return runner, context, kwargs, contract


def test_bundle_resume_next_update_and_standalone_policy(tmp_path):
    runner, context, kwargs, contract = make_runner()
    runner.run_update()
    directory = save_checkpoint(tmp_path / "update_1", context=context, contract=contract,
                                runner=runner, update=1, checkpoint_kwargs=kwargs)
    expected = runner.run_update()
    restored, _, _, _ = make_runner()
    assert load_checkpoint(directory, context=context, contract=contract, runner=restored) == 1
    actual = restored.run_update()
    assert actual.valid_transition_count == expected.valid_transition_count
    for key, tensor in runner.trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, restored.trainer.model.state_dict()[key], atol=0, rtol=0)
    policy_args = {k: v for k, v in kwargs.items() if k in (
        "config", "action_spec", "trajectory_id", "scene_manifest", "adapter_version")}
    metadata = load_ppo_policy_checkpoint(directory / "policy.pt", trainer=restored.trainer, **policy_args)
    assert metadata.update == 1
    with pytest.raises(ValueError, match="complete checkpoint directory"):
        load_ppo_checkpoint(directory / "policy.pt", trainer=restored.trainer,
                            num_envs=4, control=kwargs["control"], **policy_args)


def test_bundle_rejects_world_size_tampering_and_missing_completion(tmp_path):
    runner, context, kwargs, contract = make_runner()
    runner.run_update()
    directory = save_checkpoint(tmp_path / "update_1", context=context, contract=contract,
                                runner=runner, update=1, checkpoint_kwargs=kwargs)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_checkpoint(directory, context=context, contract=dict(contract, world_size=8), runner=runner)
    rank_file = directory / "rank_000.pt"
    rank_file.write_bytes(rank_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(directory, context=context, contract=contract, runner=runner)
    (directory / "complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(directory, context=context, contract=contract, runner=runner)
