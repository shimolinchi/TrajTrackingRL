"""Mixed incremental-arm contracts, asynchronous resets and PPO integration."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from parallel_fixtures import recipe_config
from trajectory_tracking.environment.actions import DirectActionProjector, DirectActionSpec
from trajectory_tracking.environment.incremental_arm_action import ArmTargetIncrementProjector
from trajectory_tracking.environment.core import TrajectoryDirectEnvCore
from trajectory_tracking.configuration import resolve_trajectory_direct_config
from trajectory_tracking.utils.synthetic import (
    LinearRecoveryBackend, synthetic_action_spec, synthetic_reference_trajectory,
)


def increment_spec():
    return DirectActionSpec(0, mode="arm_target_increment", arm_target_speed_rad_s=1.5)


def environment(*, incremental=True, frames=100, pre_roll=0):
    config = recipe_config()
    config = replace(config, data=replace(config.data, trajectory_count=1),
                     direct_action=increment_spec() if incremental else DirectActionSpec(0),
                     start_index=replace(config.start_index, mode="fixed", fixed_index=1),
                     perturbation=replace(config.perturbation, pre_roll_policy_steps=pre_roll,
                                          mode="random_action" if pre_roll else "none"))
    reference = synthetic_reference_trajectory(frame_count=frames)
    reference = replace(reference, joint_position=torch.full((frames, 27), 0.15))
    return TrajectoryDirectEnvCore(config=config, trajectory=reference,
        action_spec=synthetic_action_spec(),
        backend=LinearRecoveryBackend(num_envs=2, device=torch.device("cpu")), num_envs=2,
        episode_completion_mode="reference_endpoint_only")


@pytest.mark.parametrize("batch", [1, 3, 19])
def test_formula_hand_parity_clipping_and_zero_hold(batch):
    spec = synthetic_action_spec()
    projector = ArmTargetIncrementProjector(spec, increment_spec())
    absolute = DirectActionProjector(spec, DirectActionSpec(0))
    raw = torch.linspace(-3, 3, batch * 27).reshape(batch, 27)
    previous = torch.full_like(raw, 0.99)
    result = projector.apply(raw, previous_joint_target=previous)
    expected = (previous[:, :7] + 0.025 * raw[:, :7].clamp(-1, 1)).clamp(-1, 1)
    torch.testing.assert_close(result.executed_action[:, :7], expected, rtol=0, atol=0)
    assert torch.equal(result.executed_action[:, 7:], absolute.apply(raw).executed_action[:, 7:])
    assert torch.equal(result.raw_action, raw)
    assert torch.equal(previous, torch.full_like(raw, 0.99))
    held = projector.apply(torch.zeros_like(raw), previous_joint_target=result.executed_action)
    assert torch.equal(held.executed_action[:, :7], result.executed_action[:, :7])


def test_limit_saturation_has_no_windup_and_inverse_is_reachable_only():
    projector = ArmTargetIncrementProjector(synthetic_action_spec(), increment_spec())
    previous = torch.full((2, 27), 0.99)
    for _ in range(100):
        previous = projector.apply(torch.ones_like(previous), previous_joint_target=previous).executed_action
    reversed_target = projector.apply(-torch.ones_like(previous), previous_joint_target=previous).executed_action
    torch.testing.assert_close(reversed_target[:, :7], torch.full((2, 7), 0.975))
    torch.testing.assert_close(projector.normalize(reversed_target, previous_joint_target=previous),
                               -torch.ones_like(previous), atol=2e-6, rtol=0)
    with pytest.raises(ValueError, match="one-step increment"):
        projector.normalize(torch.zeros_like(previous), previous_joint_target=previous)
    for bad in (None, torch.zeros(2, 7), previous.double(), torch.full_like(previous, float("nan"))):
        with pytest.raises(ValueError):
            projector.apply(torch.zeros_like(previous), previous_joint_target=bad)


def previous_target_observation(env):
    offset = 0
    for name, width in env.config.observation.feature_layout():
        if name == "previous_policy_action":
            return env.current_observation()[:, offset:offset + width]
        offset += width
    raise AssertionError("previous target is missing")


@pytest.mark.parametrize("pre_roll", [0, 2])
def test_target_accumulation_reset_observation_and_checkpoint(pre_roll):
    env = environment(pre_roll=pre_roll)
    env.reset(start_indices=torch.tensor([1, 5]))
    assert env.current_observation().shape == (2, 427)
    for _ in range(4):
        previous = env.last_executed_action.clone()
        torch.testing.assert_close(previous_target_observation(env),
                                   env.direct_action.normalize_executed_target(previous))
        env.current_observation()
        assert torch.equal(previous, env.last_executed_action)
        result = env.step(torch.ones(2, 27), auto_reset=False)
        torch.testing.assert_close(result.action.executed_action[:, :7],
                                   (previous[:, :7] + 0.025).clamp(-1, 1))
    # Restore the integrator directly, without an intervening observation/reset.
    snapshot = env.rollout_state_dict()
    restored = environment(pre_roll=pre_roll)
    restored.load_rollout_state_dict(snapshot)
    actual = restored.step(-torch.ones(2, 27), auto_reset=False)
    expected = env.step(-torch.ones(2, 27), auto_reset=False)
    torch.testing.assert_close(actual.action.executed_action, expected.action.executed_action, rtol=0, atol=0)
    env.reset(start_indices=torch.tensor([2, 4]))
    if pre_roll == 0:
        torch.testing.assert_close(env.last_executed_action, torch.full((2, 27), 0.15))


@pytest.mark.parametrize("pre_roll", [0, 2])
def test_async_resets_and_reward_parity_with_identical_executed_targets(pre_roll):
    mixed = environment(frames=20, pre_roll=pre_roll)
    absolute = environment(incremental=False, frames=20, pre_roll=pre_roll)
    starts = torch.tensor([14, 1])
    mixed.reset(start_indices=starts)
    absolute.reset(start_indices=starts)
    for i in range(12):
        raw = torch.full((2, 27), 0.4 if i % 2 else -0.2)
        target = mixed.direct_action.apply(raw, previous_joint_target=mixed.last_executed_action).executed_action
        result = mixed.step(raw)
        baseline = absolute.step(absolute.direct_action.normalize(target))
        torch.testing.assert_close(result.action.executed_action, baseline.action.executed_action)
        torch.testing.assert_close(result.reward, baseline.reward, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result.action_smoothness_diagnostics.penalty,
                                   baseline.action_smoothness_diagnostics.penalty, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mixed.last_executed_action, absolute.last_executed_action)
        torch.testing.assert_close(previous_target_observation(mixed),
                                   mixed.direct_action.normalize_executed_target(mixed.last_executed_action))
    assert mixed.episode_ids[0] > mixed.episode_ids[1]


def test_configuration_scope_and_reject_invalid_combinations():
    config = recipe_config()
    path = Path(__file__).resolve().parents[1] / "src/trajectory_tracking/resources"
    base = replace(config, direct_action=DirectActionSpec(0))
    assert config.direct_action == increment_spec()
    assert config.observation.feature_dim == base.observation.feature_dim == 427
    assert config.compatibility_hash != base.compatibility_hash
    assert config.to_manifest()["observation"]["previous_policy_action"]["source"] == "rollout.previous_joint_target"
    for speed in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            replace(increment_spec(), arm_target_speed_rad_s=speed)
    with pytest.raises(ValueError):
        DirectActionSpec(0, arm_target_speed_rad_s=1.5)
    with pytest.raises(ValueError, match="previous target observation"):
        replace(config, observation=replace(config.observation, include_previous_policy_action=False))
    with pytest.raises(ValueError, match="60 Hz"):
        replace(config, timing=replace(config.timing, control_hz=30,
                                       flow_anchor_offsets_steps=(1, 15, 30, 45, 60)))


def test_complete_checkpoint_preserves_next_ppo_update(tmp_path, monkeypatch):
    import test_parallel_checkpoint as checkpoint_tests
    config = replace(recipe_config(), direct_action=increment_spec())
    monkeypatch.setattr(checkpoint_tests, "recipe_config", lambda: config)
    checkpoint_tests.test_bundle_resume_next_update_and_standalone_policy(tmp_path)


def test_real_ppo_update_retains_raw_sample_likelihood():
    from trajectory_tracking.training.ppo import PPOTrainer
    from trajectory_tracking.training.trainer import PPOOverfitRunner
    env = environment()
    trainer = PPOTrainer(observation_dim=427, action_dim=27,
        config=replace(env.config.ppo, hidden_dims=(16, 16)), device=torch.device("cpu"), seed=42)
    samples, log_probs = [], []
    original_act, original_update = trainer.act, trainer.update
    def act(obs):
        result = original_act(obs)
        samples.append(result[1].clone())
        log_probs.append(result[2].clone())
        return result
    def update(batch):
        torch.testing.assert_close(batch.pre_tanh_actions, torch.stack(samples).reshape(-1, 27))
        torch.testing.assert_close(batch.old_log_prob, torch.stack(log_probs).reshape(-1))
        return original_update(batch)
    trainer.act, trainer.update = act, update
    stats = PPOOverfitRunner(env, trainer).run_update()
    assert stats.valid_transition_count == 64
    assert torch.isfinite(trainer.model.actor_mean.weight).all()
