"""Actual history provenance, same-episode future, and disabled-path parity."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trajectory_tracking.configuration import resolve_trajectory_direct_config
from trajectory_tracking.environment.core import TrajectoryDirectEnvCore
from trajectory_tracking.environment.robot_temporal_observation import (
    ActualJointHistory,
    normalize,
)
from trajectory_tracking.utils.synthetic import (
    LinearRecoveryBackend,
    synthetic_action_spec,
    synthetic_reference_trajectory,
)
from parallel_fixtures import recipe_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/trajectory_tracking/learning"


def load(name="current"):
    config = recipe_config()
    if name == "baseline":
        return replace(config, observation=replace(config.observation,
            robot_joint_temporal_context=False,
            current_reference_robot_target_features=("arm_joint_position",)))
    return config


def task(enabled=True, pre_roll=0, frames=24, full_current_reference=False):
    config = load()
    config = replace(
        config,
        data=replace(config.data, trajectory_count=1),
        observation=replace(config.observation, robot_joint_temporal_context=enabled,
            current_reference_robot_target_features=(("arm_joint_position", "hand_joint_position")
                if full_current_reference else ("arm_joint_position",))),
        start_index=replace(config.start_index, mode="fixed", fixed_index=1),
        perturbation=replace(
            config.perturbation,
            pre_roll_policy_steps=pre_roll,
            mode="random_action" if pre_roll else "none",
        ),
        experiment_plugin=None if pre_roll else config.experiment_plugin,
    )
    reference = synthetic_reference_trajectory(frame_count=frames)
    reference = replace(
        reference,
        joint_position=(torch.arange(frames)[:, None].expand(-1, 27).float() / 100),
    )
    return TrajectoryDirectEnvCore(
        config=config,
        trajectory=reference,
        action_spec=synthetic_action_spec(),
        backend=LinearRecoveryBackend(num_envs=2, device=torch.device("cpu")),
        num_envs=2,
        episode_completion_mode="reference_endpoint_only",
    )


def test_temporal_scope_and_dimensions():
    base, new = load("baseline"), load()
    assert base.observation.feature_dim == 191
    assert new.observation.feature_dim == 427
    assert sum(width for _, width in new.observation.feature_layout()) == 427
    assert "robot_joint_temporal_context" not in base.observation.to_manifest()
    with pytest.raises(TypeError):
        replace(new.observation, robot_joint_temporal_context="true")


def test_history_mask_reset_alias_and_checkpoint():
    history = ActualJointHistory(2, "cpu")
    initial = torch.stack((torch.zeros(27), torch.full((27,), 100.0)))
    history.reset(torch.tensor([0, 1]), initial)
    for step in range(1, 10):
        current = initial + step
        history.advance(current, torch.tensor([True, False]))
        current.fill_(-1000)  # Backend buffer reuse must not mutate stored history.
    torch.testing.assert_close(
        history.features()[0, :, 0], torch.tensor([1.0, 5.0, 7.0, 8.0])
    )
    torch.testing.assert_close(history.features()[1], torch.full((4, 27), 100.0))
    saved = history.state_dict()
    history.reset(torch.tensor([0]), initial + 20)
    assert torch.equal(history.features()[1], saved[1, [8, 4, 2, 1]])
    history.load_state_dict(saved)
    torch.testing.assert_close(history.values, saved)
    with pytest.raises(ValueError):
        history.load_state_dict(saved[:, :8])
    with pytest.raises(ValueError):
        history.load_state_dict(torch.full_like(saved, float("nan")))


def test_observation_actual_past_reference_future_and_read_only():
    env = task()
    reset = env.reset(start_indices=torch.tensor([1, 4]))
    assert reset.observation.shape == (2, 407)
    actual = [env.live_state.joint_position.clone()]
    for _ in range(9):
        env.step(torch.full((2, 27), 0.7), auto_reset=False)
        actual.append(env.live_state.joint_position.clone())
    observation = env.current_observation()
    extra = observation[:, 191:].reshape(2, 8, 27)
    for index, lag in enumerate((8, 4, 2, 1)):
        torch.testing.assert_close(extra[:, index], actual[-1 - lag])
    for index, offset in enumerate((1, 2, 4, 8)):
        expected = (env.reference_cursors + offset).clamp_max(23).float() / 100
        torch.testing.assert_close(
            extra[:, index + 4], expected[:, None].expand(-1, 27)
        )
    before = env.rollout_state_dict()
    torch.testing.assert_close(env.current_observation(), observation)
    torch.testing.assert_close(
        env._joint_history.values, before["episode"]["actual_joint_history"]
    )
    # Evaluation resets can be undone without discarding or contaminating history.
    env.reset(start_indices=torch.tensor([0, 0]), start_index_range="full_reference")
    env.load_rollout_state_dict(before)
    torch.testing.assert_close(env.current_observation(), observation)


def test_old_prefix_and_rewards_unchanged_and_async_reset():
    new, old = task(frames=12), task(False, frames=12)
    starts = torch.tensor([9, 1])
    torch.testing.assert_close(
        new.reset(start_indices=starts).observation[:, :191],
        old.reset(start_indices=starts).observation,
    )
    for _ in range(4):
        action = torch.full((2, 27), 0.3)
        a, b = new.step(action), old.step(action)
        torch.testing.assert_close(a.next_observation[:, :191], b.next_observation)
        torch.testing.assert_close(a.reward, b.reward)
        torch.testing.assert_close(a.terminated, b.terminated)
        if a.reset_started[0]:
            expected = new.live_state.joint_position[0].expand(4, -1)
            torch.testing.assert_close(new._joint_history.features()[0], expected)
    assert new.episode_ids[0] > new.episode_ids[1]


def test_terminal_future_and_history_padding():
    env = task(frames=12)
    reset = env.reset(start_indices=torch.tensor([9, 10]))
    extra = reset.observation[:, 191:].reshape(2, 8, 27)
    torch.testing.assert_close(extra[1, :4], torch.full((4, 27), 0.10))
    torch.testing.assert_close(extra[1, 4:], torch.full((4, 27), 0.11))
    env.step(torch.zeros(2, 27), auto_reset=False)
    terminal = env.current_observation()[:, 191:].reshape(2, 8, 27)
    torch.testing.assert_close(terminal[1, 4:], torch.full((4, 27), 0.11))


def test_normalization_includes_hand_and_uses_per_joint_limits():
    lower = torch.arange(27).float() - 2
    upper = lower + torch.arange(1, 28).float()
    values = lower.expand(2, 8, 27).clone()
    values[:, 1::2] = upper
    result = normalize(values, lower, upper).reshape(2, 8, 27)
    torch.testing.assert_close(result[:, ::2], -torch.ones(2, 4, 27))
    torch.testing.assert_close(result[:, 1::2], torch.ones(2, 4, 27))


def test_future_ragged_bank_identity_no_padding_or_cursor_advance():
    from trajectory_tracking.data.reference import build_reference_episode
    from trajectory_tracking.data.variable_length import VariableLengthReferenceBank

    rows = []
    for index, length in enumerate((12, 20)):
        row = synthetic_reference_trajectory(frame_count=length)
        rows.append(
            replace(
                row,
                trajectory_id=f"row-{index}",
                joint_position=torch.arange(length)[:, None].expand(-1, 27).float()
                + 100 * index,
            )
        )
    bank = VariableLengthReferenceBank(rows)
    episode = build_reference_episode(
        bank,
        start_indices=torch.tensor([9, 9]),
        trajectory_indices=torch.tensor([1, 0]),
        pre_roll_policy_steps=0,
        flow_anchor_offsets_steps=(1, 2, 4, 8, 16),
    )
    cursor = torch.tensor([16, 10])
    actual = episode.joint_position_targets(cursor, (1, 2, 4, 8))
    expected = torch.tensor([[117.0, 118.0, 119.0, 119.0], [11.0, 11.0, 11.0, 11.0]])
    torch.testing.assert_close(actual, expected[:, :, None].expand(-1, -1, 27))
    assert cursor.tolist() == [16, 10]
    assert episode.trajectory_indices.tolist() == [1, 0]


def test_pre_roll_tracks_actual_states_and_checkpoint_rejects_old_schema():
    env = task(pre_roll=2)
    env.reset()
    torch.testing.assert_close(
        env._joint_history.values[:, 0], env.live_state.joint_position
    )
    torch.testing.assert_close(
        env._joint_history.features()[:, 0], torch.full((2, 27), 0.01)
    )
    env.step(torch.full((2, 27), 0.4))
    state = env.rollout_state_dict()
    old = task(False, pre_roll=2)
    with pytest.raises(ValueError, match="schema"):
        old.load_rollout_state_dict(state)


def test_407d_ppo_rollout_update_and_policy_checkpoint():
    from trajectory_tracking.training.ppo import create_ppo_trainer
    from trajectory_tracking.training.trainer import PPOOverfitRunner

    torch.set_num_threads(2)
    env = task()
    recipe = replace(
        env.config.ppo,
        hidden_dims=(16,),
        update_epochs=1,
        num_minibatches=1,
        rollout_steps_per_env=4,
    )
    trainer = create_ppo_trainer(
        observation_dim=407,
        action_dim=27,
        config=recipe,
        device=torch.device("cpu"),
        seed=42,
    )
    stats = PPOOverfitRunner(env=env, trainer=trainer).run_update()
    assert stats.valid_transition_count > 0
    assert torch.isfinite(torch.tensor(stats.mean_reward))
    restored = create_ppo_trainer(
        observation_dim=407,
        action_dim=27,
        config=recipe,
        device=torch.device("cpu"),
        seed=42,
    )
    restored.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(
        restored.deterministic_action(env.current_observation()),
        trainer.deterministic_action(env.current_observation()),
    )


def test_427_config_is_only_temporal_and_complete_current_reference_overlay():
    base, full = load("baseline"), load()
    assert full == replace(base, observation=replace(base.observation,
        robot_joint_temporal_context=True,
        current_reference_robot_target_features=("arm_joint_position", "hand_joint_position")))
    assert full.observation.feature_dim == 427
    assert full.data.trajectory_count == 2402
    assert full.data.selection_sha256 == "e809ccceba0ebfac721d5d3da7aab5baa1e84e5800d5293010884d4721e631ad"


def test_427_actual_history_complete_current_reference_and_checkpoint_ppo():
    from trajectory_tracking.training.ppo import create_ppo_trainer
    from trajectory_tracking.training.trainer import PPOOverfitRunner
    full, old = task(full_current_reference=True), task()
    starts = torch.tensor([1, 4])
    a, b = full.reset(start_indices=starts), old.reset(start_indices=starts)
    assert a.observation.shape == (2, 427)
    torch.testing.assert_close(a.observation[:, :97], b.observation[:, :97])
    torch.testing.assert_close(a.observation[:, 117:211], b.observation[:, 97:191])
    torch.testing.assert_close(a.observation[:, 97:117], starts[:, None].expand(-1, 20).float() / 100)
    actual = [full.live_state.joint_position.clone()]
    for _ in range(9):
        full.step(torch.full((2, 27), .7), auto_reset=False)
        actual.append(full.live_state.joint_position.clone())
    observation = full.current_observation()
    extra = observation[:, 211:].reshape(2, 8, 27)
    for index, lag in enumerate((8, 4, 2, 1)):
        torch.testing.assert_close(extra[:, index], actual[-1-lag])
    torch.testing.assert_close(observation[:, 97:117], full.reference_cursors[:, None].expand(-1, 20).float()/100)
    state = full.rollout_state_dict()
    full.reset(start_indices=starts)
    full.load_rollout_state_dict(state)
    torch.testing.assert_close(full.current_observation(), observation)
    ppo = replace(full.config.ppo, hidden_dims=(16,), rollout_steps_per_env=4, update_epochs=1, num_minibatches=1)
    trainer = create_ppo_trainer(observation_dim=427, action_dim=27, config=ppo, device=torch.device("cpu"), seed=42)
    stats = PPOOverfitRunner(env=full, trainer=trainer).run_update()
    assert stats.valid_transition_count > 0 and torch.isfinite(torch.tensor(stats.mean_reward))
    restored = create_ppo_trainer(observation_dim=427, action_dim=27, config=ppo, device=torch.device("cpu"), seed=42)
    restored.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(restored.deterministic_action(full.current_observation()),
                               trainer.deterministic_action(full.current_observation()))
