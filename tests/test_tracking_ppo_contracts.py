"""PPO numerical regressions ported from the retired flat implementation."""

from dataclasses import replace
import unittest
import torch
from trajectory_tracking.training.ppo import (
    PPOConfig,
    PPOTrainer,
    PPOUpdateStats,
    RolloutBatch,
    RolloutBuffer,
)


class IndependentPPOTest(unittest.TestCase):
    @staticmethod
    def config() -> PPOConfig:
        return PPOConfig(
            rollout_steps_per_env=4,
            update_epochs=2,
            num_minibatches=2,
            actor_learning_rate=0.0003,
            critic_learning_rate=0.0003,
            learner_reward_scale=1.0,
            learner_failure_penalty=1.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.1,
            clip_value_loss=True,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
            target_kl=0.2,
            actor_lr_schedule="constant",
            actor_learning_rate_min=1e-05,
            actor_learning_rate_max=0.0003,
            actor_learning_rate_adaptation_factor=1.5,
            actor_lr_clip_fraction_threshold=0.5,
            arm_initial_std=0.05,
            hand_initial_std=0.05,
            hidden_dims=(16, 16),
        )

    def test_one_small_update_is_finite_and_changes_actor(self) -> None:
        config = self.config()
        device = torch.device("cpu")
        trainer = PPOTrainer(
            observation_dim=8, action_dim=27, config=config, device=device, seed=3
        )
        buffer = RolloutBuffer(
            steps=4, num_envs=4, observation_dim=8, action_dim=27, device=device
        )
        observation = torch.randn(4, 8)
        before = trainer.model.actor_mean.weight.detach().clone()
        for index in range(4):
            raw, pre_tanh, log_prob, value = trainer.act(observation)
            reward = -raw.square().mean(dim=-1) + 0.1 * observation[:, 0]
            buffer.add(
                observation=observation,
                pre_tanh_action=pre_tanh,
                old_log_prob=log_prob,
                old_value=value,
                reward=reward,
                terminated=torch.full((4,), index == 3, dtype=torch.bool),
                truncated=torch.zeros(4, dtype=torch.bool),
                valid=torch.ones(4, dtype=torch.bool),
            )
            observation = torch.randn(4, 8)
        batch = buffer.finish(
            last_value=torch.zeros(4), gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        stats = trainer.update(batch)
        values = (
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.approximate_kl,
            stats.clip_fraction,
            stats.gradient_norm,
        )
        self.assertTrue(torch.isfinite(torch.tensor(values)).all().item())
        self.assertFalse(torch.equal(before, trainer.model.actor_mean.weight))

    def test_update_uses_exactly_the_configured_nonempty_minibatches(self) -> None:
        config = replace(
            self.config(),
            update_epochs=1,
            num_minibatches=8,
            target_kl=100.0,
            actor_lr_clip_fraction_threshold=1.0,
        )
        trainer = PPOTrainer(
            observation_dim=8,
            action_dim=27,
            config=config,
            device=torch.device("cpu"),
            seed=7,
        )
        observation = torch.randn(10, 8)
        _, pre_tanh, old_log_prob, old_value = trainer.act(observation)
        returns = torch.linspace(-1.0, 1.0, 10)
        stats = trainer.update(
            RolloutBatch(
                observations=observation,
                pre_tanh_actions=pre_tanh,
                old_log_prob=old_log_prob,
                old_value=old_value,
                returns=returns,
                advantages=returns - old_value,
            )
        )
        self.assertEqual(stats.minibatches, 8)
        self.assertFalse(stats.early_stopped)

    def test_nonterminal_rollout_bootstraps_without_marking_truncation(self) -> None:
        buffer = RolloutBuffer(
            steps=1,
            num_envs=1,
            observation_dim=2,
            action_dim=27,
            device=torch.device("cpu"),
        )
        buffer.add(
            observation=torch.zeros(1, 2),
            pre_tanh_action=torch.zeros(1, 27),
            old_log_prob=torch.zeros(1),
            old_value=torch.tensor([2.0]),
            reward=torch.tensor([1.0]),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            valid=torch.tensor([True]),
        )
        batch = buffer.finish(
            last_value=torch.tensor([3.0]), gamma=0.9, gae_lambda=0.95
        )
        self.assertTrue(torch.allclose(batch.advantages, torch.tensor([1.7])))
        self.assertTrue(torch.allclose(batch.returns, torch.tensor([3.7])))
        self.assertEqual(batch.observations.data_ptr(), buffer.observations.data_ptr())
        self.assertEqual(
            batch.pre_tanh_actions.data_ptr(), buffer.pre_tanh_actions.data_ptr()
        )

    def test_rollout_buffer_excludes_inactive_slots_and_can_finish_early(self) -> None:
        buffer = RolloutBuffer(
            steps=3,
            num_envs=2,
            observation_dim=2,
            action_dim=27,
            device=torch.device("cpu"),
        )
        for reward, terminated, valid in (
            ([1.0, 2.0], [False, True], [True, True]),
            ([3.0, 999.0], [True, False], [True, False]),
        ):
            buffer.add(
                observation=torch.zeros(2, 2),
                pre_tanh_action=torch.zeros(2, 27),
                old_log_prob=torch.zeros(2),
                old_value=torch.zeros(2),
                reward=torch.tensor(reward),
                terminated=torch.tensor(terminated),
                truncated=torch.zeros(2, dtype=torch.bool),
                valid=torch.tensor(valid),
            )
        batch = buffer.finish(last_value=torch.zeros(2), gamma=1.0, gae_lambda=1.0)
        self.assertEqual(batch.batch_size, 3)
        self.assertTrue(torch.equal(batch.advantages, torch.tensor([4.0, 2.0, 3.0])))
