"""The single production PPO implementation for trajectory-tracking training.

The current benchmark is the learner promoted from experiment 85.  Its action
distribution, running normalizers, network initialization, loss, and adaptive
learning-rate behavior live here instead of behind an experiment selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Literal

import torch
from torch import Tensor, nn


# ``TRAJECTORY_PPO_IMPLEMENTATION_VERSION`` is intentionally frozen because it
# participates in the compatibility hash of the already-running experiment-85
# lineage.  The benchmark profile below records the behavioral override that
# was present in that lineage's resolved manifest.
TRAJECTORY_PPO_IMPLEMENTATION_VERSION = "trajectory_ppo_v4_terminal_failure_penalty"
TRAJECTORY_PPO_BENCHMARK_IMPLEMENTATION = "rltracking_no_reference_ablation_v1"
TRAJECTORY_PPO_BENCHMARK_SOURCE_REVISION = (
    "47af8a8fe83ae9a8a109afb799582f1d8c081c16"
)
TRAJECTORY_PPO_BENCHMARK_HIDDEN_DIMS = (512, 512, 512, 256, 512, 128, 64)
TRAJECTORY_PPO_BENCHMARK_INITIAL_STD = exp(-1.0)


@dataclass(frozen=True, slots=True)
class PPOBenchmarkLineageSpec:
    """Frozen serialization marker for experiment 85's promoted learner.

    The source YAML key remains ``experiment_plugin`` solely so existing
    experiment-85 resolved configs and checkpoints keep their compatibility
    hash.  It is no longer a runtime plugin or a learner/task switch.  The
    configuration layer may retain this learner-lineage marker while recording
    an explicit observation-only extension; the unextended manifest remains
    byte-for-byte compatible with experiment 85.
    """

    name: Literal["rltracking_no_reference_ablation_v1"]
    learner_enabled: bool
    task_runtime_enabled: bool
    evaluation_mode: Literal["configured_task"] = "configured_task"

    def __post_init__(self) -> None:
        expected = (
            self.name == TRAJECTORY_PPO_BENCHMARK_IMPLEMENTATION
            and self.learner_enabled is True
            and self.task_runtime_enabled is False
            and self.evaluation_mode == "configured_task"
        )
        if not expected:
            raise ValueError(
                "experiment_plugin is an archived compatibility key and only "
                "the promoted experiment-85 benchmark profile is supported"
            )

    def to_manifest(self) -> dict[str, object]:
        """Preserve experiment 85's exact manifest and compatibility hash."""

        return {
            "name": self.name,
            "source_revision": TRAJECTORY_PPO_BENCHMARK_SOURCE_REVISION,
            "variant": "rltracking_learner__benchmark_task",
            "learner_enabled": True,
            "task_runtime_enabled": False,
            "observation_lock": {
                "robot_reference_state_added": False,
                "required_object_target_representation": "error_reference_delta",
                "required_include_reference_robot_state": False,
                "semantic": "experiment_56_observation_D_unchanged",
            },
            "common_evaluation": {
                "episode_completion": "configured",
                "trajectory_assignment": "configured",
                "start_indices": "configured",
            },
            "learner_profile": {
                "source": "RLTracking stage1-formal",
                "action_distribution": "clipped_unsquashed_normal",
                "observation_running_normalization": True,
                "value_running_normalization": True,
                "observation_clip": 5.0,
                "value_clip": 5.0,
                "policy_shift_guard": "adaptive_lr_only_no_update_abort",
                "adaptive_kl_high_multiplier": 2.0,
                "bounds_loss_coefficient": 1.0e-4,
                "actor_mean_initialization": "pytorch_default",
                "minibatch_target_size": 5000,
            },
            "task_runtime_profile": {"source": "experiment_56_benchmark"},
        }


@dataclass(frozen=True, slots=True)
class PPOConfig:
    rollout_steps_per_env: int
    update_epochs: int
    num_minibatches: int
    actor_learning_rate: float
    critic_learning_rate: float
    learner_reward_scale: float
    learner_failure_penalty: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    target_kl: float
    actor_lr_schedule: Literal["constant", "adaptive_kl"]
    actor_learning_rate_min: float
    actor_learning_rate_max: float
    actor_learning_rate_adaptation_factor: float
    actor_lr_clip_fraction_threshold: float
    arm_initial_std: float
    hand_initial_std: float
    hidden_dims: tuple[int, ...]
    activation: Literal["elu", "relu", "tanh"] = "elu"
    clip_value_loss: bool = True
    learner_profile: str = "benchmark"

    def __post_init__(self) -> None:
        for name in ("rollout_steps_per_env", "update_epochs", "num_minibatches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive = (
            "actor_learning_rate",
            "critic_learning_rate",
            "learner_reward_scale",
            "clip_coef",
            "value_loss_coef",
            "max_grad_norm",
            "target_kl",
            "actor_learning_rate_min",
            "actor_learning_rate_max",
            "actor_learning_rate_adaptation_factor",
            "actor_lr_clip_fraction_threshold",
            "arm_initial_std",
            "hand_initial_std",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError("entropy_coef must be finite and non-negative")
        if (
            not isfinite(self.learner_failure_penalty)
            or self.learner_failure_penalty < 0
        ):
            raise ValueError("learner_failure_penalty must be finite and non-negative")
        if self.learner_reward_scale > 1.0:
            raise ValueError("learner_reward_scale must not exceed 1")
        if not 0 <= self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma must be in [0,1] and gae_lambda in (0,1]")
        if self.clip_coef >= 1.0:
            raise ValueError("clip_coef must be less than 1")
        if self.actor_lr_schedule not in ("constant", "adaptive_kl"):
            raise ValueError("unsupported actor_lr_schedule")
        if not (
            self.actor_learning_rate_min
            <= self.actor_learning_rate
            <= self.actor_learning_rate_max
        ):
            raise ValueError(
                "actor_learning_rate must be within its configured min/max"
            )
        if self.actor_learning_rate_adaptation_factor <= 1.0:
            raise ValueError(
                "actor_learning_rate_adaptation_factor must be greater than 1"
            )
        if self.actor_lr_clip_fraction_threshold > 1.0:
            raise ValueError("actor_lr_clip_fraction_threshold must not exceed 1")
        hidden = tuple(self.hidden_dims)
        if not hidden or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in hidden
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if self.activation not in ("elu", "relu", "tanh"):
            raise ValueError("unsupported PPO activation")
        if not isinstance(self.clip_value_loss, bool):
            raise TypeError("clip_value_loss must be bool")
        object.__setattr__(self, "hidden_dims", hidden)
        if self.learner_profile not in ("benchmark",):
            raise ValueError("unsupported PPO learner_profile")

    def to_manifest(self) -> dict[str, object]:
        payload = {
            "algorithm": "ppo",
            "implementation_version": TRAJECTORY_PPO_IMPLEMENTATION_VERSION,
            "rollout_boundary": "fixed_steps_with_asynchronous_environment_reset",
            "rollout_steps_per_env": self.rollout_steps_per_env,
            "update_epochs": self.update_epochs,
            "num_minibatches": self.num_minibatches,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "learner_reward_scale": self.learner_reward_scale,
            "learner_failure_penalty": self.learner_failure_penalty,
            "learner_reward_formula": (
                "environment_reward*learner_reward_scale-"
                "learner_failure_penalty*episode_failed"
            ),
            "episode_failed_definition": ("episode_completed_and_not_episode_success"),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
            "clip_value_loss": self.clip_value_loss,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "actor_learning_rate_schedule": {
                "mode": self.actor_lr_schedule,
                "minimum": self.actor_learning_rate_min,
                "maximum": self.actor_learning_rate_max,
                "adaptation_factor": self.actor_learning_rate_adaptation_factor,
                "clip_fraction_threshold": (self.actor_lr_clip_fraction_threshold),
                "decrease_when": ("observed_kl>target_kl_or_clip_fraction>threshold"),
                "increase_when": (
                    "observed_kl<0.5*target_kl_and_clip_fraction<0.5*threshold"
                ),
            },
            "minibatch_partition": "exact_nonempty_chunks",
            "policy_shift_guard": "pre_post_minibatch_kl_or_clip_fraction",
            "initial_std": {
                "arm": self.arm_initial_std,
                "hand": self.hand_initial_std,
            },
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
        }
        return payload


def _activation(name: str) -> type[nn.Module]:
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[name]


def _mlp(
    input_dim: int, hidden_dims: tuple[int, ...], activation: str
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    activation_type = _activation(activation)
    for next_width in hidden_dims:
        layers.extend((nn.Linear(width, next_width), activation_type()))
        width = next_width
    return nn.Sequential(*layers)


def _normal_log_prob(mean: Tensor, std: Tensor, raw_action: Tensor) -> Tensor:
    return (
        -0.5
        * (
            ((raw_action - mean) / std).square()
            + 2.0 * torch.log(std)
            + log(2.0 * torch.pi)
        )
    ).sum(dim=-1)


def _policy_ratio_diagnostics(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    *,
    clip_coef: float,
) -> tuple[Tensor, Tensor, Tensor]:
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).to(torch.float32).mean()
    return ratio, approximate_kl, clip_fraction


class ActorCritic(nn.Module):
    """Benchmark actor/value MLPs with checkpointed running statistics."""

    def __init__(
        self, observation_dim: int, action_dim: int, config: PPOConfig
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim != 27:
            raise ValueError("PPO requires positive observations and 27 actions")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor_body = _mlp(observation_dim, config.hidden_dims, config.activation)
        self.actor_mean = nn.Linear(config.hidden_dims[-1], action_dim)
        self.critic_body = _mlp(observation_dim, config.hidden_dims, config.activation)
        self.critic_value = nn.Linear(config.hidden_dims[-1], 1)
        initial_std = torch.full((action_dim,), config.hand_initial_std)
        initial_std[:7] = config.arm_initial_std
        self.log_std = nn.Parameter(initial_std.log())
        # Match the stage1-formal learner used by experiment 85.  Float64
        # accumulators reduce drift; buffers make normalization checkpointed.
        self.register_buffer(
            "observation_mean", torch.zeros(observation_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "observation_variance", torch.ones(observation_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "observation_count", torch.ones((), dtype=torch.float64)
        )
        self.register_buffer("value_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("value_variance", torch.ones((), dtype=torch.float64))
        self.register_buffer("value_count", torch.ones((), dtype=torch.float64))

    def distribution_parameters(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.actor_mean(self.actor_body(observation))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def value_normalized(self, observation: Tensor) -> Tensor:
        return self.critic_value(self.critic_body(observation)).squeeze(-1)

    @torch.no_grad()
    def update_running_statistics(
        self,
        observations: Tensor,
        returns: Tensor,
    ) -> None:
        self._merge_statistics(
            observations,
            mean=self.observation_mean,
            variance=self.observation_variance,
            count=self.observation_count,
        )
        self._merge_statistics(
            returns,
            mean=self.value_mean,
            variance=self.value_variance,
            count=self.value_count,
        )

    @staticmethod
    def _merge_statistics(
        values: Tensor,
        *,
        mean: Tensor,
        variance: Tensor,
        count: Tensor,
    ) -> None:
        if values.shape[0] <= 0:
            raise ValueError("running-normalization batch must not be empty")
        values64 = values.to(dtype=torch.float64)
        batch_mean = values64.mean(dim=0)
        batch_variance = values64.var(dim=0, unbiased=True)
        batch_count = values64.new_tensor(float(values.shape[0]))
        delta = batch_mean - mean
        total = count + batch_count
        new_mean = mean + delta * batch_count / total
        combined = (
            variance * count
            + batch_variance * batch_count
            + delta.square() * count * batch_count / total
        ) / total
        mean.copy_(new_mean)
        variance.copy_(combined.clamp_min(1.0e-6))
        count.copy_(total)


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    observations: Tensor
    pre_tanh_actions: Tensor
    old_log_prob: Tensor
    old_value: Tensor
    returns: Tensor
    advantages: Tensor

    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]


class RolloutBuffer:
    """One bounded synchronous ``[T,N]`` rollout segment."""

    def __init__(
        self,
        *,
        steps: int,
        num_envs: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        environment_major: bool = False,
    ) -> None:
        if steps <= 0 or num_envs <= 0 or observation_dim <= 0 or action_dim != 27:
            raise ValueError("invalid rollout buffer dimensions")
        self.steps = steps
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.environment_major = environment_major
        self.observations = torch.empty(
            steps, num_envs, observation_dim, dtype=torch.float32, device=device
        )
        self.pre_tanh_actions = torch.empty(
            steps, num_envs, action_dim, dtype=torch.float32, device=device
        )
        self.old_log_prob = torch.empty(
            steps, num_envs, dtype=torch.float32, device=device
        )
        self.old_value = torch.empty(
            steps, num_envs, dtype=torch.float32, device=device
        )
        self.rewards = torch.empty(steps, num_envs, dtype=torch.float32, device=device)
        self.terminated = torch.empty(steps, num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.empty(steps, num_envs, dtype=torch.bool, device=device)
        self.valid = torch.empty(steps, num_envs, dtype=torch.bool, device=device)
        self.cursor = 0

    def add(
        self,
        *,
        observation: Tensor,
        pre_tanh_action: Tensor,
        old_log_prob: Tensor,
        old_value: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        valid: Tensor,
    ) -> None:
        if self.cursor >= self.steps:
            raise RuntimeError("rollout buffer is full")
        expected = {
            "observation": (
                observation,
                (self.num_envs, self.observation_dim),
                torch.float32,
            ),
            "pre_tanh_action": (
                pre_tanh_action,
                (self.num_envs, self.action_dim),
                torch.float32,
            ),
            "old_log_prob": (old_log_prob, (self.num_envs,), torch.float32),
            "old_value": (old_value, (self.num_envs,), torch.float32),
            "reward": (reward, (self.num_envs,), torch.float32),
            "terminated": (terminated, (self.num_envs,), torch.bool),
            "truncated": (truncated, (self.num_envs,), torch.bool),
            "valid": (valid, (self.num_envs,), torch.bool),
        }
        for name, (value, shape, dtype) in expected.items():
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device != self.device
                or value.requires_grad
            ):
                raise ValueError(f"{name} has invalid rollout metadata")
            if dtype == torch.float32 and not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains non-finite values")
        if self.cursor < self.steps - 1 and truncated.any().item():
            raise ValueError(
                "first implementation permits truncation only at rollout end"
            )
        if torch.logical_and(terminated | truncated, ~valid).any().item():
            raise ValueError("an invalid rollout slot cannot terminate or truncate")
        index = self.cursor
        self.observations[index].copy_(observation)
        self.pre_tanh_actions[index].copy_(pre_tanh_action)
        self.old_log_prob[index].copy_(old_log_prob)
        self.old_value[index].copy_(old_value)
        self.rewards[index].copy_(reward)
        self.terminated[index].copy_(terminated)
        self.truncated[index].copy_(truncated)
        self.valid[index].copy_(valid)
        self.cursor += 1

    @torch.no_grad()
    def finish(
        self, *, last_value: Tensor, gamma: float, gae_lambda: float,
        allow_empty: bool = False,
    ) -> RolloutBatch:
        if self.cursor <= 0:
            raise RuntimeError("rollout must contain a transition before GAE")
        if (
            not isinstance(last_value, Tensor)
            or last_value.shape != (self.num_envs,)
            or last_value.dtype != torch.float32
            or last_value.device != self.device
            or not torch.isfinite(last_value).all().item()
        ):
            raise ValueError("last_value must be finite float32 [N]")
        used_steps = self.cursor
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for index in range(used_steps - 1, -1, -1):
            next_value = (
                last_value if index == used_steps - 1 else self.old_value[index + 1]
            )
            valid = self.valid[index].to(dtype=torch.float32)
            bootstrap = (~self.terminated[index]).to(dtype=torch.float32)
            episode_continues = (~(self.terminated[index] | self.truncated[index])).to(
                dtype=torch.float32
            )
            delta = valid * (
                self.rewards[index]
                + gamma * bootstrap * next_value
                - self.old_value[index]
            )
            gae = valid * (delta + gamma * gae_lambda * episode_continues * gae)
            advantages[index] = gae
        returns = advantages + self.old_value
        if not torch.logical_and(
            torch.isfinite(advantages[:used_steps]).all(),
            torch.isfinite(returns[:used_steps]).all(),
        ).item():
            raise RuntimeError("GAE produced non-finite values")
        observations = self.observations[:used_steps].reshape(-1, self.observation_dim)
        pre_tanh_actions = self.pre_tanh_actions[:used_steps].reshape(
            -1, self.action_dim
        )
        old_log_prob = self.old_log_prob[:used_steps].reshape(-1)
        old_value = self.old_value[:used_steps].reshape(-1)
        flat_returns = returns[:used_steps].reshape(-1)
        flat_advantages = advantages[:used_steps].reshape(-1)
        valid = self.valid[:used_steps].reshape(-1)
        if self.environment_major:
            order = torch.arange(used_steps * self.num_envs, device=self.device).reshape(
                used_steps, self.num_envs).transpose(0, 1).reshape(-1)
            observations, pre_tanh_actions, old_log_prob, old_value, flat_returns, flat_advantages, valid = (
                tensor.index_select(0, order) for tensor in
                (observations, pre_tanh_actions, old_log_prob, old_value, flat_returns, flat_advantages, valid)
            )
        if not valid.any().item() and not allow_empty:
            raise RuntimeError("rollout contains no valid PPO transitions")
        if valid.all().item():
            return RolloutBatch(
                observations=observations,
                pre_tanh_actions=pre_tanh_actions,
                old_log_prob=old_log_prob,
                old_value=old_value,
                returns=flat_returns,
                advantages=flat_advantages,
            )
        indices = valid.nonzero(as_tuple=False).squeeze(-1)
        return RolloutBatch(
            observations=observations.index_select(0, indices),
            pre_tanh_actions=pre_tanh_actions.index_select(0, indices),
            old_log_prob=old_log_prob.index_select(0, indices),
            old_value=old_value.index_select(0, indices),
            returns=flat_returns.index_select(0, indices),
            advantages=flat_advantages.index_select(0, indices),
        )


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    """Update diagnostics; ``approximate_kl`` is the post-step observed maximum."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    actor_learning_rate: float
    gradient_norm: float
    explained_variance: float
    minibatches: int
    early_stopped: bool


class PPOTrainer:
    """Experiment-85 learner promoted to the sole production PPO path."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        if seed < 0:
            raise ValueError("PPO seed must be non-negative")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ActorCritic(observation_dim, action_dim, config)
        # Preserve float64 normalization buffers while parameters remain float32.
        self.model = model.to(device=device)
        self.config = config
        self.device = device
        self._actor_parameters = (
            tuple(self.model.actor_body.parameters())
            + tuple(self.model.actor_mean.parameters())
            + (self.model.log_std,)
        )
        self._critic_parameters = tuple(self.model.critic_body.parameters()) + tuple(
            self.model.critic_value.parameters()
        )
        self.optimizer = torch.optim.Adam(
            (
                {"params": self._actor_parameters, "lr": config.actor_learning_rate},
                {"params": self._critic_parameters, "lr": config.critic_learning_rate},
            ),
            eps=1.0e-5,
        )
        self.sample_generator = torch.Generator(device=device)
        self.sample_generator.manual_seed(seed + 1)
        self.shuffle_generator = torch.Generator(device=device)
        self.shuffle_generator.manual_seed(seed + 2)
        self.execution = None  # Optional synchronous backend; never owned by the environment.

    @property
    def actor_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _normalized_observation(self, observation: Tensor) -> Tensor:
        if (
            observation.ndim != 2
            or observation.shape[0] <= 0
            or observation.shape[1] != self.model.observation_dim
            or observation.dtype != torch.float32
            or observation.device != self.device
            or not torch.isfinite(observation).all().item()
        ):
            raise ValueError("PPO observation metadata is invalid")
        return (
            (observation - self.model.observation_mean.float())
            / torch.sqrt(self.model.observation_variance.float() + 1.0e-5)
        ).clamp(-5.0, 5.0)

    def _normalized_value_target(self, value: Tensor) -> Tensor:
        return (
            (value - self.model.value_mean.float())
            / torch.sqrt(self.model.value_variance.float() + 1.0e-5)
        ).clamp(-5.0, 5.0)

    def _denormalized_value(self, normalized: Tensor) -> Tensor:
        return (
            normalized
            * torch.sqrt(self.model.value_variance.float() + 1.0e-5)
            + self.model.value_mean.float()
        )

    def _adapt_actor_learning_rate(self, observed_kl: float) -> float:
        current = self.actor_learning_rate
        if self.config.actor_lr_schedule == "constant":
            return current
        factor = self.config.actor_learning_rate_adaptation_factor
        if observed_kl > 2.0 * self.config.target_kl:
            updated = max(current / factor, self.config.actor_learning_rate_min)
        elif observed_kl < 0.5 * self.config.target_kl:
            updated = min(current * factor, self.config.actor_learning_rate_max)
        else:
            updated = current
        self.optimizer.param_groups[0]["lr"] = updated
        return float(updated)

    @torch.no_grad()
    def act(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.model.eval()
        normalized = self._normalized_observation(observation)
        mean, std = self.model.distribution_parameters(normalized)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=self.sample_generator,
        )
        raw_action = mean + std * noise
        action = raw_action.clamp(-1.0, 1.0)
        log_prob = _normal_log_prob(mean, std, raw_action)
        value = self._denormalized_value(self.model.value_normalized(normalized))
        return action, raw_action, log_prob, value

    @torch.no_grad()
    def value(self, observation: Tensor) -> Tensor:
        self.model.eval()
        normalized = self._normalized_observation(observation)
        return self._denormalized_value(self.model.value_normalized(normalized))

    @torch.no_grad()
    def deterministic_action(self, observation: Tensor) -> Tensor:
        self.model.eval()
        normalized = self._normalized_observation(observation)
        mean, _ = self.model.distribution_parameters(normalized)
        return mean.clamp(-1.0, 1.0)

    def _evaluate_actions(
        self,
        observation: Tensor,
        raw_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = self._normalized_observation(observation)
        mean, std = self.model.distribution_parameters(normalized)
        log_prob = _normal_log_prob(mean, std, raw_action)
        entropy = (0.5 + 0.5 * log(2.0 * torch.pi) + torch.log(std)).sum(dim=-1)
        value = self.model.value_normalized(normalized)
        return log_prob, entropy, value, mean

    def update(self, batch: RolloutBatch) -> PPOUpdateStats:
        execution = self.execution
        if batch.batch_size <= 0 and execution is None:
            raise ValueError("rollout batch must not be empty")
        tensors = (
            batch.observations,
            batch.pre_tanh_actions,
            batch.old_log_prob,
            batch.old_value,
            batch.returns,
            batch.advantages,
        )
        if any(
            value.device != self.device or value.dtype != torch.float32
            for value in tensors
        ):
            raise ValueError("rollout batch must be float32 on the trainer device")
        observer = getattr(self, "memory_observer", None)
        if observer is not None:
            observer("normalizer_begin", local_valid_n=batch.batch_size)
        if execution is None:
            self.model.update_running_statistics(
                batch.observations, torch.cat((batch.old_value, batch.returns)),
            )
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std(unbiased=False) + 1.0e-8
            )
        else:
            execution.merge(self.model, batch.observations, "observation", benchmark=True)
            execution.merge(self.model, torch.cat((batch.old_value, batch.returns)),
                            "value", benchmark=True)
            advantages = execution.advantages(batch.advantages, unbiased=False)
        if observer is not None:
            observer("normalizer_end", local_valid_n=batch.batch_size)
        original_batch = batch
        if execution is not None:
            batch, plan = execution.prepare(batch)
            if advantages.numel() == 0:
                advantages = batch.advantages
        if observer is not None:
            observer("batch_plan_ready", local_valid_n=batch.batch_size)
        size = batch.batch_size
        minibatch_count = min(self.config.num_minibatches, size)
        totals = torch.zeros(6, dtype=torch.float64)
        processed = 0
        minibatches = 0
        maximum_kl = 0.0
        self.model.train()
        for _ in range(self.config.update_epochs):
            if execution is None:
                permutation = torch.randperm(size, device=self.device,
                                             generator=self.shuffle_generator)
                batches = ((indices, indices.numel(), 1.0) for indices in
                           torch.tensor_split(permutation, minibatch_count))
            else:
                batches = ((part.indices, part.count, part.weight) for part in
                           plan.batches(generator=self.shuffle_generator,
                                        count=self.config.num_minibatches))
            for indices, count, weight in batches:
                observations = batch.observations.index_select(0, indices)
                raw_actions = batch.pre_tanh_actions.index_select(0, indices)
                new_log_prob, entropy, new_value, mean_action = self._evaluate_actions(
                    observations,
                    raw_actions,
                )
                old_log_prob = batch.old_log_prob.index_select(0, indices)
                ratio, _, _ = _policy_ratio_diagnostics(
                    new_log_prob,
                    old_log_prob,
                    clip_coef=self.config.clip_coef,
                )
                selected_advantage = advantages.index_select(0, indices)
                policy_loss = -torch.minimum(
                    ratio * selected_advantage,
                    ratio.clamp(
                        1.0 - self.config.clip_coef,
                        1.0 + self.config.clip_coef,
                    )
                    * selected_advantage,
                ).mean()
                selected_return = self._normalized_value_target(
                    batch.returns.index_select(0, indices)
                )
                old_value = self._normalized_value_target(
                    batch.old_value.index_select(0, indices)
                )
                if self.config.clip_value_loss:
                    clipped_value = old_value + (new_value - old_value).clamp(
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    value_loss = 0.5 * torch.maximum(
                        (new_value - selected_return).square(),
                        (clipped_value - selected_return).square(),
                    ).mean()
                else:
                    value_loss = 0.5 * (new_value - selected_return).square().mean()
                bounds_loss = (
                    (mean_action - 1.1).clamp_min(0.0).square()
                    + (-mean_action - 1.1).clamp_min(0.0).square()
                ).sum(dim=-1).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                    + 1.0e-4 * bounds_loss
                )
                if not torch.isfinite(loss).item():
                    raise RuntimeError("PPO objective is non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                if execution is None:
                    loss.backward()
                else:
                    execution.backward(loss, weight, self.model)
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self._actor_parameters,
                    self.config.max_grad_norm,
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    self._critic_parameters,
                    self.config.max_grad_norm,
                )
                gradient_norm = torch.maximum(actor_norm, critic_norm)
                if not torch.isfinite(gradient_norm).item():
                    raise RuntimeError("PPO gradient norm is non-finite")
                self.optimizer.step()
                with torch.no_grad():
                    post_log_prob, _, _, _ = self._evaluate_actions(
                        observations,
                        raw_actions,
                    )
                    _, post_kl, post_clip = _policy_ratio_diagnostics(
                        post_log_prob,
                        old_log_prob,
                        clip_coef=self.config.clip_coef,
                    )
                global_kl = (float(post_kl) if execution is None
                             else execution.weighted(post_kl, count))
                maximum_kl = max(maximum_kl, global_kl)
                totals += torch.tensor(
                    (
                        policy_loss.item(),
                        value_loss.item(),
                        entropy_mean.item(),
                        post_kl.item(),
                        post_clip.item(),
                        gradient_norm.item(),
                    ),
                    dtype=torch.float64,
                ) * count
                processed += count
                minibatches += 1
        if observer is not None:
            observer("ppo_epochs_end", minibatches=minibatches)
        self.model.eval()
        if processed <= 0 and execution is None:
            raise RuntimeError("PPO update processed no minibatches")
        variance = batch.returns.var(unbiased=False)
        explained_variance = (
            torch.tensor(float("nan"), device=self.device)
            if variance <= 1.0e-12
            else 1.0
            - (batch.returns - batch.old_value).var(unbiased=False) / variance
        )
        mean = totals / processed if execution is None else execution.totals(totals, processed)
        if execution is not None:
            explained_variance = torch.tensor(execution.explained_variance(
                original_batch.returns, original_batch.old_value))
        actor_learning_rate = self._adapt_actor_learning_rate(maximum_kl)
        return PPOUpdateStats(
            policy_loss=float(mean[0]),
            value_loss=float(mean[1]),
            entropy=float(mean[2]),
            approximate_kl=maximum_kl,
            clip_fraction=float(mean[4]),
            actor_learning_rate=actor_learning_rate,
            gradient_norm=float(mean[5]),
            explained_variance=float(explained_variance.item()),
            minibatches=minibatches,
            early_stopped=False,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sample_generator": self.sample_generator.get_state(),
            "shuffle_generator": self.shuffle_generator.get_state(),
            "implementation": TRAJECTORY_PPO_BENCHMARK_IMPLEMENTATION,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("implementation") != TRAJECTORY_PPO_BENCHMARK_IMPLEMENTATION:
            raise ValueError("PPO checkpoint implementation mismatch")
        self.model.load_state_dict(state["model"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]

        def cpu_generator_state(name: str) -> Tensor:
            value = state.get(name)
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.uint8
                or value.ndim != 1
                or value.numel() == 0
            ):
                raise ValueError(f"PPO checkpoint {name} is not a ByteTensor RNG state")
            return value.detach().cpu().clone()

        self.sample_generator.set_state(cpu_generator_state("sample_generator"))
        self.shuffle_generator.set_state(cpu_generator_state("shuffle_generator"))


def create_ppo_trainer(**kwargs) -> PPOTrainer:
    """Explicit learner registry; the default retains historical checkpoints."""
    return PPOTrainer(**kwargs)


__all__ = [
    "create_ppo_trainer",
    "ActorCritic",
    "PPOBenchmarkLineageSpec",
    "PPOConfig",
    "PPOTrainer",
    "PPOUpdateStats",
    "RolloutBatch",
    "RolloutBuffer",
    "TRAJECTORY_PPO_BENCHMARK_HIDDEN_DIMS",
    "TRAJECTORY_PPO_BENCHMARK_IMPLEMENTATION",
    "TRAJECTORY_PPO_BENCHMARK_INITIAL_STD",
    "TRAJECTORY_PPO_BENCHMARK_SOURCE_REVISION",
    "TRAJECTORY_PPO_IMPLEMENTATION_VERSION",
]
