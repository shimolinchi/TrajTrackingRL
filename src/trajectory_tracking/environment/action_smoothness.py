"""Optional finite-difference costs on executed normalized joint targets.

No action filtering, policy changes, or reference-robot inputs. History belongs
to each environment, survives rollout boundaries, and is cleared on reset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor

from .actions import CanonicalActionSpec


@dataclass(frozen=True, slots=True)
class ActionSmoothnessSpec:
    enabled: bool = False
    first_difference_weight: float = 0.0
    second_difference_weight: float = 0.0
    arm_weight: float = 1.0
    hand_weight: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("action smoothness enabled must be bool")
        weights = (self.first_difference_weight, self.second_difference_weight,
                   self.arm_weight, self.hand_weight)
        if any(isinstance(x, bool) or not isfinite(x) or x < 0 for x in weights):
            raise ValueError("action smoothness weights must be finite and nonnegative")
        if not self.enabled and any(weights[:2]):
            raise ValueError("nonzero action smoothness weights require enabled=true")
        if any(weights[:2]) and not any(weights[2:]):
            raise ValueError("action smoothness needs at least one weighted action group")

    def to_manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "implementation": "executed_normalized_action_differences_v1",
            "action": "bounded_normalized_absolute_joint_target",
            "first_difference": "(a_t-a_previous)*(control_hz/60)",
            "second_difference": "(a_t-2*a_previous+a_older)*(control_hz/60)^2",
            "reduction": "mean_square_within_each_explicit_action_group",
            "composition": "public_tracking_reward-minus-weighted_cost; no clipping",
            "coefficient_unit": "public_reward_points_before_learner_reward_scale",
            "reset": "omit_first_difference_until_2_actions_and_second_until_3",
            "history": "valid_policy_steps_only; checkpointed; no_observation_change",
        }


@dataclass(frozen=True, slots=True)
class ActionSmoothnessDiagnostics:
    first_mse: Tensor  # [N,2], arm then hand; unavailable samples are zero.
    second_mse: Tensor
    joint_acceleration_mse: Tensor  # actual qdot difference / dt, rad^2/s^4.
    first_valid: Tensor
    second_valid: Tensor
    valid: Tensor
    penalty: Tensor  # public reward points, nonnegative.


class ActionSmoothnessReward:
    def __init__(self, spec: ActionSmoothnessSpec, *, action_spec: CanonicalActionSpec,
                 num_envs: int, control_hz: float, device: torch.device) -> None:
        if not spec.enabled or num_envs <= 0 or not isfinite(control_hz) or control_hz <= 0:
            raise ValueError("action smoothness requires an enabled spec and valid geometry")
        self.spec = spec
        self.control_hz = control_hz
        self.groups = tuple(torch.tensor(g.indices, device=device) for g in action_spec.groups)
        self.previous = torch.zeros(num_envs, 27, device=device, dtype=torch.float32)
        self.older = torch.zeros_like(self.previous)
        self.previous_joint_velocity = torch.zeros_like(self.previous)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.int64)

    def reset(self, env_ids: Tensor) -> None:
        self.previous[env_ids] = 0
        self.older[env_ids] = 0
        self.previous_joint_velocity[env_ids] = 0
        self.count[env_ids] = 0

    def clear(self) -> None:
        self.previous.zero_()
        self.older.zero_()
        self.previous_joint_velocity.zero_()
        self.count.zero_()

    @torch.no_grad()
    def evaluate(self, action: Tensor, joint_velocity: Tensor,
                 valid: Tensor) -> ActionSmoothnessDiagnostics:
        if (action.shape != self.previous.shape or action.dtype != torch.float32
                or action.device != self.previous.device
                or joint_velocity.shape != action.shape or joint_velocity.dtype != action.dtype
                or joint_velocity.device != action.device or valid.shape != self.count.shape
                or valid.dtype != torch.bool or valid.device != action.device):
            raise ValueError("action smoothness expects float32 [N,27] and bool [N]")
        first_valid = valid & (self.count >= 1)
        second_valid = valid & (self.count >= 2)
        rate = self.control_hz / 60.0

        def group_mse(delta: Tensor, mask: Tensor) -> Tensor:
            squared = torch.where(mask[:, None], delta, 0.0).square()
            return torch.stack([squared.index_select(1, ids).mean(1) for ids in self.groups], 1)

        first = group_mse((action - self.previous) * rate, first_valid)
        second = group_mse((action - 2 * self.previous + self.older) * rate**2, second_valid)
        acceleration = group_mse(
            (joint_velocity - self.previous_joint_velocity) * self.control_hz, first_valid)
        costs = self.spec.first_difference_weight * first + self.spec.second_difference_weight * second
        penalty = self.spec.arm_weight * costs[:, 0] + self.spec.hand_weight * costs[:, 1]
        self.older[valid] = self.previous[valid]
        self.previous[valid] = action[valid]
        self.previous_joint_velocity[valid] = joint_velocity[valid]
        self.count[valid] = (self.count[valid] + 1).clamp_max(2)
        return ActionSmoothnessDiagnostics(first, second, acceleration,
                                           first_valid, second_valid, valid.clone(), penalty)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).detach().clone() for name in
                ("previous", "older", "previous_joint_velocity", "count")}

    def load_state_dict(self, payload: object) -> None:
        expected = self.state_dict()
        if not isinstance(payload, dict) or payload.keys() != expected.keys():
            raise ValueError("invalid action smoothness state keys")
        validated = {}
        for name, template in expected.items():
            value = payload[name]
            if (not isinstance(value, Tensor) or value.dtype != template.dtype
                    or value.shape != template.shape or not torch.isfinite(value).all().item()):
                raise ValueError(f"invalid action smoothness state {name}")
            if name == "count":
                if ((value < 0) | (value > 2)).any().item():
                    raise ValueError("invalid action smoothness history count")
            elif name != "previous_joint_velocity" and (value.abs() > 1).any().item():
                raise ValueError("action smoothness history is outside normalized bounds")
            validated[name] = value.to(device=template.device).clone()
        for name, value in validated.items():
            getattr(self, name).copy_(value)


SMOOTHNESS_METRIC_NAMES = (
    "action_first_difference_arm_mse", "action_first_difference_hand_mse",
    "action_second_difference_arm_mse", "action_second_difference_hand_mse",
    "actual_arm_joint_acceleration_rms_rad_s2", "actual_hand_joint_acceleration_rms_rad_s2",
    "action_smoothness_penalty", "unpenalized_tracking_reward",
    "action_first_difference_samples", "action_second_difference_samples",
)


class ActionSmoothnessMetricAccumulator:
    """One device-local reduction; never count reset/warm-up zeros as smoothness."""

    def __init__(self) -> None:
        self.sums: Tensor | None = None

    def add(self, diagnostics: ActionSmoothnessDiagnostics, tracking_reward: Tensor) -> None:
        d = diagnostics
        vector = torch.cat((
            d.first_mse.double().sum(0), d.second_mse.double().sum(0),
            d.joint_acceleration_mse.double().sum(0),
            torch.stack((d.penalty.double().sum(),
                         torch.where(d.valid, tracking_reward, 0.0).double().sum(),
                         d.first_valid.double().sum(), d.second_valid.double().sum(),
                         d.valid.double().sum())),
        ))
        self.sums = vector if self.sums is None else self.sums + vector

    def result(self) -> dict[str, float | int | None] | None:
        if self.sums is None:
            return None
        values = self.sums.cpu().tolist()
        first_count, second_count, valid_count = values[8:11]
        denominators = (first_count, first_count, second_count, second_count,
                        first_count, first_count, valid_count, valid_count)
        result = {name: (value / n if n else None) for name, value, n in
                  zip(SMOOTHNESS_METRIC_NAMES[:8], values[:8], denominators, strict=True)}
        for name in SMOOTHNESS_METRIC_NAMES[4:6]:
            if result[name] is not None:
                result[name] = sqrt(result[name])
        result[SMOOTHNESS_METRIC_NAMES[8]] = int(first_count)
        result[SMOOTHNESS_METRIC_NAMES[9]] = int(second_count)
        return result
