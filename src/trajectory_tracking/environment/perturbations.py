"""Explicit reset-time random-action perturbation planning and receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .actions import (
    CanonicalActionSpec,
    DeltaActionComposition,
    JointTargetDeltaComposer,
    JointTargetDeltaSpec,
)
from ..data.reference import ReferenceEpisodeBatch


@dataclass(frozen=True, slots=True)
class RandomActionPerturbationSpec:
    mode: Literal["none", "random_action"]
    pre_roll_policy_steps: int
    distribution: Literal["uniform"]
    active_joint_groups: tuple[Literal["arm", "hand"], ...]
    arm_delta_scale_rad: float
    hand_delta_scale_rad: float
    joint_limit_margin_rad: float

    def __post_init__(self) -> None:
        if self.mode not in ("none", "random_action"):
            raise ValueError("perturbation mode must be none or random_action")
        if (
            isinstance(self.pre_roll_policy_steps, bool)
            or not isinstance(self.pre_roll_policy_steps, int)
            or self.pre_roll_policy_steps < 0
        ):
            raise ValueError("pre_roll_policy_steps must be a non-negative integer")
        if self.mode == "none" and self.pre_roll_policy_steps != 0:
            raise ValueError("mode=none requires pre_roll_policy_steps=0")
        if self.mode == "random_action" and self.pre_roll_policy_steps == 0:
            raise ValueError("mode=random_action requires a positive pre-roll")
        if self.distribution != "uniform":
            raise ValueError("first perturbation supports only uniform sampling")
        groups = tuple(self.active_joint_groups)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("active_joint_groups must be a unique non-empty subset")
        if any(group not in ("arm", "hand") for group in groups):
            raise ValueError("unknown perturbation joint group")
        values = (
            self.arm_delta_scale_rad,
            self.hand_delta_scale_rad,
            self.joint_limit_margin_rad,
        )
        if not torch.isfinite(torch.tensor(values)).all().item():
            raise ValueError("perturbation scales/margin must be finite")
        if self.arm_delta_scale_rad <= 0 or self.hand_delta_scale_rad <= 0:
            raise ValueError("perturbation scales must be positive")
        if self.joint_limit_margin_rad < 0:
            raise ValueError("perturbation joint-limit margin must be non-negative")
        object.__setattr__(self, "active_joint_groups", groups)

    def to_manifest(self) -> dict[str, object]:
        return {
            "type": ("none" if self.mode == "none" else "random_joint_action_delta"),
            "mode": self.mode,
            "pre_roll_policy_steps": self.pre_roll_policy_steps,
            "distribution": self.distribution,
            "active_joint_groups": list(self.active_joint_groups),
            "arm_delta_scale_rad": self.arm_delta_scale_rad,
            "hand_delta_scale_rad": self.hand_delta_scale_rad,
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "physics_steps_consumed": self.pre_roll_policy_steps,
            "reference_steps_consumed": self.pre_roll_policy_steps,
            "ppo_rollout_steps_consumed": 0,
            "episode_steps_consumed": 0,
            "recorded_action_used": self.mode == "random_action",
        }


@dataclass(frozen=True, slots=True)
class PerturbationPlan:
    start_indices: Tensor
    raw_delta: Tensor
    action_composition: DeltaActionComposition | None
    pre_roll_reference_cursors: Tensor
    executed_pre_roll_actions: Tensor
    final_reference_cursors: Tensor

    @property
    def pre_roll_policy_steps(self) -> int:
        return self.executed_pre_roll_actions.shape[1]


@dataclass(frozen=True, slots=True)
class PerturbationReceipt:
    plan: PerturbationPlan
    post_joint_position: Tensor
    post_object_pose_robot_base: Tensor
    physics_steps_consumed: int
    reference_steps_consumed: int
    ppo_rollout_steps_consumed: int = 0
    episode_steps_consumed: int = 0

    def __post_init__(self) -> None:
        count = self.plan.start_indices.numel()
        for name, value, shape in (
            ("post_joint_position", self.post_joint_position, (count, 27)),
            (
                "post_object_pose_robot_base",
                self.post_object_pose_robot_base,
                (count, 7),
            ),
        ):
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != shape
                or value.dtype != torch.float32
                or value.device != self.plan.start_indices.device
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"{name} must be finite float32 {shape}")
            object.__setattr__(self, name, value.detach().clone())
        expected = self.plan.pre_roll_policy_steps
        if self.physics_steps_consumed != expected:
            raise ValueError("perturbation receipt physics count disagrees with plan")
        if self.reference_steps_consumed != expected:
            raise ValueError("physics and reference pre-roll counts must match")
        if self.ppo_rollout_steps_consumed != 0 or self.episode_steps_consumed != 0:
            raise ValueError("reset perturbation must not consume PPO/episode steps")


class RandomActionPerturbation:
    """Sample one delta, then follow recorded actions for the remaining pre-roll."""

    def __init__(
        self,
        action_spec: CanonicalActionSpec,
        spec: RandomActionPerturbationSpec,
        *,
        generator: torch.Generator,
    ) -> None:
        delta_spec = JointTargetDeltaSpec.from_group_scales(
            action_spec,
            arm_scale_rad=spec.arm_delta_scale_rad,
            hand_scale_rad=spec.hand_delta_scale_rad,
            joint_limit_margin_rad=spec.joint_limit_margin_rad,
        )
        self.action_spec = action_spec
        self.spec = spec
        self.generator = generator
        self.composer = JointTargetDeltaComposer(action_spec, delta_spec)

    @torch.no_grad()
    def plan(self, episode: ReferenceEpisodeBatch) -> PerturbationPlan:
        if episode.trajectory.joint_names != self.action_spec.joint_names:
            raise ValueError("reference and perturbation canonical joint orders differ")
        if (
            episode.ppo_start_cursors.ne(
                episode.start_indices + self.spec.pre_roll_policy_steps
            )
            .any()
            .item()
        ):
            raise ValueError(
                "reference window pre-roll disagrees with perturbation spec"
            )
        count = episode.batch_size
        device = episode.trajectory.device
        steps = self.spec.pre_roll_policy_steps
        if steps == 0:
            raw = torch.zeros(count, 27, dtype=torch.float32, device=device)
            empty_cursors = torch.empty(count, 0, dtype=torch.int64, device=device)
            empty_actions = torch.empty(
                count, 0, 27, dtype=torch.float32, device=device
            )
            return PerturbationPlan(
                start_indices=episode.start_indices.detach().clone(),
                raw_delta=raw,
                action_composition=None,
                pre_roll_reference_cursors=empty_cursors,
                executed_pre_roll_actions=empty_actions,
                final_reference_cursors=episode.ppo_start_cursors.detach().clone(),
            )

        raw = torch.empty(count, 27, dtype=torch.float32, device=device).uniform_(
            -1.0, 1.0, generator=self.generator
        )
        active = set(self.spec.active_joint_groups)
        for group in self.action_spec.groups:
            if group.name not in active:
                raw[:, group.indices] = 0.0

        offsets = torch.arange(steps, dtype=torch.int64, device=device)
        cursors = episode.start_indices[:, None] + offsets[None, :]
        recorded = episode.recorded_actions(cursors).clone()
        first_reference = recorded[:, 0]
        composition = self.composer.compose(first_reference, raw)
        recorded[:, 0] = composition.executed_action
        for step in range(1, steps):
            self.action_spec.validate(
                recorded[:, step], name=f"recorded_pre_roll_action[{step}]"
            )
        return PerturbationPlan(
            start_indices=episode.start_indices.detach().clone(),
            raw_delta=raw.detach().clone(),
            action_composition=composition,
            pre_roll_reference_cursors=cursors.detach().clone(),
            executed_pre_roll_actions=recorded.detach().clone(),
            final_reference_cursors=episode.ppo_start_cursors.detach().clone(),
        )

    def receipt(
        self,
        plan: PerturbationPlan,
        *,
        post_joint_position: Tensor,
        post_object_pose_robot_base: Tensor,
    ) -> PerturbationReceipt:
        return PerturbationReceipt(
            plan=plan,
            post_joint_position=post_joint_position,
            post_object_pose_robot_base=post_object_pose_robot_base,
            physics_steps_consumed=plan.pre_roll_policy_steps,
            reference_steps_consumed=plan.pre_roll_policy_steps,
        )


__all__ = [
    "PerturbationPlan",
    "PerturbationReceipt",
    "RandomActionPerturbation",
    "RandomActionPerturbationSpec",
]
