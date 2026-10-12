"""Arm target integration at 60 Hz, with unchanged absolute hand commands."""
from dataclasses import replace

import torch
from torch import Tensor

from .actions import CanonicalActionSpec, DirectActionApplication, DirectActionProjector, DirectActionSpec

CONTROL_HZ = 60


def manifest(spec: DirectActionSpec) -> dict[str, object]:
    return {
        "implementation": "previous_target_arm_increment_hand_absolute_v1",
        "runtime_base_policy": False,
        "policy_output_semantic": "normalized_arm_target_increment_hand_absolute",
        "policy_output_bounds": [-1.0, 1.0],
        "arm": {
            "indices": list(range(7)),
            "mapping": "previous_executed_target + speed_rad_s/60 * clamp(raw,-1,1)",
            "speed_rad_s": spec.arm_target_speed_rad_s,
            "control_hz": CONTROL_HZ,
            "reset": "actual_reset_joint_position_projected_to_hard_limits",
            "state": "rollout.previous_joint_target; projected target prevents windup",
        },
        "hand": {"indices": list(range(7, 27)), "mapping": "affine_to_joint_limit_interior"},
        "executed_action_semantic": "absolute_joint_position_target",
        "executed_action_unit": "rad",
        "joint_limit_margin_rad": spec.joint_limit_margin_rad,
        "final_projection": "original_hard_joint_limits",
        "previous_policy_action_observation": "previous_executed_target_normalized_by_absolute_range",
        "target_difference_reward_coordinates": "executed_target_normalized_by_original_absolute_range",
        "likelihood": "original_raw_gaussian_sample",
        "nominal_action_composition": False,
    }


class ArmTargetIncrementProjector(DirectActionProjector):
    """The environment owns the previous target, reset masks and checkpoint state."""

    def __init__(self, action_spec: CanonicalActionSpec, direct_spec: DirectActionSpec):
        if direct_spec.mode != "arm_target_increment":
            raise ValueError("arm target increment requires its explicit action mode")
        if action_spec.groups[0].indices != tuple(range(7)):
            raise ValueError("mixed action requires canonical arm7 then hand20 order")
        super().__init__(action_spec, replace(direct_spec, mode="absolute", arm_target_speed_rad_s=None))
        self.direct_spec = direct_spec
        self.step_scale_rad = direct_spec.arm_target_speed_rad_s / CONTROL_HZ

    def _previous(self, previous: Tensor | None, action: Tensor) -> Tensor:
        if (not isinstance(previous, Tensor) or previous.shape != action.shape
                or previous.device != action.device):
            raise ValueError("previous target must match the action batch and device")
        self.action_spec.validate(previous, name="previous_joint_target")
        return previous

    @torch.no_grad()
    def apply(self, raw_action: Tensor, *, previous_joint_target: Tensor | None = None) -> DirectActionApplication:
        absolute = super().apply(raw_action)
        previous = self._previous(previous_joint_target, raw_action)
        requested = absolute.requested_action.clone()
        requested[:, :7] = previous[:, :7] + self.step_scale_rad * absolute.raw_action[:, :7]
        executed = absolute.executed_action.clone()
        arm = previous[:, :7] + self.step_scale_rad * absolute.bounded_normalized_action[:, :7]
        lower, upper = self._safe_bounds(raw_action)
        executed[:, :7] = arm.clamp(lower[:7], upper[:7])
        projected = absolute.projection_mask.clone()
        projected[:, :7] = requested[:, :7].ne(executed[:, :7])
        return replace(absolute, requested_action=requested, executed_action=executed,
                       projection_mask=projected)

    @torch.no_grad()
    def normalize(self, canonical_action: Tensor, *, previous_joint_target: Tensor | None = None) -> Tensor:
        """Reject recorded targets that cannot be reached in one bounded step."""
        result = super().normalize(canonical_action)
        previous = self._previous(previous_joint_target, canonical_action)
        arm = (canonical_action[:, :7] - previous[:, :7]) / self.step_scale_rad
        if (arm.abs() > 1 + 32 * torch.finfo(torch.float32).eps).any().item():
            raise ValueError("nominal arm target outside one-step increment range")
        result[:, :7] = arm.clamp(-1, 1)
        return result

    def normalize_executed_target(self, executed: Tensor) -> Tensor:
        return super().normalize(executed)
