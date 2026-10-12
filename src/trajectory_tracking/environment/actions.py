"""Canonical targets, reset deltas, and zero-base residual projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ActionGroupSpec:
    name: str
    joint_names: tuple[str, ...]
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.name not in ("arm", "hand"):
            raise ValueError("action group must be 'arm' or 'hand'")
        if not self.joint_names or len(self.joint_names) != len(self.indices):
            raise ValueError("action group joint names and indices must match")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("action group contains duplicate joints")
        if len(set(self.indices)) != len(self.indices) or min(self.indices) < 0:
            raise ValueError("action group indices must be unique and non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalActionSpec:
    """Explicit 27-D absolute joint-position target contract."""

    joint_names: tuple[str, ...]
    groups: tuple[ActionGroupSpec, ...]
    position_lower_rad: tuple[float, ...]
    position_upper_rad: tuple[float, ...]
    semantic: str = "absolute_joint_position_target"
    unit: str = "rad"

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        lower = tuple(float(value) for value in self.position_lower_rad)
        upper = tuple(float(value) for value in self.position_upper_rad)
        if len(names) != 27 or len(set(names)) != 27:
            raise ValueError("canonical action requires 27 unique joint names")
        if len(lower) != 27 or len(upper) != 27:
            raise ValueError("canonical action bounds must contain 27 values")
        if not all(torch.isfinite(torch.tensor(lower + upper)).tolist()):
            raise ValueError("canonical action bounds must be finite")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("every canonical joint must have lower < upper")
        groups = tuple(self.groups)
        if tuple(group.name for group in groups) != ("arm", "hand"):
            raise ValueError("canonical groups must be ordered arm, hand")
        if tuple(len(group.indices) for group in groups) != (7, 20):
            raise ValueError("canonical groups must contain 7 arm and 20 hand joints")
        covered = tuple(index for group in groups for index in group.indices)
        selected_names = tuple(name for group in groups for name in group.joint_names)
        if set(covered) != set(range(27)) or len(set(covered)) != 27:
            raise ValueError("action groups must cover canonical indices exactly once")
        if selected_names != tuple(names[index] for index in covered):
            raise ValueError("action group names and canonical indices disagree")
        if self.semantic != "absolute_joint_position_target" or self.unit != "rad":
            raise ValueError(
                "first task supports absolute joint-position targets in rad"
            )
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "position_lower_rad", lower)
        object.__setattr__(self, "position_upper_rad", upper)

    @property
    def action_dim(self) -> int:
        return 27

    def bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.tensor(
                self.position_lower_rad, dtype=reference.dtype, device=reference.device
            ),
            torch.tensor(
                self.position_upper_rad, dtype=reference.dtype, device=reference.device
            ),
        )

    def validate(self, value: Tensor, *, name: str, check_bounds: bool = True) -> int:
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 27:
            raise ValueError(f"{name} must have shape [N,27]")
        if value.shape[0] <= 0 or value.dtype != torch.float32:
            raise ValueError(f"{name} must be non-empty float32")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} contains non-finite values")
        if check_bounds:
            lower, upper = self.bounds(value)
            if torch.logical_or(value < lower, value > upper).any().item():
                raise ValueError(f"{name} violates hard joint limits")
        return value.shape[0]

    def to_manifest(self) -> dict[str, object]:
        return {
            "semantic": self.semantic,
            "unit": self.unit,
            "dtype": "float32",
            "shape": [27],
            "joint_names": list(self.joint_names),
            "position_lower_rad": list(self.position_lower_rad),
            "position_upper_rad": list(self.position_upper_rad),
            "groups": [
                {
                    "name": group.name,
                    "joint_names": list(group.joint_names),
                    "indices": list(group.indices),
                }
                for group in self.groups
            ],
        }


@dataclass(frozen=True, slots=True)
class JointTargetDeltaSpec:
    scale_rad: tuple[float, ...]
    joint_limit_margin_rad: float

    def __post_init__(self) -> None:
        scale = tuple(float(value) for value in self.scale_rad)
        if len(scale) != 27 or any(value <= 0.0 for value in scale):
            raise ValueError("joint-target delta scale must contain 27 positive values")
        margin = float(self.joint_limit_margin_rad)
        if (
            not torch.isfinite(torch.tensor(scale + (margin,))).all().item()
            or margin < 0
        ):
            raise ValueError(
                "delta scale/margin must be finite and margin non-negative"
            )
        object.__setattr__(self, "scale_rad", scale)
        object.__setattr__(self, "joint_limit_margin_rad", margin)

    @classmethod
    def from_group_scales(
        cls,
        action_spec: CanonicalActionSpec,
        *,
        arm_scale_rad: float,
        hand_scale_rad: float,
        joint_limit_margin_rad: float,
    ) -> JointTargetDeltaSpec:
        values = [0.0] * 27
        by_name = {"arm": arm_scale_rad, "hand": hand_scale_rad}
        for group in action_spec.groups:
            for index in group.indices:
                values[index] = float(by_name[group.name])
        return cls(tuple(values), joint_limit_margin_rad)

    def to_manifest(self) -> dict[str, object]:
        return {
            "semantic": "reset_absolute_joint_position_target_delta",
            "normalized_raw_bounds": [-1.0, 1.0],
            "scale_rad": list(self.scale_rad),
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "projection": "per_environment_direction_preserving_alpha",
        }


@dataclass(frozen=True, slots=True)
class DeltaActionComposition:
    reference_action: Tensor
    raw_delta: Tensor
    bounded_raw_delta: Tensor
    requested_delta_rad: Tensor
    projection_alpha: Tensor
    applied_delta_rad: Tensor
    executed_action: Tensor
    raw_clip_mask: Tensor
    projection_mask: Tensor


class JointTargetDeltaComposer:
    """Compose a bounded delta for the reset-time perturbation only."""

    def __init__(
        self,
        action_spec: CanonicalActionSpec,
        delta_spec: JointTargetDeltaSpec,
    ) -> None:
        margin = delta_spec.joint_limit_margin_rad
        if any(
            2.0 * margin >= upper - lower
            for lower, upper in zip(
                action_spec.position_lower_rad,
                action_spec.position_upper_rad,
                strict=True,
            )
        ):
            raise ValueError("joint-limit margin removes a joint's safe interior")
        self.action_spec = action_spec
        self.delta_spec = delta_spec

    @torch.no_grad()
    def compose(
        self, reference_action: Tensor, raw_delta: Tensor
    ) -> DeltaActionComposition:
        batch = self.action_spec.validate(
            reference_action, name="reference_action", check_bounds=True
        )
        if (
            not isinstance(raw_delta, Tensor)
            or raw_delta.shape != (batch, 27)
            or raw_delta.dtype != torch.float32
            or raw_delta.device != reference_action.device
            or not torch.isfinite(raw_delta).all().item()
        ):
            raise ValueError("raw_delta must be finite float32 [N,27] on one device")
        reference = reference_action.detach().clone()
        raw = raw_delta.detach().clone()
        bounded = raw.clamp(-1.0, 1.0)
        scale = torch.tensor(
            self.delta_spec.scale_rad, dtype=torch.float32, device=raw.device
        )
        requested = bounded * scale
        lower, upper = self.action_spec.bounds(reference)
        margin = self.delta_spec.joint_limit_margin_rad
        safe_lower = lower + margin
        safe_upper = upper - margin

        # Preserve an already valid recorded target exactly at alpha=0.  If it
        # sits in the margin zone, block outward motion while allowing inward motion.
        allowed_lower = torch.minimum(reference, safe_lower)
        allowed_upper = torch.maximum(reference, safe_upper)
        denominator = torch.where(
            requested.ne(0), requested, torch.ones_like(requested)
        )
        alpha_per_joint = torch.where(
            requested > 0,
            (allowed_upper - reference) / denominator,
            torch.where(
                requested < 0,
                (allowed_lower - reference) / denominator,
                torch.ones_like(requested),
            ),
        ).clamp(0.0, 1.0)
        alpha = alpha_per_joint.amin(dim=1, keepdim=True)
        executed = reference + alpha * requested
        tolerance = (
            32.0
            * torch.finfo(torch.float32).eps
            * torch.maximum(
                torch.maximum(lower.abs(), upper.abs()), torch.ones_like(lower)
            )
        )
        if (
            torch.logical_or(executed < lower - tolerance, executed > upper + tolerance)
            .any()
            .item()
        ):
            raise RuntimeError("safety projection produced a hard-limit violation")
        executed = torch.maximum(torch.minimum(executed, upper), lower)
        applied = executed - reference
        return DeltaActionComposition(
            reference_action=reference,
            raw_delta=raw,
            bounded_raw_delta=bounded,
            requested_delta_rad=requested,
            projection_alpha=alpha,
            applied_delta_rad=applied,
            executed_action=executed,
            raw_clip_mask=raw.ne(bounded),
            projection_mask=alpha.lt(1.0).expand_as(requested) & requested.ne(0),
        )


@dataclass(frozen=True, slots=True)
class DirectActionSpec:
    """Map a zero-base residual directly to an absolute joint target."""

    joint_limit_margin_rad: float
    mode: str = "absolute"
    arm_target_speed_rad_s: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("absolute", "arm_target_increment"):
            raise ValueError("unknown direct action mode")
        speed = self.arm_target_speed_rad_s
        if self.mode == "arm_target_increment":
            if (speed is None or isinstance(speed, bool)
                    or not torch.isfinite(torch.tensor(float(speed))).item() or speed <= 0):
                raise ValueError("arm target speed must be finite and positive")
            if self.joint_limit_margin_rad != 0:
                raise ValueError("arm target increment uses original hard joint limits")
            object.__setattr__(self, "arm_target_speed_rad_s", float(speed))
        elif speed is not None:
            raise ValueError("arm target speed requires arm_target_increment mode")
        margin = float(self.joint_limit_margin_rad)
        if not torch.isfinite(torch.tensor(margin)).item() or margin < 0.0:
            raise ValueError("direct-action margin must be finite and non-negative")
        object.__setattr__(self, "joint_limit_margin_rad", margin)

    def to_manifest(self) -> dict[str, object]:
        if self.mode == "arm_target_increment":
            from .incremental_arm_action import manifest
            return manifest(self)
        return {
            "residual_interpretation": "zero_base_full_action",
            "conceptual_base_action": {
                "value": 0.0,
                "coordinates": "normalized_policy_action",
            },
            "runtime_base_policy": False,
            "policy_output_semantic": "normalized_absolute_joint_position_target",
            "policy_output_bounds": [-1.0, 1.0],
            "executed_action_semantic": "absolute_joint_position_target",
            "executed_action_unit": "rad",
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "mapping": "affine_to_joint_limit_interior",
            "nominal_action_composition": False,
        }


@dataclass(frozen=True, slots=True)
class DirectActionApplication:
    raw_action: Tensor
    bounded_normalized_action: Tensor
    requested_action: Tensor
    executed_action: Tensor
    raw_clip_mask: Tensor
    projection_mask: Tensor


class DirectActionProjector:
    """Apply one zero-base residual without adding a trajectory action."""

    def __init__(
        self,
        action_spec: CanonicalActionSpec,
        direct_spec: DirectActionSpec,
    ) -> None:
        margin = direct_spec.joint_limit_margin_rad
        if any(
            2.0 * margin >= upper - lower
            for lower, upper in zip(
                action_spec.position_lower_rad,
                action_spec.position_upper_rad,
                strict=True,
            )
        ):
            raise ValueError("joint-limit margin removes a joint's action interval")
        self.action_spec = action_spec
        self.direct_spec = direct_spec

    def _safe_bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        lower, upper = self.action_spec.bounds(reference)
        margin = self.direct_spec.joint_limit_margin_rad
        return lower + margin, upper - margin

    @torch.no_grad()
    def normalize(self, canonical_action: Tensor) -> Tensor:
        """Invert the direct mapping, primarily for the nominal replay gate."""

        self.action_spec.validate(
            canonical_action, name="canonical_action", check_bounds=False
        )
        hard_lower, hard_upper = self.action_spec.bounds(canonical_action)
        tolerance = 32.0 * torch.finfo(torch.float32).eps
        if (
            torch.logical_or(
                canonical_action < hard_lower - tolerance,
                canonical_action > hard_upper + tolerance,
            )
            .any()
            .item()
        ):
            raise ValueError("canonical_action violates hard joint limits")
        # Recorded float32 targets can land one ULP outside a URDF limit after
        # serialization.  Match apply()'s endpoint correction without hiding a
        # physically meaningful out-of-range command.
        canonical_action = torch.maximum(
            torch.minimum(canonical_action, hard_upper), hard_lower
        )
        safe_lower, safe_upper = self._safe_bounds(canonical_action)
        if (
            torch.logical_or(
                canonical_action < safe_lower - tolerance,
                canonical_action > safe_upper + tolerance,
            )
            .any()
            .item()
        ):
            raise ValueError("canonical action lies outside the direct-action interval")
        center = 0.5 * (safe_lower + safe_upper)
        half_range = 0.5 * (safe_upper - safe_lower)
        return ((canonical_action - center) / half_range).clamp(-1.0, 1.0)

    @torch.no_grad()
    def apply(self, raw_action: Tensor) -> DirectActionApplication:
        if (
            not isinstance(raw_action, Tensor)
            or raw_action.ndim != 2
            or raw_action.shape[0] <= 0
            or raw_action.shape[1] != 27
            or raw_action.dtype != torch.float32
            or not torch.isfinite(raw_action).all().item()
        ):
            raise ValueError("raw_action must be finite float32 [N,27]")
        raw = raw_action.detach().clone()
        bounded = raw.clamp(-1.0, 1.0)
        safe_lower, safe_upper = self._safe_bounds(raw)
        center = 0.5 * (safe_lower + safe_upper)
        half_range = 0.5 * (safe_upper - safe_lower)
        requested = center + raw * half_range
        executed = center + bounded * half_range
        projected = requested.ne(executed)
        # For asymmetric float32 intervals, center + half_range can round one
        # ULP above safe_upper (and likewise below safe_lower).  The policy
        # coordinate is already bounded, so this is numerical endpoint
        # correction rather than an additional action projection.
        executed = torch.maximum(torch.minimum(executed, safe_upper), safe_lower)
        self.action_spec.validate(
            executed, name="executed_direct_action", check_bounds=True
        )
        return DirectActionApplication(
            raw_action=raw,
            bounded_normalized_action=bounded,
            requested_action=requested,
            executed_action=executed,
            raw_clip_mask=raw.ne(bounded),
            projection_mask=projected,
        )


def make_ordered_action_spec(
    *,
    joint_names: tuple[str, ...],
    position_lower_rad: tuple[float, ...],
    position_upper_rad: tuple[float, ...],
) -> CanonicalActionSpec:
    """Build the declared Wuji grouping from an already canonical joint order."""

    names = tuple(joint_names)
    return CanonicalActionSpec(
        joint_names=names,
        groups=(
            ActionGroupSpec("arm", names[:7], tuple(range(7))),
            ActionGroupSpec("hand", names[7:], tuple(range(7, 27))),
        ),
        position_lower_rad=position_lower_rad,
        position_upper_rad=position_upper_rad,
    )


__all__ = [
    "ActionGroupSpec",
    "CanonicalActionSpec",
    "DeltaActionComposition",
    "DirectActionApplication",
    "DirectActionProjector",
    "DirectActionSpec",
    "JointTargetDeltaComposer",
    "JointTargetDeltaSpec",
    "make_ordered_action_spec",
]
