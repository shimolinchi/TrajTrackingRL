"""Opt-in actual-acceleration shaping; no simulator, action filter, or history.

Acceleration uses velocities bracketing ONE valid control step. Reset teleports
therefore cannot leak into a later step or require new checkpoint state.
"""

from dataclasses import asdict, dataclass
from math import ceil, isfinite
import re

import torch
from torch import Tensor

from .actions import CanonicalActionSpec


@dataclass(frozen=True, slots=True)
class JointAccelerationSpec:
    enabled: bool = False
    coefficient: float = 0.0
    arm_scale_rad_s2: float = 170.0
    hand_scale_rad_s2: float = 500.0
    calibration_required: bool = False
    calibration_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.calibration_required) is not bool:
            raise ValueError("acceleration flags must be boolean")
        for name in ("coefficient", "arm_scale_rad_s2", "hand_scale_rad_s2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"invalid acceleration {name}")
            if name != "coefficient" and value == 0:
                raise ValueError("acceleration scales must be positive")
        if not self.enabled and (self.coefficient or self.calibration_required or self.calibration_sha256):
            raise ValueError("disabled acceleration must not carry reward/calibration settings")
        if self.calibration_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.calibration_sha256):
            raise ValueError("invalid acceleration calibration SHA-256")

    def require_ready(self) -> None:
        if self.calibration_required and not self.calibration_sha256:
            raise ValueError("joint acceleration calibration is pending; prepare calibrated configs first")

    def to_manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "implementation": "actual_control_step_acceleration_huber_v1",
            "acceleration": "(qdot_after-qdot_before)*control_hz; rad/s^2",
            "huber_delta": 1.0,
            "group_reduction": "0.5*mean_arm7 + 0.5*mean_hand20",
            "composition": "subtract_public_reward_before_single_learner_scale",
            "mask": "valid_policy_transitions_only; no_reset_or_preroll",
        }


@dataclass(frozen=True, slots=True)
class JointAccelerationDiagnostics:
    arm: Tensor
    hand: Tensor
    loss: Tensor
    penalty: Tensor
    valid: Tensor

    def prefix(self, count: int) -> "JointAccelerationDiagnostics":
        return JointAccelerationDiagnostics(*(x[:count] for x in
            (self.arm, self.hand, self.loss, self.penalty, self.valid)))


class JointAccelerationReward:
    def __init__(self, spec: JointAccelerationSpec, *, action_spec: CanonicalActionSpec,
                 control_hz: float, device: torch.device) -> None:
        spec.require_ready()
        if not spec.enabled or not isfinite(control_hz) or control_hz <= 0:
            raise ValueError("acceleration requires enabled spec and positive control Hz")
        self.spec, self.control_hz = spec, control_hz
        self.groups = tuple(torch.tensor(g.indices, device=device) for g in action_spec.groups)
        if tuple(len(g) for g in self.groups) != (7, 20):
            raise ValueError("acceleration requires explicit arm7/hand20 action groups")

    @torch.no_grad()
    def evaluate(self, before: Tensor, after: Tensor, valid: Tensor) -> JointAccelerationDiagnostics:
        if (before.ndim != 2 or before.shape[1] != 27 or before.shape != after.shape
                or before.dtype != torch.float32 or after.dtype != before.dtype
                or before.device != after.device or valid.device != before.device
                or valid.shape != before.shape[:1] or valid.dtype != torch.bool):
            raise ValueError("acceleration requires float32 [N,27] velocities and bool [N] mask")
        acceleration = torch.where(valid[:, None], (after - before) * self.control_hz, 0.0)
        if not torch.isfinite(acceleration).all().item():
            raise ValueError("nonfinite acceleration on a valid transition")
        arm, hand = (acceleration.index_select(1, ids) for ids in self.groups)

        def huber_mean(value: Tensor, scale: float) -> Tensor:
            z = (value / scale).abs()
            return torch.where(z <= 1, 0.5 * z.square(), z - 0.5).mean(-1)

        loss = 0.5 * (huber_mean(arm, self.spec.arm_scale_rad_s2)
                      + huber_mean(hand, self.spec.hand_scale_rad_s2))
        return JointAccelerationDiagnostics(arm, hand, loss,
                                             self.spec.coefficient * loss, valid.clone())


class JointAccelerationAccumulator:
    """Transition-weighted metrics; exact P95 retained only for independent eval.

P95 is the nearest-rank absolute acceleration across valid steps and joints in
each group, NOT the percentile of per-episode RMS. Training keeps constant memory.
"""

    def __init__(self, *, percentiles: bool = False) -> None:
        self.sums: Tensor | None = None
        self.percentiles = percentiles
        self.samples: list[tuple[Tensor, Tensor]] = []

    def add(self, d: JointAccelerationDiagnostics, tracking_reward: Tensor) -> None:
        values = torch.stack((d.arm.double().square().mean(-1).sum(),
                              d.hand.double().square().mean(-1).sum(),
                              d.loss.double().sum(), d.penalty.double().sum(),
                              torch.where(d.valid, tracking_reward, 0).double().sum(),
                              d.valid.double().sum()))
        self.sums = values if self.sums is None else self.sums + values
        if self.percentiles:
            self.samples.append((d.arm[d.valid].abs().flatten().cpu(),
                                 d.hand[d.valid].abs().flatten().cpu()))

    def result(self) -> dict[str, float | int] | None:
        if self.sums is None:
            return None
        arm, hand, loss, penalty, tracking, count = self.sums.cpu().tolist()
        if count <= 0:
            return None
        result = {
            "valid_transition_count": int(count),
            "arm_rms_rad_s2": (arm / count) ** 0.5,
            "hand_rms_rad_s2": (hand / count) ** 0.5,
            "mean_normalized_huber": loss / count,
            "mean_penalty_public": penalty / count,
            "mean_unpenalized_tracking_reward": tracking / count,
        }
        if self.percentiles:
            for i, name in enumerate(("arm", "hand")):
                samples = torch.cat([pair[i] for pair in self.samples])
                result[f"{name}_abs_p95_rad_s2"] = float(
                    samples.kthvalue(ceil(0.95 * samples.numel())).values)
        return result
