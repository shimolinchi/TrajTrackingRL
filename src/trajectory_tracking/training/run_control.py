"""Training-budget and deterministic-evaluation gate semantics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class NominalReplayGateSpec:
    """Acceptance thresholds for the no-offset recorded-action replay."""

    mean_error_threshold_m: float
    success_rate_threshold: float
    enforcement: Literal["blocking", "diagnostic"] = "blocking"

    def __post_init__(self) -> None:
        mean_threshold = float(self.mean_error_threshold_m)
        success_threshold = float(self.success_rate_threshold)
        if not isfinite(mean_threshold) or mean_threshold <= 0.0:
            raise ValueError("nominal replay mean-error threshold must be positive")
        if not isfinite(success_threshold) or not 0.0 <= success_threshold <= 1.0:
            raise ValueError("nominal replay success-rate threshold must be in [0,1]")
        if self.enforcement not in ("blocking", "diagnostic"):
            raise ValueError("nominal replay enforcement must be blocking or diagnostic")
        object.__setattr__(self, "mean_error_threshold_m", mean_threshold)
        object.__setattr__(self, "success_rate_threshold", success_threshold)

    def passes(
        self,
        *,
        mean_flow_error_m: float,
        episode_success_rate: float,
    ) -> bool:
        mean_error = float(mean_flow_error_m)
        success_rate = float(episode_success_rate)
        if not isfinite(mean_error) or mean_error < 0.0:
            raise ValueError(
                "nominal replay mean error must be finite and non-negative"
            )
        if not isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError("nominal replay success rate must be finite and in [0,1]")
        return (
            mean_error <= self.mean_error_threshold_m
            and success_rate >= self.success_rate_threshold
        )

    def to_manifest(self) -> dict[str, float | str]:
        return {
            "mean_error_threshold_m": self.mean_error_threshold_m,
            "success_rate_threshold": self.success_rate_threshold,
            "enforcement": self.enforcement,
        }


@dataclass(frozen=True, slots=True)
class TrainingControlSpec:
    """Absolute update budget plus a consecutive deterministic-success gate."""

    max_updates: int
    evaluation_interval: int
    target_success_rate: float
    success_patience: int
    success_early_stop_enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("max_updates", "evaluation_interval", "success_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        target = float(self.target_success_rate)
        if not isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_success_rate must be finite and in [0,1]")
        object.__setattr__(self, "target_success_rate", target)
        if type(self.success_early_stop_enabled) is not bool:
            raise ValueError("success_early_stop_enabled must be boolean")

    def should_evaluate(self, update: int) -> bool:
        if isinstance(update, bool) or not isinstance(update, int):
            raise TypeError("update must be an integer")
        if not 1 <= update <= self.max_updates:
            raise ValueError("update must be within the configured budget")
        return update % self.evaluation_interval == 0 or update == self.max_updates

    def gate_manifest(self) -> dict[str, object]:
        """Fields that must stay fixed when resuming; max_updates may increase."""

        payload = {
            "evaluation_interval": self.evaluation_interval,
            "target_success_rate": self.target_success_rate,
            "success_patience": self.success_patience,
        }
        # Preserve historical resume manifests when the default is unchanged.
        if not self.success_early_stop_enabled:
            payload["success_early_stop_enabled"] = False
        return payload

    def to_manifest(self) -> dict[str, object]:
        return {"max_updates": self.max_updates, **self.gate_manifest()}


@dataclass(slots=True)
class ConsecutiveSuccessGate:
    spec: TrainingControlSpec
    consecutive_successes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.consecutive_successes, bool)
            or not isinstance(self.consecutive_successes, int)
            or self.consecutive_successes < 0
            or self.consecutive_successes > self.spec.success_patience
        ):
            raise ValueError("consecutive_successes must not exceed success_patience")

    @property
    def satisfied(self) -> bool:
        return (self.spec.success_early_stop_enabled
                and self.consecutive_successes >= self.spec.success_patience)

    def observe(self, success_rate: float) -> bool:
        rate = float(success_rate)
        if not isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("success_rate must be finite and in [0,1]")
        if rate >= self.spec.target_success_rate:
            self.consecutive_successes = min(
                self.consecutive_successes + 1,
                self.spec.success_patience,
            )
        else:
            self.consecutive_successes = 0
        return self.satisfied


__all__ = [
    "ConsecutiveSuccessGate",
    "NominalReplayGateSpec",
    "TrainingControlSpec",
]
