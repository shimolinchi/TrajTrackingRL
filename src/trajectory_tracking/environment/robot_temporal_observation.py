"""Optional actual-history / reference-lookahead joint-position observation.

One sample is one policy/control step, not a physics substep. The buffer is
owned by the environment; assembling an observation never advances history.
"""

import torch
from torch import Tensor

HISTORY_STEPS = (8, 4, 2, 1)
FUTURE_STEPS = (1, 2, 4, 8)
FEATURE_DIM = 27 * (len(HISTORY_STEPS) + len(FUTURE_STEPS))
ROLLOUT_SCHEMA_VERSION = 10


def feature_layout() -> tuple[tuple[str, int], ...]:
    return tuple(
        (f"actual_joint_position_t_minus_{step}_normalized", 27)
        for step in HISTORY_STEPS
    ) + tuple(
        (f"reference_joint_position_t_plus_{step}_normalized", 27)
        for step in FUTURE_STEPS
    )


def manifest() -> dict[str, object]:
    return {
        "implementation_version": "actual_history_reference_future_joint_position_v1",
        "history_offsets_control_steps": [-step for step in HISTORY_STEPS],
        "future_offsets_control_steps": list(FUTURE_STEPS),
        "history_source": "actual_simulated_joint_positions_not_reference_or_commands",
        "future_source": "same_episode_reference_joint_positions",
        "joint_order": "canonical_action_spec_arm7_hand20",
        "normalization": "runtime_joint_hard_bounds_midpoint_half_range_no_clipping",
        "reset_history": "repeat_actual_reset_state_until_history_available",
        "pre_roll": "record_actual_control_step_states",
        "terminal_policy": "clamp_future_to_per_trajectory_last_valid_frame",
        "feature_placement": "append_after_existing_observation",
    }


def normalize(joint_positions: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return ((joint_positions - (lower + upper) / 2) / ((upper - lower) / 2)).flatten(1)


class ActualJointHistory:
    """Lag-indexed [env, lag, canonical joint] storage, including current lag 0."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.values = torch.zeros(num_envs, max(HISTORY_STEPS) + 1, 27, device=device)

    def clear(self) -> None:
        self.values.zero_()

    def reset(self, env_ids: Tensor, actual_joint_position: Tensor) -> None:
        self.values[env_ids] = actual_joint_position[env_ids, None, :]

    def advance(self, actual_joint_position: Tensor, advanced_mask: Tensor) -> None:
        # Only simulator transitions belonging to this episode enter history.
        # Inactive endpoint slots stay frozen; deferred resets are filled later.
        rows = advanced_mask.nonzero(as_tuple=False).squeeze(-1)
        self.values[rows, 1:] = self.values[rows, :-1].clone()
        self.values[rows, 0] = actual_joint_position[rows]

    def features(self) -> Tensor:
        return self.values[:, HISTORY_STEPS]

    def state_dict(self) -> Tensor:
        return self.values.detach().clone()

    def validate_state(self, state: object) -> Tensor:
        if (
            not isinstance(state, Tensor)
            or state.shape != self.values.shape
            or state.dtype != self.values.dtype
            or not torch.isfinite(state).all().item()
        ):
            raise ValueError("invalid actual joint history rollout state")
        return state.detach().to(self.values.device).clone()

    def load_state_dict(self, state: object) -> None:
        self.values.copy_(self.validate_state(state))
