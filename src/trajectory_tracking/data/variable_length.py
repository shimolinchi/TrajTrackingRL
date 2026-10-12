"""Optional ragged reference storage and per-row reset bounds.

Padding is storage only: endpoints, flow clamping, resets and valid transitions
always use the original row length. Equal-length benchmark banks are unchanged.
"""

from copy import copy
from dataclasses import replace

import torch

from .reference import ReferenceStartSampler, ReferenceTrajectoryBank


class VariableLengthReferenceBank(ReferenceTrajectoryBank):
    def __init__(self, trajectories):
        lengths = [row.frame_count for row in trajectories]
        maximum = max(lengths)
        padded = []
        for row in trajectories:
            fields = {}
            for name in self._FRAME_FIELDS:
                value = getattr(row, name)
                fields[name] = torch.cat(
                    [
                        value,
                        value[-1:].expand(maximum - row.frame_count, *value.shape[1:]),
                    ]
                )
            padded.append(replace(row, **fields))
        super().__init__(tuple(padded))
        self.valid_frame_counts = torch.tensor(
            lengths, dtype=torch.int64, device=self.device
        )

    def terminal_indices(self, trajectory_indices):
        return self.valid_frame_counts[trajectory_indices] - 1

    def with_terminal_frame_counts(self, frame_counts, *, protocol, metadata):
        """Return a read-only evaluation view with earlier per-row endpoints."""
        values = torch.as_tensor(
            frame_counts, dtype=torch.int64, device=self.valid_frame_counts.device
        )
        if (values.shape != self.valid_frame_counts.shape
                or (values < 2).any().item()
                or (values > self.valid_frame_counts).any().item()):
            raise ValueError("evaluation frame counts must shorten existing rows")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("evaluation terminal protocol must be named")
        if not isinstance(metadata, dict):
            raise ValueError("evaluation terminal metadata must be a dictionary")
        result = copy(self)
        result.valid_frame_counts = values
        if hasattr(result, "_host_valid_frame_counts"):
            result._host_valid_frame_counts = values.cpu()
        result.heldout_terminal_protocol = protocol
        result.heldout_terminal_metadata = dict(metadata)
        return result

    def frame_sequence(self, name, trajectory_index):
        return super().frame_sequence(name, trajectory_index)[
            : int(self.valid_frame_counts[trajectory_index])
        ]

    def gather_frames(self, name, trajectory_indices, frame_indices):
        limits = self.valid_frame_counts[trajectory_indices].reshape(
            len(trajectory_indices), *((1,) * (frame_indices.ndim - 1))
        )
        if (frame_indices >= limits).any().item():
            raise ValueError("reference gather would read storage padding")
        return super().gather_frames(name, trajectory_indices, frame_indices)

    def make_start_sampler(self, **kwargs):
        return VariableLengthStartSampler(
            frame_counts=self.valid_frame_counts, **kwargs
        )


class VariableLengthStartSampler(ReferenceStartSampler):
    def __init__(self, *, frame_counts, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "uniform_valid":
            raise ValueError(
                "variable-length windows currently require uniform-valid starts"
            )
        self.row_full_maximum = frame_counts - 2 - kwargs["pre_roll_policy_steps"]
        self.row_maximum = torch.minimum(
            self.row_full_maximum,
            torch.floor(frame_counts * self.maximum_trajectory_fraction).to(
                torch.int64
            ),
        ).clamp_max(self.maximum)
        if (self.row_maximum < self.minimum).any().item():
            raise ValueError("a variable-length row has no legal reset start")

    def _row_maximum(self, trajectory_indices, count, *, full=False):
        if trajectory_indices is None or trajectory_indices.shape != (count,):
            raise ValueError(
                "variable-length starts require one trajectory index per row"
            )
        values = self.row_full_maximum if full else self.row_maximum
        if (trajectory_indices < 0).any().item() or (
            trajectory_indices >= len(values)
        ).any().item():
            raise ValueError("invalid variable-length trajectory index")
        return values[trajectory_indices]

    def sample(self, count, *, trajectory_indices=None):
        high = self._row_maximum(trajectory_indices, count)
        unit = torch.rand(count, device=self.device, generator=self.generator)
        return self.minimum + torch.floor(unit * (high - self.minimum + 1)).to(
            torch.int64
        )

    def validate_indices(
        self, start_indices, *, count, range_mode="configured", trajectory_indices=None
    ):
        values = super().validate_indices(
            start_indices, count=count, range_mode=range_mode
        )
        high = self._row_maximum(
            trajectory_indices, count, full=range_mode == "full_reference"
        )
        if (values > high).any().item():
            raise ValueError("explicit start exceeds its trajectory's valid range")
        return values

    def evenly_spaced(self, count):
        raise ValueError(
            "variable-length evaluation must specify trajectory-aligned starts"
        )
