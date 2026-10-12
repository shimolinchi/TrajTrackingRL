"""Read-only CPU reference storage with exact per-step gathers on the execution device.

Only storage placement changes. Full rows, float32 values, endpoints and sampling
remain identical to a resident VariableLengthReferenceBank.
"""
import torch
from .variable_length import VariableLengthReferenceBank


class HostReferenceBank(VariableLengthReferenceBank):
    @classmethod
    def from_bank(cls, bank, *, device):
        if not isinstance(bank, VariableLengthReferenceBank) or bank.device.type != 'cpu':
            raise ValueError('host bank must be materialized on CPU')
        result = cls.__new__(cls)
        result.__dict__.update(bank.__dict__)
        result._execution_device = torch.device(device)
        result._host_valid_frame_counts = bank.valid_frame_counts
        result.valid_frame_counts = bank.valid_frame_counts.to(device)
        return result

    @property
    def device(self):
        return self._execution_device

    def _indices(self, trajectory_indices):
        if (trajectory_indices.ndim != 1 or trajectory_indices.dtype != torch.int64
                or trajectory_indices.device != self.device):
            raise ValueError('incompatible host-bank trajectory indices')
        rows = trajectory_indices.cpu()
        if ((rows < 0) | (rows >= self.trajectory_count)).any().item():
            raise ValueError('host-bank trajectory index out of range')
        return rows

    def gather_frames(self, name, trajectory_indices, frame_indices):
        if name not in self._FRAME_FIELDS:
            raise ValueError('unknown host-bank frame field')
        rows = self._indices(trajectory_indices)
        if (frame_indices.dtype != torch.int64 or frame_indices.device != self.device
                or frame_indices.shape[0] != len(rows)):
            raise ValueError('incompatible host-bank frame indices')
        frames = frame_indices.cpu()
        limits = self._host_valid_frame_counts[rows].reshape(len(rows), *((1,) * (frames.ndim-1)))
        if ((frames < 0) | (frames >= limits)).any().item():
            raise ValueError('host-bank gather would read padding or an invalid frame')
        rows = rows.reshape(len(rows), *((1,) * (frames.ndim-1))).expand_as(frames)
        return getattr(self, name)[rows, frames].to(self.device)

    def gather_trajectory_values(self, name, trajectory_indices):
        if name not in ('object_points_object_frame','robot_root_pose_env_local','table_root_pose_env_local'):
            raise ValueError('unknown host-bank trajectory field')
        return getattr(self, name).index_select(0,self._indices(trajectory_indices)).to(self.device)

    def frame_sequence(self, name, trajectory_index):
        """Expose a single valid browser sequence on the execution device."""
        if name not in self._FRAME_FIELDS:
            raise ValueError("unknown host-bank frame field")
        if not 0 <= trajectory_index < self.trajectory_count:
            raise ValueError("host-bank trajectory index out of range")
        count = int(self._host_valid_frame_counts[trajectory_index])
        return getattr(self, name)[trajectory_index, :count].to(self.device)


def load_reference_collection(config, *, project_root, device, sources, storage="execution"):
    """Choose placement without changing the adapter, admission or scientific config."""
    from .interprior import load_interprior_trajectory_collection
    if storage not in ("cpu", "execution"):
        raise ValueError("reference storage must be cpu or execution")
    bank = load_interprior_trajectory_collection(
        config, project_root=project_root,
        device=torch.device("cpu") if storage == "cpu" else device, sources=sources)
    return HostReferenceBank.from_bank(bank, device=device) if storage == "cpu" else bank
