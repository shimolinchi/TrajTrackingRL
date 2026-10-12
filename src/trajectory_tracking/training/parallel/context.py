"""Small collective boundary. World size one never initializes distributed Torch."""

from datetime import timedelta
import ctypes
import os

import torch
import torch.distributed as dist


def _preloaded_nccl_version():
    """Torch may report its build version, not an LD_PRELOAD override.

    Query the globally loaded library without opening any new library. Without
    a globally visible symbol return unknown, rather than mislabel a build ID.
    """
    try:
        query = ctypes.CDLL(None).ncclGetVersion
    except AttributeError:
        return None
    query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    value = ctypes.c_int()
    if query(ctypes.byref(value)) != 0:
        raise RuntimeError("loaded NCCL version query failed")
    return [value.value // 10000, value.value // 100 % 100, value.value % 100]


class ParallelContext:
    def __init__(self, device: torch.device, *, rank: int = 0, world_size: int = 1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid parallel rank/world size")
        self.device, self.rank, self.world_size = device, rank, world_size
        self.tensor_group = None

    @classmethod
    def from_torchrun(cls, *, cpu: bool = False, timeout_seconds: int = 1800,
                      cuda_device_indices: tuple[int, ...] | None = None):
        size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if cuda_device_indices is not None:
            if (cpu or len(cuda_device_indices) != size
                    or len(set(cuda_device_indices)) != size
                    or any(type(i) is not int or i < 0 for i in cuda_device_indices)):
                raise ValueError("CUDA device indices must uniquely assign one device per worker")
        if not 0 <= local_rank < size:
            raise ValueError("local rank is outside the single-node worker range")
        index = local_rank if cuda_device_indices is None else cuda_device_indices[local_rank]
        device = torch.device("cpu" if cpu else f"cuda:{index}")
        if size > 1:
            dist.init_process_group(
                "gloo", rank=rank, world_size=size,
                timeout=timedelta(seconds=timeout_seconds),
            )
        return cls(device, rank=rank, world_size=size)

    def initialize_training_collectives(self, *, timeout_seconds=1800):
        """Separate simulator startup from GPU training communication.

        Startup metadata uses the CPU/Gloo default group. The separate tensor
        group is created only after every worker has initialized its simulator.
        """
        self.barrier()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            if self.world_size > 1:
                self.tensor_group = dist.new_group(
                    backend="nccl", timeout=timedelta(seconds=timeout_seconds))

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def device_manifest(self):
        binding = {"rank": self.rank, "torch_device": str(self.device),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            binding.update(name=properties.name, capability=[properties.major, properties.minor],
                           uuid=str(getattr(properties, "uuid", "unavailable")),
                           torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                           torch_reported_nccl_version=list(torch.cuda.nccl.version()),
                           preloaded_nccl_runtime_version=_preloaded_nccl_version(),
                           communication_environment={key: value for key, value in os.environ.items()
                                                      if key.startswith("NCCL_") or key == "LD_PRELOAD"})
        return binding

    def reduce(self, value: torch.Tensor, operation: str = "sum") -> torch.Tensor:
        result = value.detach().to(self.device).clone()
        if self.world_size > 1:
            dist.all_reduce(result, op={"sum": dist.ReduceOp.SUM,
                                       "max": dist.ReduceOp.MAX,
                                       "min": dist.ReduceOp.MIN}[operation], group=self.tensor_group)
        return result.to(value.device)

    def gather(self, value):
        """Small CPU metadata only; never transport rollout tensors here."""
        if self.world_size == 1:
            return [value]
        results = [None] * self.world_size
        dist.all_gather_object(results, value)
        return results

    def broadcast(self, value):
        """Startup/control metadata only; tensor rollouts remain device-local."""
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def gather_vector(self, value, counts):
        """Variable-length sample-ID metadata, never observation/action tensors."""
        if self.world_size == 1:
            return value
        padded = value.new_zeros(max(counts))
        padded[:value.numel()] = value
        received = [torch.empty_like(padded) for _ in counts]
        dist.all_gather(received, padded, group=self.tensor_group)
        return torch.cat([part[:count] for part, count in zip(received, counts, strict=True)])

    def agree(self, value, *, name: str):
        values = self.gather(value)
        if any(item != values[0] for item in values):
            raise ValueError(f"ranks disagree on {name}")

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def close(self):
        # Never barrier in exception cleanup: a failed worker may already be gone.
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def synchronize_model(self, model):
        if self.world_size > 1:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0, group=self.tensor_group)

    def synchronize_gradients(self, parameters):
        """One packed SUM before clipping; loss already has global sample weight.

        Explicit reduction supports both existing PPO forward APIs without a
        DDP wrapper changing checkpoint keys or inference collectives.
        """
        parameters = tuple(parameters)
        if self.world_size == 1:
            return
        gradients = [p.grad if p.grad is not None else torch.zeros_like(p)
                     for p in parameters]
        packed = torch.cat([g.reshape(-1) for g in gradients])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.tensor_group)
        offset = 0
        for parameter in parameters:
            end = offset + parameter.numel()
            gradient = packed[offset:end].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
            offset = end
