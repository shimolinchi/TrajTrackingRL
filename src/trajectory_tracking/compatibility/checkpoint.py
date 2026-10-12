"""Read trusted checkpoints with historical task/config module names.

Runtime code imports only current modules. These aliases are applied while
unpickling an explicitly loaded checkpoint, never through sys.meta_path.
Serialized task/version strings stay frozen for existing compatibility hashes.
"""

import io
import pickle

import torch


def module_name(name):
    if name == "trajectory_residual" or name.startswith("trajectory_residual."):
        name = "trajectory_tracking" + name[len("trajectory_residual") :]
    if name == "trajectory_tracking.learning_configuration":
        return "trajectory_tracking.configuration"
    return name


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module_name(module), name)


Pickler = pickle.Pickler
dump = pickle.dump
dumps = pickle.dumps


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, **kwargs):
    return load(io.BytesIO(data), **kwargs)


def load_checkpoint(path, *, map_location="cpu"):
    """Trusted project archives only, with normal contract validation by caller."""
    import sys

    return torch.load(
        path,
        map_location=map_location,
        pickle_module=sys.modules[__name__],
        weights_only=False,
    )
