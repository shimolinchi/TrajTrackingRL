"""The baseline uses complete, producer-success TRO trajectories."""
from importlib import import_module

DATA_ADAPTERS = {"tro_success_continuous": "trajectory_tracking.data.tro_success_continuous"}

def resolve_data_adapter(name: str):
    if name not in DATA_ADAPTERS:
        raise ValueError(f"unsupported baseline dataset adapter: {name}")
    return import_module(DATA_ADAPTERS[name])
