"""Dataset-owned terminal boundaries for held-out evaluation only."""

from collections.abc import Mapping

from .registry import resolve_data_adapter


DEFAULT_PROTOCOL = "heldout_local0_reference_endpoint_only_v1"


def apply_heldout_terminal(config, sources, bank):
    """Return a bank view and audit data without changing the training bank."""
    if config.data.adapter == "interprior":
        return bank, DEFAULT_PROTOCOL, {}
    adapter = resolve_data_adapter(config.data.adapter)
    resolver = getattr(adapter, "heldout_terminal_spec", None)
    if resolver is None:
        return bank, DEFAULT_PROTOCOL, {}
    spec = resolver(config, sources)
    if (not isinstance(spec, Mapping)
            or not isinstance(spec.get("protocol"), str)
            or not isinstance(spec.get("metadata"), Mapping)):
        raise ValueError("invalid held-out terminal specification")
    view = bank.with_terminal_frame_counts(
        spec.get("frame_counts"), protocol=spec["protocol"],
        metadata=dict(spec["metadata"]),
    )
    return view, spec["protocol"], dict(spec["metadata"])


def terminal_audit(bank):
    """Read protocol metadata attached to an already prepared evaluation bank."""
    return (
        getattr(bank, "heldout_terminal_protocol", DEFAULT_PROTOCOL),
        dict(getattr(bank, "heldout_terminal_metadata", {})),
    )
