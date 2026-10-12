"""Opt-in orchestration; the registered data adapter owns admission evidence."""

from ..data.registry import resolve_data_adapter


def producer_admission(config, project_root, *, enabled=False, synthetic=False):
    if not enabled:
        return None
    if synthetic:
        raise ValueError("source-label-admission requires a real source adapter")
    try:
        adapter = resolve_data_adapter(config.data.adapter)
    except ValueError as error:
        raise ValueError("source-label-admission requires a real source adapter") from error
    validate = getattr(adapter, "source_label_admission", None)
    if validate is None:
        raise ValueError("source-label-admission requires a real source adapter with producer evidence")
    return validate(config, project_root)
