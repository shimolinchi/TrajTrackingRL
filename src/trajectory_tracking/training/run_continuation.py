"""Runtime-only continuation identity; scientific checkpoint validation stays strict."""

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path


def continuation_identity(checkpoint, contract):
    """Follow explicit checkpoint lineage, never infer an owner from latest paths.

    The current checkpoint's tensors are verified by the normal strict loader.
    Here only committed JSON metadata is read, before any tracking or video writes.
    """
    if checkpoint is None:
        return None
    current = Path(checkpoint).resolve(strict=True)
    first = current
    seen = set()
    parent_manifest = None
    history_step = 0
    for _ in range(128):
        if current in seen:
            raise ValueError("cyclic continuation lineage")
        seen.add(current)
        complete_path = current / "complete.json"
        complete = json.loads(complete_path.read_text())
        if complete.get("version") != "synchronous_ppo_checkpoint_v1" or complete.get("contract") != contract:
            raise ValueError("continuation checkpoint contract mismatch")
        if current.parent.name != "checkpoints":
            raise ValueError("continuation requires checkpoint within its original run")
        run_dir = current.parent.parent
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if manifest.get("parallel") != contract:
            raise ValueError("continuation run manifest differs from checkpoint")
        if parent_manifest is None:
            parent_manifest = manifest
            history = run_dir / "wandb_history.json"
            if history.exists():
                row = json.loads(history.read_text())
                history_step = row.get("next_history_step")
                if (row.get("run_id") != manifest.get("wandb", {}).get("run_id")
                        or type(history_step) is not int or history_step < 0):
                    raise ValueError("invalid continuation W&B history receipt")
            update = complete["update"]
            complete_hash = sha256(complete_path.read_bytes()).hexdigest()
        prior = manifest.get("continuation")
        if prior is None:
            return dict(parent_checkpoint=str(first), parent_update=update,
                        complete_sha256=complete_hash, video_owner_run_dir=str(run_dir),
                        wandb=parent_manifest.get("wandb", {}), next_history_step=history_step)
        current = Path(prior["parent_checkpoint"]).resolve(strict=True)
    raise ValueError("continuation lineage exceeds bounded depth")


def acquire_run_writer(run_dir):
    """One new-entrypoint writer per experiment, including continuation segments.

    Caller retains the fd for its entire process lifetime. Never unlink this lock.
    Legacy runs without this lock must also have stopped before explicit resume.
    """
    fd = os.open(Path(run_dir) / ".training_writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise RuntimeError("another writer owns this experiment; resume refused")
    return fd


def resume_tracking_identity(continuation, *, project, group):
    """An online continuation must use the same project, group and actual ID."""
    from urllib.parse import urlsplit
    prior = continuation["wandb"]
    run_id = prior.get("run_id")
    parts = urlsplit(prior.get("url") or "").path.strip("/").split("/")
    prior_project = prior.get("project")
    if prior_project is None and len(parts) == 4 and parts[2] == "runs" and parts[3] == run_id:
        prior_project = parts[1]  # Older manifests record only the verified run URL.
    if not run_id or prior_project != project or prior.get("group") != group:
        raise ValueError("online continuation requires the same recorded W&B identity/project/group")
    entity = prior.get("entity") or (parts[0] if len(parts) == 4 else None)
    if not entity:
        raise ValueError("online continuation requires recorded W&B entity")
    return dict(run_id=run_id, entity=entity)
