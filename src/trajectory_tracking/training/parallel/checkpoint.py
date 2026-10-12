"""A canonical inference checkpoint plus atomic, hash-checked rank resume state."""

from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from trajectory_tracking.compatibility.checkpoint import load_checkpoint as load_archive

from ..artifacts import _atomic_json, save_ppo_checkpoint


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(directory, *, context, contract, runner, update, checkpoint_kwargs):
    directory = Path(directory)
    if context.primary:
        directory.mkdir(parents=True, exist_ok=False)
    context.barrier()
    rank_path = directory / f"rank_{context.rank:03d}.pt"
    state = {
        "environment": runner.env.rollout_state_dict(),
        "sample_generator": runner.trainer.sample_generator.get_state(),
        "shuffle_generator": runner.trainer.shuffle_generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state(context.device)
                     if context.device.type == "cuda" else None),
    }
    temporary = rank_path.with_suffix(".partial")
    torch.save(state, temporary)
    os.replace(temporary, rank_path)
    hashes = context.gather({"file": rank_path.name, "sha256": file_hash(rank_path)})
    if context.primary:
        policy = save_ppo_checkpoint(
            directory / "policy.pt", trainer=runner.trainer, update=update,
            environment_state=state["environment"], parallel_manifest=contract,
            **checkpoint_kwargs,
        )
        hashes.append({"file": policy.name, "sha256": file_hash(policy)})
        _atomic_json(directory / "complete.json", {
            "version": "synchronous_ppo_checkpoint_v1", "update": update,
            "contract": contract, "files": hashes,
        })
    context.barrier()
    return directory


def load_checkpoint(directory, *, context, contract, runner):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest.get("version") != "synchronous_ppo_checkpoint_v1" or manifest["contract"] != contract:
        raise ValueError("distributed resume contract mismatch (including world size/partition)")
    expected = {"policy.pt"} | {f"rank_{rank:03d}.pt" for rank in range(context.world_size)}
    rows = manifest["files"]
    if len(rows) != len(expected) or {row["file"] for row in rows} != expected:
        raise ValueError("incomplete distributed checkpoint file set")
    for row in rows:
        if file_hash(directory / row["file"]) != row["sha256"]:
            raise ValueError("distributed checkpoint checksum mismatch")
    policy = load_archive(directory / "policy.pt", map_location=context.device)
    if policy.get("parallel") != contract or policy["update"] != manifest["update"]:
        raise ValueError("checkpoint header differs from completion manifest")
    runner.trainer.load_state_dict(policy["trainer"])
    state = load_archive(directory / f"rank_{context.rank:03d}.pt",
                       map_location=context.device)
    runner.env.load_rollout_state_dict(state["environment"])
    runner.trainer.sample_generator.set_state(state["sample_generator"].cpu())
    runner.trainer.shuffle_generator.set_state(state["shuffle_generator"].cpu())
    torch.set_rng_state(state["torch_rng"].cpu())
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), context.device)
    context.barrier()
    return manifest["update"]
