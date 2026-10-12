"""Explicit source-label admission for complete TRO continuous publications.

This contract does not certify action replay. Both the continuous publication and original producer must mark success,
and the original collector must have completed. Relabeled failures are excluded.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import tro_object_replay as cache
from .interprior import InterpriorTrajectorySource
from .tro_source_alignment import align_source

ADAPTER_NAME = "tro_success_continuous"
VERSION = "tro_success_continuous_source_label_v1"
PROTOCOL = "producer_success_complete_continuous_no_replay_v1"


def successful_candidate(row, metadata=None):
    return (row.get("episode_status") == "success"
            and row.get("collection_status") == "completed"
            and (metadata is None or (metadata.get("episode_status") == "success"
                                     and metadata.get("collection_status") == "completed"))
            and row.get("training_eligible") is True
            and row.get("trajectory_written") is True)


def validate_arrays(arrays, count):
    if count < 2 or any(len(a) != count or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("invalid_length_or_nonfinite")
    if (not np.all(np.diff(arrays["physics_step"].astype(np.int64)) == 2)
            or not np.allclose(np.diff(arrays["time_s"]), 1 / 60, atol=1e-6, rtol=0)
            or np.any(np.diff(arrays["source_frame_index"]) <= 0)):
        raise ValueError("invalid_continuous_timing_or_source_identity")
    for key in ("ee_current_pose_w", "object_root_current_pose_w"):
        if np.max(np.abs(np.linalg.norm(arrays[key][:, 3:], axis=-1) - 1)) > 2e-3:
            raise ValueError("invalid_pose_quaternion")


def selection_payload(config, root):
    path = config.data.selection_path(root)
    if cache.digest(path) != config.data.selection_sha256:
        raise ValueError("continuous selection checksum changed")
    payload = json.loads(path.read_text())
    if (payload.get("adapter_version") != VERSION or payload.get("admission_protocol") != PROTOCOL
            or payload.get("action_replay_performed") is not False
            or payload.get("count") != config.data.selection_count
            or len(payload["episodes"]) != payload["count"]
            or config.data.frame_start != 0 or config.data.frame_stop != payload["frame_stop"]):
        raise ValueError("continuous source-label contract changed")
    split = payload["split"]
    if (config.data.selection_index, config.data.trajectory_count) not in (
            (0, split["train_count"]), (split["train_count"], split["test_count"])):
        raise ValueError("complete frozen partitions required")
    from .heldout_split import heldout_test_config
    train = replace(config, data=replace(config.data, selection_index=0,
        trajectory_count=split["train_count"], expected_episode_uid=payload["episodes"][0]["episode_uid"]))
    heldout_test_config(train, root)
    return payload


def entries(config, payload):
    rows = payload["episodes"][config.data.selection_index:config.data.selection_index + config.data.trajectory_count]
    if not rows or rows[0]["episode_uid"] != config.data.expected_episode_uid:
        raise ValueError("continuous partition identity changed")
    if any(r.get("original_collection_status") != "completed"
           or r.get("admission_protocol") != PROTOCOL
           or r.get("action_replay_performed") is not False for r in rows):
        raise ValueError("continuous selection contains a non-producer-success entry")
    return tuple(cache.ReplayEntry(r["episode_uid"], r["attempt_uid"], r["shard_relative"], 0, r) for r in rows)


def selection_entries(config, root):
    return entries(config, selection_payload(config, root))


def resolve_sources(config, root, environ):
    payload = selection_payload(config, root)
    upstream = root  # Source collector revision remains recorded provenance.
    dataset = config.data.dataset_root(environ)
    if cache.digest(dataset / "dataset_metadata.json") != payload["dataset_metadata_sha256"]:
        raise ValueError("continuous dataset metadata changed")
    for relative, expected in payload["source_hashes"].items():
        if cache.digest(cache.child(dataset, relative)) != expected:
            raise ValueError("continuous source publication changed")
    sources, hashes = [], {}
    for entry in entries(config, payload):
        shard = cache.child(dataset / "success_continuous/shards", entry.shard_relative)
        if shard not in hashes:
            hashes[shard] = cache.digest(shard / "commit.json")
        if hashes[shard] != entry.replay_audit["source_shard_commit_sha256"]:
            raise ValueError("continuous shard changed")
        if cache.digest(cache.cache_path(config, root, entry)) != entry.replay_audit["cache_sha256"]:
            raise ValueError("continuous numeric cache changed")
        sources.append(InterpriorTrajectorySource(entry, upstream, shard))
    return tuple(sources)


def resolve_scene_source(config, scene_source, root, environ=None):
    payload = selection_payload(config, root)
    aligned, evidence = align_source(scene_source, config.data.dataset_root(environ))
    if evidence["dataset_metadata_sha256"] != payload["dataset_metadata_sha256"]:
        raise ValueError("continuous source physics metadata changed")
    source = cache.resolve_scene_source(config, aligned, root, environ, payload=payload)
    # Train-only singleton assets still need physical slots, but no test identity.
    names = sorted(payload["objects"])
    missing = payload["split"].get("train_only_singleton_objects", [])
    return replace(source, object_layout=source.object_layout + tuple(names.index(uid) for uid in missing))


def validate_episode(config, source, row, metadata, frames):
    audit = source.entry.replay_audit
    if (not successful_candidate(row, metadata) or row["slot"] != 0
            or row["episode_uid"] != source.entry.episode_uid
            or row["attempt_uid"] != source.entry.attempt_uid
            or row["object_uid"] != audit["object_uid"]
            or row["collection_status"] != audit["original_collection_status"]
            or audit["admission_protocol"] != PROTOCOL):
        raise ValueError("continuous source/cache identity changed")
    for name, expected in (("robot", cache.TRAJECTORY_ISAAC_SCENE.robot.sha256),
            ("table", cache.TRAJECTORY_ISAAC_SCENE.table.sha256), ("object", audit["object_urdf_sha256"])):
        if metadata[name + "_urdf_sha256"] != expected:
            raise ValueError("continuous source asset changed")
    count = audit["frame_stop"]
    if (len(frames["source_frame_index"][0]) != count
            or np.any(np.diff(frames["source_frame_index"][0]) <= 0)
            or not np.all(np.diff(frames["physics_step"][0]) == 2)
            or not np.allclose(np.diff(frames["time_s"][0]), 1 / 60, atol=1e-6, rtol=0)):
        raise ValueError("continuous source timing changed")


def load_collection(config, *, project_root, device, environ=None, sources=None):
    root = Path(project_root).resolve()
    sources = resolve_sources(config, root, environ) if sources is None else sources
    return cache.load_collection(config, project_root=root, device=device, environ=environ,
        sources=sources, episode_validator=validate_episode)


def source_label_admission(config, project_root):
    """Validate the frozen producer envelope; source files are checked before GPU scene creation."""
    from hashlib import sha256
    import re
    path = config.data.selection_path(project_root)
    if path is None:
        raise ValueError("source-label-admission requires a frozen selection proof")
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != config.data.selection_sha256:
        raise ValueError("producer admission selection SHA-256 mismatch")
    proof = json.loads(raw)

    def require(condition, message):
        if not condition:
            raise ValueError("invalid producer admission proof: " + message)

    def is_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    require(isinstance(proof, dict), "expected an object")
    require(proof.get("adapter_version") == VERSION, "adapter version")
    require(proof.get("admission_protocol") == PROTOCOL, "admission protocol")
    require(proof.get("producer_success_required") is True, "producer success is not required")
    require(proof.get("action_replay_performed") is False, "replay status must be explicitly false")
    require(is_hash(proof.get("dataset_metadata_sha256")), "dataset metadata hash")
    hashes = proof.get("source_hashes")
    require(isinstance(hashes, dict) and bool(hashes)
            and all(isinstance(k, str) and k and is_hash(v) for k, v in hashes.items()), "source hashes")
    episodes = proof.get("episodes")
    require(isinstance(episodes, list) and bool(episodes), "episodes")
    require(type(proof.get("count")) is int
            and proof["count"] == len(episodes) == config.data.selection_count, "selection count")
    require(proof.get("frame_start") == 0 and type(proof.get("frame_stop")) is int
            and proof["frame_stop"] > 0, "complete trajectory frame bounds")
    split = proof.get("split")
    require(isinstance(split, dict) and split.get("protocol") == "whole_episode_grouped_train_test_v1",
            "frozen grouped split")
    require(all(type(split.get(k)) is int and split[k] > 0 for k in ("train_count", "test_count"))
            and split["train_count"] + split["test_count"] == len(episodes), "split counts")
    require(is_hash(split.get("source_selection_sha256")), "source selection hash")
    identities = set()
    for row in episodes:
        require(isinstance(row, dict), "episode audit")
        uid = row.get("episode_uid")
        require(isinstance(uid, str) and uid and uid not in identities, "unique episode identity")
        identities.add(uid)
        require(row.get("original_collection_status") == "completed", "original producer completion")
        require(row.get("admission_protocol") == PROTOCOL, "episode admission protocol")
        require(row.get("action_replay_performed") is False, "episode replay status")
        require(type(row.get("frame_stop")) is int and 0 < row["frame_stop"] <= proof["frame_stop"],
                "episode frame bound")
    return dict(
        admission_protocol=PROTOCOL, adapter=ADAPTER_NAME, adapter_version=VERSION,
        selection_path=str(path), selection_sha256=digest,
        dataset_metadata_sha256=proof["dataset_metadata_sha256"], source_hashes=hashes,
        producer_success_required=True, action_replay_performed=False, nominal_gate="NOT_EXECUTED",
        reason="Explicit producer-success source-label admission; nominal action replay waived, not passed",
        selection_count=len(episodes), train_count=split["train_count"], test_count=split["test_count"],
        source_selection_sha256=split["source_selection_sha256"],
        learned_evaluation="unchanged",
    )
