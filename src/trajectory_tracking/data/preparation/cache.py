"""Bounded publication provenance reads and canonical numeric cache serialization."""
from hashlib import sha256
import json
from pathlib import Path
import re
import numpy as np
from trajectory_tracking.data.interprior import _quat_apply_wxyz
from trajectory_tracking.data.tro_object_replay import digest
ARRAYS = ("robot_joint_pos", "robot_joint_vel", "robot_joint_pos_target", "phase",
          "physics_step", "time_s", "ee_current_pose_w", "object_root_current_pose_w",
          "object_root_linear_vel_w", "object_root_angular_vel_w", "source_frame_index")
METADATA_KEYS = ("env_origin_w", "robot_root_pose_w", "table_root_pose_w",
    "robot_joint_names_lab_order", "episode_status", "collection_status",
    "robot_urdf_sha256", "table_urdf_sha256", "object_urdf_sha256")

def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    with path.open("x") as stream:
        stream.write(text)

def selected_metadata(path, slots):
    """Stream provenance once; avoid decoding large unused episode records."""
    result, checksum = {}, sha256()
    with path.open("rb") as stream:
        for line in stream:
            checksum.update(line)
            match = re.match(rb'\s*\{\s*"derived_slot"\s*:\s*(\d+)', line[:100])
            if match and int(match[1]) not in slots:
                continue
            item = json.loads(line)
            if item["derived_slot"] not in slots:
                continue
            meta = item["raw_episode_row"]["metadata"]
            if item["derived_slot"] in result:
                raise ValueError("duplicate original metadata slot")
            result[item["derived_slot"]] = dict(
                source_identity=[item["source_run_uid"], item["source_shard"], item["source_slot"]],
                metadata={key: meta[key] for key in
                          (*METADATA_KEYS, "lifted_definition", "object_physics_import")})
    if set(result) != slots:
        raise ValueError("published collector metadata does not cover selected slots")
    return result, checksum.hexdigest()


def robot_base_pose(pose, root):
    inverse = np.asarray(root[3:], dtype=np.float32).copy()
    inverse /= np.linalg.norm(inverse)
    inverse[1:] *= -1
    position = _quat_apply_wxyz(inverse, pose[:, :3] - root[:3], np)
    q = pose[:, 3:] / np.linalg.norm(pose[:, 3:], axis=-1, keepdims=True)
    rotation = np.concatenate(
        [
            inverse[:1] * q[:, :1] - (inverse[1:] * q[:, 1:]).sum(-1, keepdims=True),
            inverse[:1] * q[:, 1:]
            + q[:, :1] * inverse[1:]
            + np.cross(inverse[1:], q[:, 1:]),
        ],
        axis=-1,
    )
    return np.concatenate([position, rotation], axis=-1).astype(np.float32)


def save_window(case, output, replay):
    from trajectory_tracking.data.interprior import _quat_apply_wxyz

    arrays, meta = case["arrays"], case["metadata"]
    start, stop = case["start"], case["stop"]
    saved = {key: arrays[key][start:stop] for key in ("time_s", "physics_step", "phase", "robot_joint_pos", "robot_joint_vel", "robot_joint_pos_target")}
    root = np.asarray(meta["robot_root_pose_w"], dtype=np.float32)
    inverse = root[3:].copy() / np.linalg.norm(root[3:])
    inverse[1:] *= -1
    for key in ("ee_current_pose", "object_root_current_pose"):
        saved[key] = robot_base_pose(arrays[key + "_w"][start:stop], root)
    for key in ("object_root_linear_vel", "object_root_angular_vel"):
        saved[key] = _quat_apply_wxyz(inverse, arrays[key + "_w"][start:stop], np)
    saved["source_frame_index"] = (arrays["source_frame_index"][start:stop]
        if "source_frame_index" in arrays else np.arange(start, stop))
    for key in ("source_time_s", "source_physics_step"):
        if key in arrays:
            saved[key] = arrays[key][start:stop]
    saved["row_json"] = json.dumps(dict(case["row"], slot=0))
    saved["metadata_json"] = json.dumps(meta)
    path = Path(output) / "windows" / (case["row"]["episode_uid"] + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(saved):
                raise ValueError(f"existing cache schema differs: {path}")
            for key, value in saved.items():
                matches = (json.loads(str(existing[key])) == json.loads(value)
                           if key.endswith("_json") else
                           existing[key].dtype == np.asarray(value).dtype
                           and np.array_equal(existing[key], value))
                if not matches:
                    raise ValueError(f"existing cache content differs: {path}: {key}")
    else:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **saved)
    return {**replay, "cache_relative": path.relative_to(output).as_posix(), "cache_sha256": digest(path)}
