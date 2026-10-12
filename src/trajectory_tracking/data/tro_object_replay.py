"""Checksummed numeric windows and multi-object assets; admission belongs to the adapter."""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .interprior import InterpriorSelectionEntry, InterpriorTrajectorySource, _git
from ..environment.scene_config import TRAJECTORY_ISAAC_SCENE, override_trajectory_scene_object

ASSET_ROOT_ENV = "TRAJTRACKING_OBJECT_ASSET_ROOT"


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("TRO artifact escapes its configured root")
    return path


@dataclass(frozen=True, slots=True)
class ReplayEntry(InterpriorSelectionEntry):
    replay_audit: dict






def cache_path(config, root, entry):
    path = child(config.data.selection_path(root).parent, entry.replay_audit["cache_relative"])
    if path.name != entry.episode_uid + ".npz":
        raise ValueError("TRO cache identity changed")
    return path




def object_scene(source, spec, asset_root):
    """Use real checksummed geometry; retain the benchmark robot/physics."""
    return override_trajectory_scene_object(source,
        object_urdf=child(asset_root, spec["urdf_relative"]), root_env=ASSET_ROOT_ENV,
        relative_path=spec["urdf_relative"], sha256_digest=spec["urdf_sha256"],
        object_uid=spec["object_uid"], object_id=spec["object_id"],
        missing_inertial_density_kg_m3=spec["density_kg_m3"],
        collider_type=spec["collider_type"], replace_cylinders_with_capsules=False,
        object_scale=tuple(spec["scale_xyz"]), object_size_m=tuple(spec["size_m"]))


def resolve_scene_source(config, scene_source, root, environ=None, *, payload=None):
    if payload is None:
        raise ValueError("an adapter-verified selection payload is required")
    env = os.environ if environ is None else environ
    asset_root = Path(env.get(ASSET_ROOT_ENV, ""))
    if not asset_root.is_absolute():
        raise ValueError(f"set {ASSET_ROOT_ENV} to the absolute asset root")
    names = sorted(payload["objects"])
    variants = tuple(object_scene(scene_source, payload["objects"][uid], asset_root) for uid in names)
    # The first test_count slots score each held-out row exactly once. Remaining
    # slots repeat this asset sequence; training samples same-object rows only.
    test = payload["episodes"][payload["split"]["train_count"]:]
    return replace(variants[0], object_variants=variants,
        object_layout=tuple(names.index(row["object_uid"]) for row in test))




def load_collection(config, *, project_root, device, environ=None, sources=None, episode_validator=None):
    import pyarrow as pa
    import torch
    import xml.etree.ElementTree as ET
    from .interprior import _ShardRuntime, _materialize_trajectory
    from .variable_length import VariableLengthReferenceBank

    root = Path(project_root).resolve()
    if sources is None or episode_validator is None:
        raise ValueError("adapter-verified sources and episode validator are required")
    sources = tuple(sources)
    rows, kinematics, names = [], None, None
    urdf = root / "submodules/Interprior" / TRAJECTORY_ISAAC_SCENE.robot.relative_path
    if digest(urdf) != TRAJECTORY_ISAAC_SCENE.robot.sha256:
        raise ValueError("TRO canonical robot changed")
    bounds = {joint.attrib["name"]: (float(joint.find("limit").attrib["lower"]),
        float(joint.find("limit").attrib["upper"])) for joint in ET.parse(urdf).getroot().findall("joint")
        if joint.find("limit") is not None}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 4))
        for source in sources:
            with np.load(cache_path(config, root, source.entry), allow_pickle=False) as cache:
                row, metadata = (json.loads(str(cache[key])) for key in ("row_json", "metadata_json"))
                arrays = {key: cache[key][None] for key in cache.files if key not in ("row_json", "metadata_json")}
            count = source.entry.replay_audit["frame_stop"]
            arrays["valid_frames"] = np.array([count])
            local = replace(config, data=replace(config.data, frame_stop=count))
            reference, kinematics = _materialize_trajectory(local, source, project_root=root,
                device=torch.device("cpu"), np=np,
                runtime=_ShardRuntime(pa.Table.from_pylist([row]), arrays, {0: metadata}),
                kinematics=kinematics, kinematics_joint_names=names,
                episode_validator=episode_validator)
            names = reference.joint_names
            limits = torch.tensor([bounds[name] for name in names])
            targets = reference.joint_position_target
            if ((targets < limits[:, 0] - 2e-4) | (targets > limits[:, 1] + 2e-4)).any().item():
                raise ValueError("TRO nominal target exceeds benchmark serialization tolerance")
            rows.append(replace(reference, joint_position_target=targets.clamp(limits[:, 0], limits[:, 1])))
        bank = VariableLengthReferenceBank(tuple(rows))
    finally:
        torch.set_num_threads(previous_threads)
    for name, value in vars(bank).items():
        if isinstance(value, torch.Tensor):
            setattr(bank, name, value.to(device))
    bank.object_uids = tuple(source.entry.replay_audit["object_uid"] for source in sources)
    return bank
