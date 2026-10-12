"""One small source-state restore/read check; never execute source actions."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from trajectory_tracking.data.tro_object_replay import digest, object_scene
from trajectory_tracking.data.tro_source_alignment import align_source
from trajectory_tracking.environment.scene_config import resolve_trajectory_scene_source, resolved_trajectory_scene_manifest
from trajectory_tracking.cli.preflight import ROOT
from trajectory_tracking.data.preparation.cache import write_once as write_json
from trajectory_tracking.data.preparation.package import physics_fingerprint
from trajectory_tracking.configuration import load_trajectory_direct_config
from trajectory_tracking.data.registry import resolve_data_adapter
from trajectory_tracking.data.heldout_split import heldout_test_config


def verify(dataset, asset_root, prepared, output, device):
    if output.exists():
        raise FileExistsError("alignment output already exists; preserve the original receipt")
    config = load_trajectory_direct_config(prepared / 'recipe.yaml', project_root=ROOT)
    contract = json.loads((prepared / 'contract.pending.json').read_text())
    if digest(prepared / 'static_alignment.json') != contract['static_alignment_sha256']:
        raise ValueError("static alignment input changed")
    adapter = resolve_data_adapter(config.data.adapter)
    adapter.source_label_admission(config, ROOT)
    adapter.resolve_sources(config, ROOT, None)
    test, _ = heldout_test_config(config, ROOT)
    adapter.resolve_sources(test, ROOT, None)
    from trajectory_tracking.cli.distributed_train import _configure_rank_runtime_paths
    _configure_rank_runtime_paths(ROOT / 'tmp' / 'publication_alignment')
    from trajectory_tracking.environment.scene import launch_simulation_app
    candidates = [r for p in sorted((prepared / "shards").glob("*.json"))
                  for r in json.loads(p.read_text())["accepted"]]
    rows = []
    for r in candidates:
        if r["object_uid"] not in {v["object_uid"] for v in rows}:
            rows.append(r)
        if len(rows) == 2:
            break
    if not rows:
        raise ValueError("no validated objects for scene certification")
    specs = json.loads((prepared / "static_alignment.json").read_text())["objects"]
    source, alignment = align_source(resolve_trajectory_scene_source(project_root=ROOT,
        scene_asset_root=Path(__import__("os").environ["TRAJTRACKING_SCENE_ASSET_ROOT"])), dataset)
    adapter.resolve_scene_source(config, source, ROOT)  # Validate every admitted asset before opening Isaac.
    variants = tuple(object_scene(source, specs[r["object_uid"]], asset_root) for r in rows)
    source = replace(variants[0], object_variants=variants, object_layout=tuple(range(len(rows))))
    caches = []
    for row in rows:
        path = prepared / row["cache_relative"]
        if digest(path) != row["cache_sha256"]:
            raise ValueError("alignment numeric input changed")
        with np.load(path, allow_pickle=False) as data:
            caches.append({k: data[k] for k in data.files})
    metadata = [json.loads(str(c["metadata_json"])) for c in caches]
    names = tuple(metadata[0]["robot_joint_names_lab_order"])
    if any(tuple(m["robot_joint_names_lab_order"]) != names for m in metadata):
        raise ValueError("mixed canonical joint order")
    launcher = launch_simulation_app(headless=True, enable_cameras=False, device=device)
    code = 1
    try:
        from trajectory_tracking.environment.scene import create_simulation_context, create_scene, apply_scene_physics
        from trajectory_tracking.environment.core import IsaacTrajectoryBackend, action_spec_from_isaac_robot
        from trajectory_tracking.data.reference import ReferenceResetState
        sim = create_simulation_context(device=device, physics=source.spec.physics)
        scene = create_scene(source, num_envs=len(rows), env_spacing=1.5)
        sim.reset()
        apply_scene_physics(scene, source)
        action = action_spec_from_isaac_robot(scene["robot"], canonical_joint_names=names)
        backend = IsaacTrajectoryBackend(simulation_context=sim, scene=scene, robot=scene["robot"],
            table=scene["table"], object_asset=scene["object"], action_spec=action,
            physics_dt=1 / 120, render=False, trajectory=SimpleNamespace(joint_names=names,
                physics_hz=120, control_hz=60, device=torch.device(device),
                hand_kinematic_reference_available=False, num_object_points=1024))
        tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        root = tensor([m["robot_root_pose_w"] for m in metadata])
        table = tensor([m["table_root_pose_w"] for m in metadata])
        origin = tensor([m["env_origin_w"] for m in metadata])
        root[:, :3] -= origin
        table[:, :3] -= origin
        ids = torch.arange(len(rows), device=device)
        results = []
        # Two state restores exercise the start and a real later reference frame;
        # zero policy/nominal actions, no trajectory replay or success scoring.
        for frame in (0, min(10, min(r["frame_stop"] for r in rows) - 1)):
            read = lambda key: tensor([c[key][frame] for c in caches])
            state = ReferenceResetState(frame_index=torch.full_like(ids, frame),
                joint_position=read("robot_joint_pos"), joint_velocity=read("robot_joint_vel"),
                previous_joint_target=read("robot_joint_pos_target"),
                object_pose_robot_base=read("object_root_current_pose"),
                object_linear_velocity_robot_base=read("object_root_linear_vel"),
                object_angular_velocity_robot_base=read("object_root_angular_vel"),
                object_points_object_frame=tensor([json.loads(str(c["row_json"]))["canonical_points_obj"] for c in caches]),
                robot_root_pose_env_local=root, table_root_pose_env_local=table)
            actual = backend.restore(state, env_ids=ids)
            q_error = float((actual.joint_position - state.joint_position).abs().max())
            pose_error = float((actual.object_pose_robot_base[:, :3] - state.object_pose_robot_base[:, :3]).abs().max())
            if not np.isfinite([q_error, pose_error]).all() or q_error > 2e-5 or pose_error > 2e-4:
                raise ValueError(f"source-state restoration mismatch: q={q_error}, position={pose_error}")
            results.append(dict(frame=frame, joint_max_error_rad=q_error, object_position_max_error_m=pose_error))
        write_json(output, dict(passed=True, utc=datetime.now(timezone.utc).isoformat(),
            producer_success_required=contract["producer_success_required"], action_replay_performed=False,
            source_actions_executed=0, evaluated_success_rate=None, scene_count=1,
            environment_count=len(rows), restore_checks=results,
            **physics_fingerprint(ROOT), selection_sha256=digest(prepared / "selection.json"), canonical_joint_names=names,
            inputs=[{k: r[k] for k in ("episode_uid", "cache_relative", "cache_sha256")} for r in rows],
            static_alignment_sha256=digest(prepared / "static_alignment.json"),
            alignment=alignment, scene_manifest=resolved_trajectory_scene_manifest(source)))
        code = 0
    finally:
        from trajectory_tracking.cli.distributed_train import _close_simulator_process
        _close_simulator_process(launcher.app, code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "asset-root", "prepared", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    a = parser.parse_args()
    a.prepared = a.prepared.resolve()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT / "tmp") and not a.output.is_relative_to(a.prepared):
        parser.error("receipt must be within prepared package or repository tmp")
    verify(a.dataset, a.asset_root, a.prepared, a.output, a.device)
