"""Opt-in TRO source-replay contract, separate from benchmark admission.

This module only interprets published configuration/metadata. Simulator code
stays in environment/scene.py. No source success label is a physics result.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .tro_object_replay import digest

PROTOCOL = "tro_source_aligned_raw_replay_v1"


def align_source(source, dataset):
    """Validate every frozen config before selecting supported scene overrides."""
    metadata = json.loads((Path(dataset) / "dataset_metadata.json").read_text())
    if metadata.get("source_environment") != "tro" or metadata.get("schema_version") != 10:
        raise ValueError("source replay requires the TRO v10 publication")
    configs = metadata.get("environment_configs", [])
    if not configs:
        raise ValueError("missing resolved source environment configurations")
    physics = replace(source.spec.physics, enable_external_forces_every_iteration=False)
    for item in configs:
        cfg = item["config"]
        assets, sim = cfg["assets"], cfg["sim"]
        actual = {**sim, **sim["physx"]}
        names = ("dt", "solver_type", "enable_external_forces_every_iteration",
            "min_position_iteration_count", "max_position_iteration_count",
            "min_velocity_iteration_count", "max_velocity_iteration_count",
            "robot_solver_position_iterations", "robot_solver_velocity_iterations",
            "bounce_threshold_velocity", "friction_offset_threshold", "friction_correlation_distance",
            "gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count")
        if any(actual.get(name) != getattr(physics, name) for name in names):
            raise ValueError("unimplemented TRO source PhysX configuration")
        if (sim["gravity"] != list(physics.gravity) or cfg["decimation"] != 2
                or assets["robot_self_collision_enabled"] is not True
                or assets["modify_asset_frictions"] is not True
                or any(assets[key] != physics.default_friction for key in
                    ("object_friction", "robot_friction", "table_friction"))
                or assets["finger_tip_friction"] != physics.fingertip_friction
                or assets["object_missing_inertial_density_kg_m3"] != 200.
                or assets["object_collider_type"] != "convex_hull"
                or cfg["reset"]["table_surface_z_offset"] != .15
                or sim["physics_material"]["restitution"] != physics.restitution):
            raise ValueError("unimplemented TRO source asset/contact configuration")
    return replace(source, spec=replace(source.spec, physics=physics,
        robot_self_collision_enabled=True)), {
        "protocol": PROTOCOL,
        "dataset_metadata_sha256": digest(Path(dataset) / "dataset_metadata.json"),
        "environment_config_hashes": [item["sha256"] for item in configs],
        "env_spacing": configs[0]["config"]["scene"]["env_spacing"],
        "source_num_envs": configs[0]["config"]["scene"]["num_envs"],
        "action": "state[t], final target[t] -> state[t+1]; no delay/smoothing",
        "collision_filters": [["link7", f"right_finger{i}_link2"] for i in range(2, 6)],
        "limitations": ["contact warm-start state is not recorded",
            "replay scene population differs from source 8192",
            "historical source collision-filter USD is not part of the publication",
            "surface-relabel override algorithm is not included in the publication; report raw criterion separately"],
    }


def explicit_window(raw_row, count):
    """Continuous raw history plus 300 transitions after first EE-follow start.

    This is an RL reference window, not a BC action chunk crossing flow ends.
    A missing/short window must not prevent full-raw replay diagnostics.
    """
    flows = raw_row.get("flow_ranges", [])
    if not flows:
        raise ValueError("missing collector flow_ranges; phase inference is forbidden")
    for index, flow in enumerate(flows):
        if (flow["flow_index"] != index or flow["flow_kind"] not in ("lift", "ee_follow")
                or not 0 <= flow["robot_motion_start_frame"] <= flow["source_frame_start"]
                    < flow["source_frame_end"] <= count):
            raise ValueError("invalid explicit collector flow boundary")
    follow = [f for f in flows if f["flow_kind"] == "ee_follow" and f["accepted"]]
    if not follow:
        return None
    first = follow[0]
    start, tracking = first["robot_motion_start_frame"], first["execution_start_frame"]
    stop = tracking + 301
    return (start, tracking, stop) if start <= tracking and stop <= count else None


def raw_task_outcome(object_z, phase, threshold_z):
    """Original collector criterion: lift confirmed after LIFT, drop in phase 5.

    Surface-relabel success is deliberately NOT claimed to be this criterion.
    """
    above = np.asarray(object_z) >= threshold_z
    follow = np.asarray(phase) == 5
    confirmed = np.maximum.accumulate(above & follow)
    return {"lift_confirmed": bool(confirmed.any()),
        "dropped_in_ee_follow": bool((confirmed & follow & ~above).any()),
        "success": bool(confirmed.any() and not (confirmed & follow & ~above).any())}
