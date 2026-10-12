"""Prepare a new producer-success continuous publication; never run GPU code."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from trajectory_tracking.data import tro_success_continuous, tro_published_continuous
from trajectory_tracking.data.tro_object_replay import digest
from trajectory_tracking.environment.scene_config import TRAJECTORY_ISAAC_SCENE
from trajectory_tracking.data.preparation.cache import ARRAYS, selected_metadata, save_window, write_once
from trajectory_tracking.workspace import ROOT, SCENE_ASSETS
from trajectory_tracking.data.preparation.split import grouped_partition


def selected_adapter(data_range):
    if data_range != "producer_success":
        raise ValueError("baseline preparation requires original producer-success records")
    from trajectory_tracking.data import tro_success_continuous
    return tro_success_continuous, True


def prepare_shard(dataset, output, name, rows, objects, data_range="producer_success"):
    adapter, candidate = selected_adapter(data_range)
    import pyarrow.parquet as pq
    import zarr

    shard = dataset / "success_continuous/shards" / name
    metadata_path = shard / "source_episode_table.jsonl"
    mappings, mapping_hash = selected_metadata(metadata_path, {r["slot"] for r in rows})
    points = {r["slot"]: r["canonical_points_obj"] for r in pq.read_table(
        shard / "episode_table.parquet", columns=["slot", "canonical_points_obj"]).to_pylist()}
    z = zarr.open_group(str(shard / "frames.zarr"), mode="r")
    # Read a bounded shard once, not the same compressed chunks for every row.
    data = {k: z[k] for k in ARRAYS}
    valid = np.asarray(z["valid_frames"][:])
    urdf = SCENE_ASSETS / TRAJECTORY_ISAAC_SCENE.robot.relative_path
    if digest(urdf) != TRAJECTORY_ISAAC_SCENE.robot.sha256:
        raise ValueError("canonical robot asset checksum mismatch")
    limits = {j.attrib["name"]: (float(j.find("limit").attrib["lower"]), float(j.find("limit").attrib["upper"]))
              for j in ET.parse(urdf).getroot().findall("joint") if j.find("limit") is not None}
    accepted, rejected = [], []
    shard_hash = digest(shard / "commit.json")
    for row in rows:
        try:
            if (not candidate(row)
                    or not candidate(row, mappings[row["slot"]]["metadata"])):
                raise ValueError("original_producer_not_success_completed")
            slot, n = row["slot"], row["frame_count"]
            if not isinstance(n, int) or n < 2:
                raise ValueError("invalid_frame_count")
            if int(valid[slot]) != n:
                raise ValueError("valid_length_disagrees_with_index")
            item = mappings[slot]
            meta = item["metadata"]
            arrays = {k: np.asarray(a[slot, :n]) for k, a in data.items()}
            adapter.validate_arrays(arrays, n)
            spec = objects[row["object_uid"]]
            for key, expected in (("robot", TRAJECTORY_ISAAC_SCENE.robot.sha256),
                    ("table", TRAJECTORY_ISAAC_SCENE.table.sha256), ("object", spec["urdf_sha256"])):
                if meta[key + "_urdf_sha256"] != expected:
                    raise ValueError("source_asset_hash_mismatch")
            names = meta["robot_joint_names_lab_order"]
            if len(names) != 27 or len(set(names)) != 27 or set(names) != set(limits):
                raise ValueError("canonical_joint_order_unrepresentable")
            bounds = np.array([limits[name] for name in names])
            targets = arrays["robot_joint_pos_target"]
            if np.any((targets < bounds[:, 0] - 2e-4) | (targets > bounds[:, 1] + 2e-4)):
                raise ValueError("joint_target_outside_serialization_tolerance")
            policy = meta["object_physics_import"]["runtime_policy"]
            if (policy["upstream_force_test_enabled"] or policy["upstream_mass_phase_switch_enabled"]
                    or not policy["constant_mass_during_episode"] or policy["disable_gravity"]):
                raise ValueError("unsupported_force_mass_gravity")
            pts = np.asarray(points[slot])
            if pts.shape != (1024, 3) or not np.isfinite(pts).all():
                raise ValueError("invalid_canonical_points")
            # Full retained publication, not frame120/follow300 cropping.
            case = dict(row={**row, "canonical_points_obj": points[slot]}, metadata=meta,
                        arrays=arrays, start=0, stop=n)
            poses = np.ascontiguousarray(arrays["object_root_current_pose_w"])
            track_hash = sha256(str((poses.shape, poses.dtype.str)).encode() + poses.tobytes()).hexdigest()
            audit = dict(episode_uid=row["episode_uid"], attempt_uid=row["attempt_uid"],
                object_uid=row["object_uid"], object_urdf_sha256=spec["urdf_sha256"],
                shard_relative=name, source_shard_commit_sha256=shard_hash,
                source_mapping_sha256=mapping_hash, frame_stop=n,
                original_collection_status=meta["collection_status"],
                original_episode_status=meta["episode_status"], published_partition="success_continuous",
                raw_source_identity=json.dumps(item["source_identity"], separators=(",", ":")),
                admission_protocol=adapter.PROTOCOL, action_replay_performed=False,
                object_track_sha256=track_hash)
            accepted.append(save_window(case, output, audit))
        except (ValueError, KeyError) as error:
            rejected.append(dict(episode_uid=row["episode_uid"], reason=str(error)))
    result = dict(shard=name, accepted=accepted, rejected=rejected)
    write_once(output / "shards" / (name + ".json"), result)
    return result


def validate_input(dataset):
    import pyarrow.parquet as pq
    required = ["dataset_metadata.json", "object_catalog.parquet", "success_continuous/commit.json",
                "success_continuous/episodes.parquet", "success_continuous/outcomes.parquet",
                "success_continuous/flow_ranges.parquet"]
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise ValueError("原始轨迹数据目录格式不支持：需要TRO V10 success_continuous；缺少 " + ", ".join(missing))
    meta = json.loads((dataset / "dataset_metadata.json").read_text())
    if meta.get("schema_version") != 10 or meta.get("source_environment") != "tro":
        raise ValueError("原始轨迹数据目录需为TRO schema_version=10；不支持裸CoACD网格或其他轨迹格式")
    commit = json.loads((dataset / "success_continuous/commit.json").read_text())
    for name in ("episodes", "outcomes", "flow_ranges"):
        if digest(dataset / "success_continuous" / (name + ".parquet")) != commit.get(name + "_sha256"):
            raise ValueError("原始轨迹索引checksum不匹配: " + name)
    columns = set(pq.read_schema(dataset / "success_continuous/episodes.parquet").names)
    expected = {"episode_uid", "attempt_uid", "object_uid", "dataset_shard", "slot", "frame_count",
                "episode_status", "collection_status", "training_eligible", "trajectory_written"}
    if not expected <= columns:
        raise ValueError("原始轨迹索引缺少字段: " + ", ".join(sorted(expected - columns)))
    commit = json.loads((dataset / "success_continuous/commit.json").read_text())
    if pq.ParquetFile(dataset / "success_continuous/episodes.parquet").metadata.num_rows != commit["episode_count"]:
        raise ValueError("原始轨迹索引记录数与commit不符")


def catalog_asset_path(asset_root, uid, spec, kind, catalog_asset_root=None):
    """Relocate a producer URI through the bundled catalog without trusting basenames."""
    from trajectory_tracking.data.tro_object_replay import child
    catalog_path = asset_root / "catalog.json"
    bundled = json.loads(catalog_path.read_text()).get(uid) if catalog_path.is_file() else None
    if bundled is not None:
        if any(bundled[key + "_sha256"] != spec[key + "_sha256"] for key in ("urdf", "mesh")):
            raise ValueError("bundled object catalog differs from source geometry")
        relative = Path(bundled[kind + "_relative"])
    else:
        original = Path(spec[kind + "_uri"])
        relative = original.relative_to(catalog_asset_root or asset_root) if original.is_absolute() else original
    path = child(asset_root, relative)
    if digest(path) != spec[kind + "_sha256"]:
        raise ValueError("source object geometry changed")
    return path


def prepare(dataset, asset_root, output, workers=4, *, seed=42, test_fraction=.2, invalid="error", catalog_asset_root=None, data_range="producer_success"):
    adapter, candidate = selected_adapter(data_range)
    import pyarrow.parquet as pq
    from trajectory_tracking.data.tro_source_alignment import align_source
    from trajectory_tracking.environment.scene_config import resolve_trajectory_scene_source

    dataset, asset_root, output = dataset.resolve(), asset_root.resolve(), output.resolve()
    if data_range == "success_continuous" and invalid != "error":
        raise ValueError("complete success_continuous requires every record; invalid=exclude is only for producer_success")
    validate_input(dataset)
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("output must be inside this checkout (ignored logs/ or tmp/)")
    if output.parts[len(ROOT.parts)] not in ("logs", "tmp"):
        raise ValueError("output must be in ignored logs/ or tmp/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty output directory; never overwrite a frozen package")
    if not 1 <= workers <= 8 or not 0 < test_fraction < 1 or invalid not in ("error", "exclude"):
        raise ValueError("workers1..8, fraction0..1 and invalid error/exclude required")
    output.mkdir(parents=True, exist_ok=True)
    (output / "windows").mkdir(exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    partition = dataset / "success_continuous"
    commit = json.loads((partition / "commit.json").read_text())
    for name in ("episodes", "outcomes", "flow_ranges"):
        if digest(partition / (name + ".parquet")) != commit[name + "_sha256"]:
            raise ValueError("publication index checksum mismatch")
    columns = [n for n in pq.read_schema(partition / "episodes.parquet").names if n != "canonical_points_obj"]
    if any("split" in n.lower() for n in columns):
        raise ValueError("explicit source split requires preserving its membership")
    rows = pq.read_table(partition / "episodes.parquet", columns=columns).to_pylist()
    import re
    if any(re.fullmatch(r"[0-9a-f]{64}", str(r["episode_uid"])) is None for r in rows):
        raise ValueError("episode_uid must be the original 64-character lowercase SHA256 identity")
    if len({r["episode_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate published identity; resolve before split")
    if any(type(r["slot"]) is not int or r["slot"] < 0 for r in rows):
        raise ValueError("invalid original shard slot")
    if len({(r["dataset_shard"], r["slot"]) for r in rows}) != len(rows):
        raise ValueError("duplicate original shard slot identity")
    candidate_count = len(rows)
    excluded_labels = [r for r in rows if not candidate(r)]
    rows = [r for r in rows if candidate(r)]
    if data_range == "success_continuous" and excluded_labels:
        raise ValueError("success_continuous contains structurally ineligible rows; no silent omission")
    if not rows:
        raise ValueError("no producer success/completed eligible continuous records")
    import re
    if any(re.fullmatch(r"[a-zA-Z0-9_-]+", str(r[k])) is None
           for r in rows for k in ("episode_uid", "dataset_shard")):
        raise ValueError("unsafe episode or shard identifier")
    catalog = {r["object_uid"]: r for r in pq.read_table(dataset / "object_catalog.parquet").to_pylist()}
    objects = {}
    for uid in sorted({r["object_uid"] for r in rows}):
        s = catalog[uid]
        paths = {}
        for kind in ("urdf", "mesh"):
            paths[kind] = catalog_asset_path(asset_root, uid, s, kind, catalog_asset_root)
        if s["collider_type"] != "convex_hull":
            raise ValueError("unsupported V10 collider_type")
        objects[uid] = dict(object_uid=uid, object_id=s["object_id"],
            urdf_relative=paths["urdf"].relative_to(asset_root).as_posix(),
            urdf_sha256=s["urdf_sha256"], mesh_sha256=s["mesh_sha256"],
            density_kg_m3=s["density_kg_m3"], collider_type=s["collider_type"],
            scale_xyz=s["scale_xyz"], size_m=(np.asarray(s["bbox_max_m"]) - s["bbox_min_m"]).tolist())
    _, alignment = align_source(resolve_trajectory_scene_source(project_root=ROOT,
        scene_asset_root=Path(__import__("os").environ["TRAJTRACKING_SCENE_ASSET_ROOT"])), dataset)
    write_once(output / "static_alignment.json", dict(alignment=alignment, objects=objects,
        admission_protocol=adapter.PROTOCOL, action_replay_performed=False,
        data_range=data_range, producer_success_required=data_range == "producer_success",
        waiver="User-selected source labels, action replay NOT_EXECUTED",
        source_split="publication has no train/test membership; new whole-source-episode split"))
    by_shard = defaultdict(list)
    for row in rows:
        by_shard[row["dataset_shard"]].append(row)
    results = []
    if workers == 1:
        results = [prepare_shard(dataset, output, name, members, objects, data_range)
                   for name, members in sorted(by_shard.items())]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(prepare_shard, dataset, output, name, members, objects, data_range)
                    for name, members in sorted(by_shard.items())]
            for future in as_completed(jobs):
                results.append(future.result())
                print(json.dumps(dict(shards_done=len(results), total_shards=len(jobs))), flush=True)
    accepted = sorted([a for r in results for a in r["accepted"]], key=lambda a: a["episode_uid"])
    rejected = [a for r in results for a in r["rejected"]]
    write_once(output / "technical_validation.json", dict(candidate_count=candidate_count, excluded_source_labels=len(excluded_labels), accepted_count=len(accepted), invalid_policy=invalid, rejected=rejected))
    if (data_range == "success_continuous" and rejected) or (invalid == "error" and any(r["reason"] != "original_producer_not_success_completed" for r in rejected)):
        raise ValueError("invalid eligible records: see technical_validation.json; no split emitted")
    identities = {r["episode_uid"]: ["attempt:" + r["attempt_uid"], "raw:" + r["raw_source_identity"],
                  "track:" + r["object_track_sha256"]] for r in accepted}
    by_object = defaultdict(list)
    for r in accepted:
        by_object[r["object_uid"]].append(r)
    train, test, group_count, singleton_objects = [], [], 0, []
    for uid, members in sorted(by_object.items()):
        if len(members) == 1:
            a, b, n = members, [], 1
            singleton_objects.append(uid)
        else:
            a, b, n = grouped_partition(members, identities, seed=seed, test_fraction=test_fraction)
        train.extend(a); test.extend(b); group_count += n
    train_ids = {i for r in train for i in identities[r["episode_uid"]]}
    test_ids = {i for r in test for i in identities[r["episode_uid"]]}
    if train_ids & test_ids:
        raise ValueError("cross-object source-episode leakage")
    if not train or not test:
        raise ValueError("need disjoint nonempty train/test partitions; singleton objects stay train-only")
    source_hashes = {"object_catalog.parquet": digest(dataset / "object_catalog.parquet")}
    for name in by_shard:
        rel = f"success_continuous/shards/{name}/source_episode_table.jsonl"
        source_hashes[rel] = digest(dataset / rel)
    payload = dict(schema_version=1, adapter_version=adapter.VERSION, admission_protocol=adapter.PROTOCOL,
        action_replay_performed=False, producer_success_required=data_range == "producer_success", data_range=data_range,
        published_partition="success_continuous", published_episode_count=commit["episode_count"], dataset_id=commit["dataset_id"],
        dataset_metadata_sha256=digest(dataset / "dataset_metadata.json"),
        source_hashes={**source_hashes, **{f"success_continuous/{name}": digest(partition / name)
            for name in ("commit.json", "episodes.parquet", "outcomes.parquet", "flow_ranges.parquet")}},
        objects={uid: objects[uid] for uid in by_object}, count=len(accepted),
        frame_start=0, frame_stop=max(r["frame_stop"] for r in accepted), episodes=accepted)
    write_once(output / "source_selection.json", payload)
    payload = dict(payload, episodes=train + test, split=dict(protocol="whole_episode_grouped_train_test_v1",
        seed=seed, requested_test_fraction=test_fraction, source_selection="source_selection.json",
        source_selection_sha256=digest(output / "source_selection.json"), train_count=len(train),
        test_count=len(test), independent_group_count=group_count, group_identities=identities,
        train_only_singleton_objects=singleton_objects))
    write_once(output / "selection.json", payload)
    write_once(output / "admission.json", dict(candidates=candidate_count, excluded_source_labels=len(excluded_labels), accepted=len(accepted),
        objects=len(by_object), train=len(train), test=len(test),
        exclusions=dict(Counter(r["reason"] for r in rejected)), rejected=rejected,
        source_collection_status=dict(Counter(r["collection_status"] for r in rows)),
        selection_sha256=digest(output / "selection.json"),
        admission_protocol=adapter.PROTOCOL, action_replay_performed=False))

    return output / "selection.json"
