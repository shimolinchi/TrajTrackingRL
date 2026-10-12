"""Portable CPU preflight and launch plan for the current multi-object recipe.

No GPU context, action replay, preparation, split or shared coordinator is used by
preflight. The immutable prepared publication is verified in place.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAJTRACKING_WORKSPACE", Path.cwd())).resolve()
PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "resources/baseline.yaml"
CONTRACT = PACKAGE / "resources/physics_contract_template.json"


def gpu_plan(text: str, per_gpu_envs: int = 8192) -> dict:
    parts = text.split(",")
    if not parts or any(not p.strip().isdigit() for p in parts):
        raise ValueError("GPU list must contain distinct nonnegative physical ordinals")
    indices = tuple(int(p) for p in parts)
    if len(set(indices)) != len(indices) or per_gpu_envs <= 0:
        raise ValueError("GPU list must be unique and per-GPU environment count positive")
    return dict(gpu_indices=list(indices), world_size=len(indices),
                per_gpu_envs=per_gpu_envs, global_envs=len(indices) * per_gpu_envs)


def alignment_source_path(root: Path, relative: str) -> Path:
    """Resolve only the explicit package relocation; never change signed hashes."""
    prefix = "src/trajectory_residual/"
    if relative.startswith(prefix):
        relative = "src/trajectory_tracking/" + relative[len(prefix):]
    return PACKAGE / relative.removeprefix("src/trajectory_tracking/")


def verify_alignment(root: Path, prepared: Path, report_path: Path, contract: dict) -> dict:
    from trajectory_tracking.data.tro_object_replay import digest
    if digest(report_path) != contract["alignment_report_sha256"]:
        raise ValueError("one-time scene alignment report signature changed")
    report = json.loads(report_path.read_text())
    if (report.get("passed") is not True or report.get("producer_success_required") is not contract.get("producer_success_required", True)
            or report.get("action_replay_performed") is not False
            or report.get("source_actions_executed") != 0 or not report.get("restore_checks")):
        raise ValueError("missing genuine scene restoration evidence; replay was not certified")
    expected = contract["static_alignment_sha256"]
    if report["static_alignment_sha256"] != expected or digest(prepared / "static_alignment.json") != expected:
        raise ValueError("static physics/asset alignment changed")
    for relative, expected in contract["physics_file_sha256"].items():
        if digest(alignment_source_path(root, relative)) != expected:
            raise ValueError(f"physics bytes changed; recertify alignment: {relative}")
    for item in contract["physics_symbols"]:
        text = alignment_source_path(root, item["path"]).read_text()
        node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == item["symbol"])
        if sha256(ast.get_source_segment(text, node).encode()).hexdigest() != item["sha256"]:
            raise ValueError("physics backend changed; recertify alignment")
    return dict(report_sha256=contract["alignment_report_sha256"],
                static_sha256=contract["static_alignment_sha256"],
                physics_bytes_identical=True, action_replay_performed=False,
                restore_checks=report["restore_checks"])


def verify(*, root=ROOT, config_path=CONFIG, contract_path=None, report_path: Path, scene_asset_root: Path) -> dict:
    import numpy as np
    from trajectory_tracking.configuration import load_trajectory_direct_config
    from trajectory_tracking.cli.producer_admission import producer_admission
    from trajectory_tracking.data.registry import resolve_data_adapter
    from trajectory_tracking.data.heldout_split import heldout_test_config
    from trajectory_tracking.data.tro_object_replay import cache_path
    from trajectory_tracking.environment.scene_config import resolve_trajectory_scene_source
    root = Path(root).resolve()
    path = Path(contract_path) if contract_path is not None else CONTRACT
    contract = json.loads(path.read_text())
    if contract.get('version') == 2:
        if contract.get('alignment_status') != 'certified':
            raise ValueError('real scene alignment certification is pending')
        if sha256(Path(config_path).read_bytes()).hexdigest() != contract['recipe_sha256']:
            raise ValueError('recipe file checksum changed')
    config = load_trajectory_direct_config(config_path, project_root=root)
    if config.compatibility_hash != contract["configuration_hash"]:
        raise ValueError("current scientific recipe changed; an explicit new contract is required")
    if config.data.selection_sha256 != contract["selection_sha256"]:
        raise ValueError("current recipe requires its exact frozen selection")
    adapter = resolve_data_adapter(config.data.adapter)
    proof = producer_admission(config, root, enabled=True)
    payload = adapter.selection_payload(config, root)
    split = payload["split"]
    for name, actual in dict(count=len(payload["episodes"]), train_count=split["train_count"],
                            test_count=split["test_count"], object_count=len(payload["objects"]),
                            train_only_singleton_count=len(split["train_only_singleton_objects"])).items():
        if actual != contract[name]:
            raise ValueError(f"frozen producer count changed: {name}")
    alignment = verify_alignment(root, config.data.selection_path(root).parent, report_path, contract)
    test, _ = heldout_test_config(config, root)
    checked = 0
    for partition in (config, test):
        for source in adapter.resolve_sources(partition, root, None):
            # Cached metadata is signed by the immutable selection; all original
            # publication metadata and shard commits are also hashed above.
            with np.load(cache_path(partition, root, source.entry), allow_pickle=False) as cache:
                row = json.loads(str(cache["row_json"]))
                metadata = json.loads(str(cache["metadata_json"]))
                frames = {k: cache[k][None] for k in ("source_frame_index", "physics_step", "time_s")}
            adapter.validate_episode(partition, source, row, metadata, frames)
            checked += 1
    scene = resolve_trajectory_scene_source(project_root=root, scene_asset_root=scene_asset_root)
    aligned = adapter.resolve_scene_source(config, scene, root)
    return dict(verified=True, configuration_hash=config.compatibility_hash,
                selection_sha256=config.data.selection_sha256, producer_success_required=proof["producer_success_required"],
                checked_original_episode_records=checked, train_count=split["train_count"],
                test_count=split["test_count"], object_count=len(payload["objects"]),
                observation_dim=config.observation.feature_dim, nominal_gate=proof["nominal_gate"],
                alignment=alignment, object_assets_verified=len(aligned.object_variants))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--per-gpu-envs", type=int, default=8192)
    parser.add_argument("--plan-only", action="store_true", help="CPU arithmetic only; no data admission")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = gpu_plan(args.gpus, args.per_gpu_envs)
    if not args.plan_only:
        result.update(verify(config_path=args.config, contract_path=args.contract,
            report_path=Path(os.environ["TRAJTRACKING_ALIGNMENT_REPORT"]),
            scene_asset_root=Path(os.environ["TRAJTRACKING_SCENE_ASSET_ROOT"])))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
