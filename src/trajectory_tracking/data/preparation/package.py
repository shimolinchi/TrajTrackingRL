"""CPU package creation/binding; a real alignment receipt is a separate prerequisite."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import yaml

from ...workspace import CONFIG, CONTRACT, ROOT, alignment_source_path
from .alignment import verify_alignment
from trajectory_tracking.data.preparation.cache import write_once
from trajectory_tracking.data.tro_object_replay import digest
from trajectory_tracking.configuration import load_trajectory_direct_config


def physics_fingerprint(root=ROOT):
    template = json.loads((CONTRACT).read_text())
    files = {name: digest(alignment_source_path(root, name)) for name in template['physics_files']}
    symbols = []
    for item in template['physics_symbols']:
        text = alignment_source_path(root, item['path']).read_text()
        node = next(n for n in ast.parse(text).body if getattr(n, 'name', None) == item['symbol'])
        symbols.append(dict(item, sha256=sha256(ast.get_source_segment(text, node).encode()).hexdigest()))
    return dict(physics_file_sha256=files, physics_symbols=symbols)


def create_recipe(prepared, root=ROOT):
    prepared = prepared.resolve()
    selection = prepared / 'selection.json'
    payload = json.loads(selection.read_text())
    raw = yaml.safe_load((CONFIG).read_text())
    from trajectory_tracking.data.preparation.publication import selected_adapter
    adapter, _ = selected_adapter(payload["data_range"])
    raw['data'].update(adapter=adapter.ADAPTER_NAME, selection=selection.relative_to(root).as_posix(), selection_sha256=digest(selection),
        selection_count=payload['count'], selection_index=0,
        trajectory_count=payload['split']['train_count'], expected_episode_uid=payload['episodes'][0]['episode_uid'],
        frame_start=0, frame_stop=payload['frame_stop'])
    path = prepared / 'recipe.yaml'
    with path.open('x') as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    config = load_trajectory_direct_config(path, project_root=root)
    contract = dict(version=2, configuration_hash=config.compatibility_hash,
        recipe_sha256=digest(path), selection_sha256=digest(selection), count=payload['count'],
        train_count=payload['split']['train_count'], test_count=payload['split']['test_count'],
        object_count=len(payload['objects']), train_only_singleton_count=len(payload['split']['train_only_singleton_objects']),
        static_alignment_sha256=digest(prepared / 'static_alignment.json'),
        alignment_report_sha256=None, alignment_status='pending_real_scene_certification',
        producer_success_required=payload["producer_success_required"], data_range=payload["data_range"],
        action_replay_performed=False, **physics_fingerprint(root))
    write_once(prepared / 'contract.pending.json', contract)
    return contract


def bind_alignment(prepared, report, *, root=ROOT, existing_contract=None):
    """Pin, never invent, a receipt whose physics and static input identity match."""
    prepared, report = prepared.resolve(), report.resolve()
    contract = json.loads((prepared / 'contract.pending.json').read_text())
    receipt = json.loads(report.read_text())
    proof = receipt
    if existing_contract is not None:
        proof = json.loads(existing_contract.read_text())
        if proof['alignment_report_sha256'] != digest(report):
            raise ValueError('existing receipt SHA does not match its original contract')
    fingerprint = physics_fingerprint(root)
    if any(proof.get(k) != v for k, v in fingerprint.items()):
        raise ValueError('receipt physics provenance differs or is absent; recertify, do not rewrite hashes')
    if contract['recipe_sha256'] != digest(prepared / 'recipe.yaml'):
        raise ValueError('recipe changed after preparation')
    config = load_trajectory_direct_config(prepared / 'recipe.yaml', project_root=root)
    if config.compatibility_hash != contract['configuration_hash']:
        raise ValueError('prepared configuration changed')
    contract.update(alignment_report_sha256=digest(report), alignment_status='certified')
    verify_alignment(root, prepared, report, contract)
    write_once(prepared / 'contract.json', contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare')
    for name in ('dataset', 'asset-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--catalog-asset-root', type=Path)
    p.add_argument('--data-range', choices=('producer_success', 'success_continuous'), default='producer_success')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-fraction', type=float, default=.2)
    p.add_argument('--invalid', choices=('error', 'exclude'), default='error')
    p = sub.add_parser('bind')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--alignment-report', type=Path, required=True)
    p.add_argument('--existing-contract', type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        from trajectory_tracking.data.preparation.publication import prepare
        prepare(args.dataset, args.asset_root, args.output, args.workers, seed=args.seed,
                test_fraction=args.test_fraction, invalid=args.invalid, catalog_asset_root=args.catalog_asset_root, data_range=args.data_range)
        result = create_recipe(args.output)
    else:
        result = bind_alignment(args.prepared, args.alignment_report, existing_contract=args.existing_contract)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
