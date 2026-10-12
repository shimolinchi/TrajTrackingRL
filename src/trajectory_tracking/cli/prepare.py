"""Resolve a raw trajectory directory into an immutable, fingerprinted local cache.

No training or GPU work is done unless --certify is explicitly selected.
"""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from trajectory_tracking.cli.preflight import ROOT, CONFIG, PACKAGE
from trajectory_tracking.data.preparation.publication import prepare, validate_input
from trajectory_tracking.data.preparation.cache import write_once
from trajectory_tracking.data.preparation.package import create_recipe, bind_alignment
from trajectory_tracking.data.tro_object_replay import digest, child


def input_fingerprint(dataset, *, data_range, seed, test_fraction, invalid, root=ROOT):
    import pyarrow.parquet as pq
    validate_input(dataset)
    files = ['dataset_metadata.json', 'object_catalog.parquet'] + [
        'success_continuous/' + n for n in ('commit.json', 'episodes.parquet', 'outcomes.parquet', 'flow_ranges.parquet')]
    names = pq.read_table(dataset / 'success_continuous/episodes.parquet', columns=['dataset_shard']).column(0).to_pylist()
    for name in sorted(set(names)):
        for file in ('commit.json', 'source_episode_table.jsonl'):
            files.append(f'success_continuous/shards/{name}/{file}')
    source = {name: digest(child(dataset, name)) for name in files}
    implementation = {name: digest(PACKAGE / name.removeprefix('src/trajectory_tracking/')) for name in (
        'src/trajectory_tracking/data/preparation/publication.py', 'src/trajectory_tracking/data/preparation/cache.py', 'src/trajectory_tracking/data/preparation/package.py',
        'src/trajectory_tracking/data/preparation/split.py', 'src/trajectory_tracking/data/tro_source_alignment.py',
        'src/trajectory_tracking/data/tro_success_continuous.py', 'src/trajectory_tracking/data/tro_published_continuous.py')}
    payload = dict(version=1, source=source, implementation=implementation,
        recipe=digest(CONFIG), data_range=data_range, seed=seed,
        test_fraction=test_fraction, invalid=invalid)
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def resolve_raw(dataset, asset_root, cache_root, *, data_range='producer_success', workers=4,
                seed=42, test_fraction=.2, invalid='error', catalog_asset_root=None,
                alignment_report=None, existing_contract=None, root=ROOT):
    dataset, asset_root, cache_root = dataset.resolve(), asset_root.resolve(), cache_root.resolve()
    if not cache_root.is_relative_to(root) or cache_root.parts[len(root.parts)] not in ('tmp', 'logs'):
        raise ValueError('cache must be within this checkout tmp/ or logs/')
    fingerprint, inputs = input_fingerprint(dataset, data_range=data_range, seed=seed,
        test_fraction=test_fraction, invalid=invalid, root=root)
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / fingerprint
    with (cache_root / (fingerprint + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError('another process is preparing this input; retry after it finishes') from e
        reused = (output / 'prepared.json').is_file()
        if not reused:
            if output.exists():
                output.rename(cache_root / (fingerprint + '.incomplete-' + uuid.uuid4().hex))
            staging = cache_root / ('.preparing-' + uuid.uuid4().hex)
            prepare(dataset, asset_root, staging, workers, seed=seed, test_fraction=test_fraction,
                    invalid=invalid, catalog_asset_root=catalog_asset_root, data_range=data_range)
            staging.rename(output)
            create_recipe(output, root=root)
            write_once(output / 'prepared.json', dict(input_fingerprint=fingerprint, inputs=inputs))
        elif json.loads((output / 'prepared.json').read_text()) != dict(input_fingerprint=fingerprint, inputs=inputs):
            raise ValueError('cached preparation identity changed')
        if alignment_report is not None:
            bind_alignment(output, alignment_report, root=root, existing_contract=existing_contract)
            write_once(output / 'receipt_location.json', dict(path=str(alignment_report.resolve())))
        report = output / 'alignment_report.json'
        locator = output / 'receipt_location.json'
        if locator.exists():
            report = Path(json.loads(locator.read_text())['path'])
        certified = (output / 'contract.json').is_file()
        return dict(cache=str(output), reused=reused, fingerprint=fingerprint,
            config=str(output / 'recipe.yaml'), contract=str(output / 'contract.json'),
            alignment_report=str(report), certified=certified,
            pending_reason=None if certified else '首次需要真实源状态恢复检查；在同一入口设置 mode="align" 后显式执行，再改回 train。')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'asset-root', 'cache-root', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--data-range', choices=('producer_success',), default='producer_success')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-fraction', type=float, default=.2)
    p.add_argument('--invalid', choices=('error', 'exclude'), default='error')
    p.add_argument('--catalog-asset-root', type=Path)
    p.add_argument('--alignment-report', type=Path)
    p.add_argument('--existing-contract', type=Path)
    p.add_argument('--certify', action='store_true')
    p.add_argument('--physical-gpu', type=int, default=0)
    p.add_argument('--minimum-free-gpu-mib', type=int, default=70000)
    a = p.parse_args()
    result = resolve_raw(a.dataset, a.asset_root, a.cache_root, data_range=a.data_range, workers=a.workers,
        seed=a.seed, test_fraction=a.test_fraction, invalid=a.invalid, catalog_asset_root=a.catalog_asset_root,
        alignment_report=a.alignment_report, existing_contract=a.existing_contract)
    if a.certify and not result['certified']:
        from .launch import exclusive_gpus
        binding = exclusive_gpus([a.physical_gpu])[0]
        # Only this explicit operation launches a small state-restore check. No policy actions or training.
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in text.splitlines()}
        if free.get(a.physical_gpu, -1) < a.minimum_free_gpu_mib:
            raise ValueError('selected GPU has insufficient free memory; no other process is changed')
        subprocess.run([sys.executable, '-m', 'trajectory_tracking.cli.certify',
            '--dataset', str(a.dataset), '--asset-root', str(a.asset_root), '--prepared', result['cache'],
            '--output', result['alignment_report'], '--device', 'cuda:0'], check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=binding['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                ISAACSIM_KIT_ARGS=f' --/renderer/activeGpu={a.physical_gpu} --/renderer/multiGpu/enabled=false'))
        bind_alignment(Path(result['cache']), Path(result['alignment_report']))
        result.update(certified=True, pending_reason=None)
    if a.certify:
        from trajectory_tracking.cli.preflight import verify
        verify(config_path=Path(result['config']), contract_path=Path(result['contract']),
               report_path=Path(result['alignment_report']),
               scene_asset_root=Path(os.environ['TRAJTRACKING_SCENE_ASSET_ROOT']))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['certified']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
