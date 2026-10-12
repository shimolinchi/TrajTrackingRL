"""Public checkpoint lineage, one video owner and strict online tracking contracts."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from trajectory_tracking.training.run_continuation import (
    continuation_identity, acquire_run_writer, resume_tracking_identity,
)
from trajectory_tracking.training.artifacts import TrajectoryWandbMirror, WandbTrackingSpec
from trajectory_tracking.visualization.evaluation_browser_recorder import EvaluationBrowserRecorder, rotate_owned, OWNER

CONTRACT = {'world_size': 8, 'config_hash': 'frozen'}
WB = dict(run_id='same', url='https://wandb.ai/entity/project/runs/same', group='group')


def checkpoint(root, parent=None):
    root.mkdir(parents=True)
    manifest = dict(parallel=CONTRACT, wandb=WB)
    if parent is not None:
        manifest['continuation'] = continuation_identity(parent, CONTRACT)
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    cp = root / 'checkpoints/update_000100'
    cp.mkdir(parents=True)
    (cp / 'complete.json').write_text(json.dumps(dict(
        version='synchronous_ppo_checkpoint_v1', update=100, contract=CONTRACT)))
    return cp


def test_multiple_continuations_share_original_owner_and_budget(tmp_path):
    first = checkpoint(tmp_path / 'first')
    second = checkpoint(tmp_path / 'second', first)
    third = checkpoint(tmp_path / 'third', second)
    owner = continuation_identity(third, CONTRACT)['video_owner_run_dir']
    assert Path(owner) == tmp_path / 'first'
    for i, cp in enumerate([first, second, third, third]):
        root = continuation_identity(cp, CONTRACT)['video_owner_run_dir']
        recorder = EvaluationBrowserRecorder(url='http://localhost', run_dir=root, commit='test', tmp_root=tmp_path)
        rotate_owned(recorder.directory, recorder.run_dir.name, reserve=1)
        stem = recorder.directory / f'eval_{i:03d}'
        stem.with_suffix('.mp4').write_bytes(b'synthetic CPU ownership fixture, not a real video')
        stem.with_suffix('.json').write_text(json.dumps(dict(owner=OWNER, run_id=recorder.run_dir.name)))
        assert len(list(recorder.directory.glob('*.mp4'))) <= 3
    assert sorted(p.stem for p in recorder.directory.glob('*.mp4')) == ['eval_001', 'eval_002', 'eval_003']
    fd = acquire_run_writer(owner)
    try:
        with pytest.raises(RuntimeError, match='another writer'):
            acquire_run_writer(owner)
    finally:
        os.close(fd)
    os.close(acquire_run_writer(owner))


def test_bad_contract_and_history_receipt_rejected(tmp_path):
    cp = checkpoint(tmp_path / 'first')
    with pytest.raises(ValueError, match='contract'):
        continuation_identity(cp, {})
    (cp.parent.parent / 'wandb_history.json').write_text(json.dumps(dict(run_id='other', next_history_step=200)))
    with pytest.raises(ValueError, match='receipt'):
        continuation_identity(cp, CONTRACT)


def test_tracking_identity_and_local_floor(tmp_path):
    cp = checkpoint(tmp_path / 'first')
    (cp.parent.parent / 'wandb_history.json').write_text(json.dumps(dict(run_id='same', next_history_step=209)))
    identity = continuation_identity(cp, CONTRACT)
    assert identity['next_history_step'] == 209
    assert resume_tracking_identity(identity, project='project', group='group') == dict(run_id='same', entity='entity')
    with pytest.raises(ValueError, match='identity'):
        resume_tracking_identity(identity, project='other', group='group')


class FakeWandb:
    def __init__(self, *, resumed=True, run_id='same', server_step=204, sdk_step=0, starting_step=0):
        self.logged, self.finished = [], []
        self.run = SimpleNamespace(id=run_id, url='https://wandb.ai/entity/project/runs/same',
            entity='entity', project='project', resumed=resumed, step=sdk_step,
            starting_step=starting_step, summary={},
            log=lambda values, step: self.logged.append((step, values)),
            finish=lambda exit_code: self.finished.append(exit_code))
        self.server_step = server_step
    def Settings(self, **kwargs): return kwargs
    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run
    def Api(self, *, timeout):
        assert timeout == 30
        return SimpleNamespace(run=lambda path: SimpleNamespace(lastHistoryStep=self.server_step))
    def define_metric(self, *args, **kwargs): pass


@pytest.mark.parametrize('sdk,starting,local,expected', [(0, 0, 0, 205), (0, 210, 0, 210), (0, 0, 215, 215), (220, 0, 0, 220)])
def test_real_sdk_zero_step_failure_cannot_recur(tmp_path, sdk, starting, local, expected):
    fake = FakeWandb(sdk_step=sdk, starting_step=starting)
    mirror = TrajectoryWandbMirror(run_dir=tmp_path, spec=WandbTrackingSpec(
        enabled=True, mode='online', project='project', resume_run_id='same', resume_next_history_step=local), wandb_module=fake)
    assert fake.kwargs['id'] == 'same' and fake.kwargs['resume'] == 'must'
    mirror.log_event(dict(event='update', update=201))
    assert fake.logged[0][0] == expected and fake.logged[0][1]['global/update'] == 201
    assert json.loads((tmp_path / 'wandb_history.json').read_text())['next_history_step'] == expected + 1
    assert fake.run.summary['training/status'] == 'running'


@pytest.mark.parametrize('kwargs', [dict(resumed=False), dict(run_id='new'), dict(server_step=None)])
def test_failed_resume_never_silently_logs_new_identity(tmp_path, kwargs):
    fake = FakeWandb(**kwargs)
    with pytest.raises(RuntimeError, match='resume identity/history'):
        TrajectoryWandbMirror(run_dir=tmp_path, spec=WandbTrackingSpec(
            enabled=True, mode='online', resume_run_id='same'), wandb_module=fake)
    assert fake.finished == [1] and not fake.logged
