"""CPU-only fault/ownership tests. Synthetic clip bytes are NOT rendered evidence."""
import errno
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from trajectory_tracking.visualization import evaluation_browser_recorder as module


def recorder(tmp_path):
    return module.EvaluationBrowserRecorder(url='http://unused', run_dir=tmp_path / 'run',
        commit='test', tmp_root=tmp_path, ready_timeout=.2, finish_timeout=.2, duration=.2)


def fake_clip(state):
    (state['scratch'] / 'clip.mp4').write_bytes(b'synthetic-bookkeeping-only')
    module.write_json(state['scratch'] / 'result.json', dict(recording_ended_monotonic=time.monotonic()))


@pytest.mark.parametrize('error_number', [errno.ENOSPC, errno.EIO, errno.EACCES])
@pytest.mark.parametrize('fail_start', [False, True])
def test_first_and_secondary_log_failure_preserve_scoring(tmp_path, monkeypatch, capsys,
                                                         error_number, fail_start):
    r = recorder(tmp_path)
    original = Path.open
    def broken(path, *args, **kwargs):
        if path.name == 'recorder.jsonl':
            raise OSError(error_number, 'injected log failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', broken)
    def no_worker(*args, **kwargs):
        raise OSError('injected spawn failure')
    monkeypatch.setattr(module.subprocess, 'Popen', no_worker)
    scoring_error = ValueError('original scoring error')
    with pytest.raises(ValueError) as caught:
        with r.scope(update=0, split='test'):
            if fail_start:
                r.frame_ready()
            raise scoring_error
    assert caught.value is scoring_error
    assert r.active is None
    assert not list(tmp_path.glob('evaluation-video-*'))
    assert 'logging failed' in capsys.readouterr().err
    with r.scope(update=1, split='test'):
        assert r.active is not None  # Failure released lock.


def test_setup_failure_still_evaluates(tmp_path, monkeypatch):
    r = recorder(tmp_path)
    def fail(**kwargs):
        raise OSError('scratch unavailable')
    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', fail)
    with r.scope(update=0, split='test'):
        r.frame_ready()
        assert r.active is None
    assert r.active is None


@pytest.mark.parametrize('scoring_fails', [False, True])
def test_finish_and_cleanup_failure_reaps_real_worker(tmp_path, monkeypatch, scoring_fails):
    r = recorder(tmp_path)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    killpg = os.killpg
    first = True
    def fail_term(pid, sig):
        nonlocal first
        if first:
            first = False
            raise OSError('injected TERM failure')
        killpg(pid, sig)
    monkeypatch.setattr(module.os, 'killpg', fail_term)
    def fail_close():
        raise OSError('injected log close failure')
    original_error = ValueError('score')
    try:
        def evaluate():
            with r.scope(update=1, split='test'):
                r.active['process'] = child
                r.active['log'] = SimpleNamespace(close=fail_close)
                # A missing result / timeout exercises finish failure.
                if scoring_fails:
                    raise original_error
        if scoring_fails:
            with pytest.raises(ValueError) as caught:
                evaluate()
            assert caught.value is original_error
        else:
            evaluate()
        assert child.poll() is not None
        assert r.active is None
        with r.scope(update=2, split='test'):
            assert r.active is not None
    finally:
        if child.poll() is None:
            killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def _publish_many(base, start, events, participant):
    r = recorder(Path(base))
    start.wait(90)
    for index in range(3):
        for attempt in range(200):
            with r.scope(update=participant * 10 + index, split='test'):
                state = r.active
                if state is not None:
                    module.rotate_owned(r.directory, r.run_dir.name, reserve=1)
                    fake_clip(state)
                    count = len(list(r.directory.glob('eval_*.mp4'))) + 1
                    events.put(('reserved', count, state['name']))
                    time.sleep(.02)
                    # No child needed for this bookkeeping test; use production publication.
                    state['process'] = SimpleNamespace(wait=lambda timeout: 0)
                    r._finish(state, time.monotonic())
                    state['process'] = None
                    events.put(('published', len(list(r.directory.glob('eval_*.mp4'))), state['name']))
                    break
            time.sleep(.01)
        else:
            raise AssertionError('bounded contention retries exhausted')


def test_two_processes_six_publications_never_exceed_three(tmp_path):
    ctx = mp.get_context('spawn')
    start, events = ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=_publish_many, args=(str(tmp_path), start, events, i)) for i in range(2)]
    for child in children:
        child.start()
    start.set()
    for child in children:
        child.join(90)
        if child.is_alive():
            child.terminate(); child.join(5)
        assert child.exitcode == 0
    results = [events.get(timeout=2) for _ in range(12)]
    assert all(count <= 3 for _, count, _ in results)
    assert sum(kind == 'published' for kind, _, _ in results) == 6
    clips = sorted((tmp_path / 'run/evaluation_videos').glob('eval_*.mp4'))
    assert len(clips) == 3
    assert [p.stem for p in clips] == sorted(name for kind, _, name in results if kind == 'published')[-3:]
    assert all(p.read_bytes() == b'synthetic-bookkeeping-only' for p in clips)
    events.close()


def test_foreign_corrupt_and_symlinks_survive(tmp_path):
    directory = recorder(tmp_path).directory
    originals = {}
    for name, metadata in [('foreign', dict(owner='someone_else', run_id='run')),
                           ('other_run', dict(owner=module.OWNER, run_id='other')),
                           ('corrupt', None)]:
        clip = directory / f'eval_{name}.mp4'; clip.write_bytes(b'user')
        sidecar = clip.with_suffix('.json')
        sidecar.write_text('invalid' if metadata is None else json.dumps(metadata))
        originals[clip] = clip.read_bytes(); originals[sidecar] = sidecar.read_bytes()
    (directory / 'eval_link.mp4').symlink_to(directory / 'eval_foreign.mp4')
    (directory / 'eval_link.json').write_text(json.dumps(dict(owner=module.OWNER, run_id='run')))
    (directory / 'eval_meta_link.json').symlink_to(directory / 'eval_link.json')
    for i in range(4):
        module.rotate_owned(directory, 'run', reserve=1)
        (directory / f'eval_{i}.mp4').write_bytes(b'synthetic')
        (directory / f'eval_{i}.json').write_text(json.dumps(dict(owner=module.OWNER, run_id='run')))
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert (directory / 'eval_link.mp4').is_symlink()
    assert (directory / 'eval_meta_link.json').is_symlink()


def _orphan_parent(base, connection):
    r = recorder(Path(base))
    r.ready_timeout = 5
    popen = subprocess.Popen
    # Exercise production frame_ready/pass_fds, replacing only the browser command.
    def worker(args, **kwargs):
        scratch = args[-1]
        code = "import pathlib,time; p=pathlib.Path(__import__('sys').argv[1]); (p/'ready.json').write_text('{}'); deadline=time.monotonic()+15\nwhile not (p.parent/'release_worker').exists() and time.monotonic()<deadline: time.sleep(.02)"
        child = popen([sys.executable, '-c', code, scratch], **kwargs)
        connection.send(child.pid)
        return child
    module.subprocess.Popen = worker
    with r.scope(update=0, split='test'):
        r.frame_ready()
        assert (r.active['scratch'] / 'ready.json').exists()
        connection.send('ready')
        os._exit(0)  # Deliberately bypass parent cleanup; worker must retain lock.


def test_worker_inherits_lock_when_parent_exits(tmp_path):
    ctx = mp.get_context('spawn'); receive, send = ctx.Pipe(False)
    parent = ctx.Process(target=_orphan_parent, args=(str(tmp_path), send))
    parent.start()
    assert receive.poll(90)
    pid = receive.recv()
    try:
        assert receive.poll(5) and receive.recv() == 'ready'
        parent.join(5); assert parent.exitcode == 0
        r = recorder(tmp_path)
        started = time.monotonic()
        with r.scope(update=1, split='test'):
            assert r.active is None
        assert time.monotonic() - started < 1
        # Release the fake worker; the kernel then releases its inherited fd.
        (tmp_path / 'release_worker').write_text('exit')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with r.scope(update=2, split='test'):
                if r.active is not None:
                    break
            time.sleep(.05)
        else:
            pytest.fail('worker exit did not release lock')
        assert r.active is None
    finally:
        if parent.is_alive():
            parent.terminate(); parent.join(5)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        receive.close(); send.close()


def test_log_write_and_stderr_failure_do_not_stop_successful_evaluation(tmp_path, monkeypatch):
    r = recorder(tmp_path)
    original = Path.open
    class BrokenStream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, value): raise OSError(errno.ENOSPC, 'write full')
        def flush(self): raise OSError(errno.EIO, 'flush failed')
    def broken(path, *args, **kwargs):
        return BrokenStream() if path.name == 'recorder.jsonl' else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', broken)
    monkeypatch.setattr(module.sys, 'stderr', BrokenStream())
    with r.scope(update=1, split='test'):
        reached = True
    assert reached and r.active is None


def test_ready_timeout_reaps_real_child(tmp_path, monkeypatch):
    r = recorder(tmp_path)
    original = subprocess.Popen
    children = []
    def unready(args, **kwargs):
        child = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', unready)
    with r.scope(update=0, split='test'):
        r.frame_ready()
        assert children[0].poll() is not None
    assert r.active is None
    with r.scope(update=1, split='test'):
        assert r.active is not None
