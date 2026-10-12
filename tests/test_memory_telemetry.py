import json
from trajectory_tracking.training.memory_telemetry import MemoryTelemetry

class FakeCuda:
    def memory_stats(self, device):
        return {'allocated_bytes.all.current':20,'reserved_bytes.all.current':80,'inactive_split_bytes.all.current':30}
    def get_allocator_backend(self): return 'native'
    def mem_get_info(self, device): return (100,200)

def test_telemetry_records_allocator_without_allocating_or_synchronizing(tmp_path):
    p=tmp_path/'memory.jsonl';MemoryTelemetry(p,'cuda:3',cuda=FakeCuda())('update_end',7)
    row=json.loads(p.read_text());assert row['allocated_bytes.all.current']==20
    assert row['inactive_split_bytes.all.current']==30
    assert row['device_used_minus_this_process_reserved_bytes']==20

def test_cpu_does_not_touch_cuda(tmp_path):
    p=tmp_path/'absent';MemoryTelemetry(p,'cpu',cuda=object())('start',0);assert not p.exists()

def test_log_failure_is_nonfatal(tmp_path,capsys):
    MemoryTelemetry(tmp_path/'absent'/'memory.jsonl','cuda:0',cuda=FakeCuda())('end',1)
    assert 'memory telemetry' in capsys.readouterr().err

def test_snapshot_only_at_requested_update(tmp_path):
    class FakeSnapshot(FakeCuda):
        def memory_snapshot(self):return [{'is_expandable':True}]
    p=tmp_path/'memory.jsonl';t=MemoryTelemetry(p,'cuda:0',cuda=FakeSnapshot(),snapshot_update=160)
    t('update_end',159);assert not p.with_suffix('.u160.snapshot.json').exists()
    t('update_end',160);assert json.loads(p.with_suffix('.u160.snapshot.json').read_text())==[{'is_expandable':True}]
