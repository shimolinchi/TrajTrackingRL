import json
import pytest
from trajectory_tracking.environment.physics_health import PhysicsHealthGuard
from trajectory_tracking.training.memory_telemetry import MemoryBudgetError, MemoryTelemetry

@pytest.mark.parametrize("message", [
    "PhysX error: collisionStackSize buffer overflow detected. Contacts have been dropped.",
    "PhysX error: PxgCudaDeviceMemoryAllocator failed to allocate memory 67108864 bytes! Result = 2",
    "PhysX Internal CUDA error. Simulation cannot continue! Error code 2!",
])
def test_native_errors_latch_and_raise_at_boundary(tmp_path, message):
    guard = PhysicsHealthGuard(tmp_path/"failure.json")
    guard.on_log("omni.physx.plugin", 0, "native.cpp", 42, message)
    guard.on_log("omni.physx.plugin", 0, "later.cpp", 1, "PhysX error: secondary failure")
    with pytest.raises(RuntimeError, match="simulation invalid"):
        guard.check()
    assert json.loads((tmp_path/"failure.json").read_text())["message"] == message

def test_warning_and_foreign_plugin_do_not_abort(tmp_path):
    guard = PhysicsHealthGuard(tmp_path/"failure.json")
    guard.on_log("omni.physx.plugin", 0, "", 0, "TGS velocity iteration behavior changed")
    guard.on_log("unrelated", 0, "", 0, "CUDA error")
    guard.check()
    assert not guard.path.exists()

def test_memory_budget_failure_is_not_swallowed(tmp_path):
    class FakeCuda:
        def memory_stats(self, device): return {}
        def get_allocator_backend(self): return "test"
        def mem_get_info(self, device): return (100, 200)
    path=tmp_path/"memory.jsonl"
    telemetry=MemoryTelemetry(path,"cuda:0",cuda=FakeCuda(),min_free_bytes=101)
    with pytest.raises(MemoryBudgetError):
        telemetry("update_begin", 42)
    assert json.loads(path.read_text())["device_free_bytes"]==100


def test_incremental_native_log_handles_partial_lines_and_latches_first_error(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[Info] [omni.physx.plugin] initialized\n"
                   "[Info] [unrelated] PhysX error: example only\n")
    guard = PhysicsHealthGuard(tmp_path / "failure.json")
    guard.watch(log)
    guard.check()
    with log.open("a") as stream:
        stream.write("[Error] [omni.physx.plugin] PhysX error: buffer over")
    guard.check()
    with log.open("a") as stream:
        stream.write("flow detected. Contacts have been dropped.\n")
    with pytest.raises(RuntimeError, match="buffer overflow"):
        guard.check()
    assert "Contacts have been dropped" in json.loads(guard.path.read_text())["message"]
    guard.close()


def test_missing_native_log_fails_closed(tmp_path):
    guard = PhysicsHealthGuard(tmp_path / "failure.json")
    with pytest.raises(FileNotFoundError):
        guard.watch(tmp_path / "missing.log")
