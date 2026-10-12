"""The driver's minor number must never substitute for the physical list index."""
from trajectory_tracking.cli.launch import exclusive_gpus

def test_index_uuid_mapping_with_reversed_driver_numbers(monkeypatch):
    xml = """<nvidia_smi_log><gpu><minor_number>7</minor_number><uuid>GPU-A</uuid>
    <processes/><fb_memory_usage><used>1 MiB</used></fb_memory_usage>
    <utilization><gpu_util>0 %</gpu_util></utilization></gpu>
    <gpu><minor_number>0</minor_number><uuid>GPU-B</uuid><processes><process_info/></processes>
    <fb_memory_usage><used>2000 MiB</used></fb_memory_usage>
    <utilization><gpu_util>80 %</gpu_util></utilization></gpu></nvidia_smi_log>"""
    def query(command, **kwargs):
        return xml if "-x" in command else "0, GPU-A" + chr(10) + "1, GPU-B" + chr(10)
    monkeypatch.setattr("trajectory_tracking.cli.launch.subprocess.check_output", query)
    assert exclusive_gpus([0]) == [{"index": 0, "uuid": "GPU-A"}]
