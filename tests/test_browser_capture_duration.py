"""A delayed first canvas frame must not extend the fixed-length recording."""
import json
import shutil
import subprocess
import pytest
from trajectory_tracking.visualization.browser_capture import _transcode_to_mp4

@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_offset_sparse_canvas_frames_have_exact_duration(tmp_path):
    source, target = tmp_path/'source.webm', tmp_path/'result.mp4'
    subprocess.run(['ffmpeg','-loglevel','error','-y','-f','lavfi','-i',
        'color=c=blue:s=64x64:r=4:d=1','-vf','setpts=PTS+0.8/TB',
        '-vsync','0','-c:v','libvpx',str(source)],check=True,timeout=30)
    _transcode_to_mp4(source,target,duration_s=2)
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=start_time,r_frame_rate','-of','json',str(target)],text=True))
    assert float(d['format']['duration']) == pytest.approx(2,abs=.04)
    assert float(d['streams'][0]['start_time']) == 0
    assert d['streams'][0]['r_frame_rate'] == '30/1'
