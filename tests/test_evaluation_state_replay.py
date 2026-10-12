from types import SimpleNamespace as NS
import json
import torch
import pytest
from trajectory_tracking.visualization.viewers import evaluation_state_replay as m

def test_real_clock_endpoint_retention_and_cleanup(tmp_path,monkeypatch):
 visual=NS(as_payload=lambda:{},asset_files={})
 monkeypatch.setattr(m,'_build_trajectory_tracking_browser_scene',lambda *a,**k:visual)
 monkeypatch.setattr(m,'build_trajectory_tracking_browser_frame',lambda *a,**k:NS(actor_states=[],actual_object_surface_points=[]))
 source=NS(object_variants=(),robot_urdf='unused',object_urdf='unused',object_override=NS(object_size_m=1))
 events=[];r=m.EvaluationStateReplay(source=source,run_dir=tmp_path,commit='test',port=0,event=lambda *a,**k:events.append((a,k)),experiment_name='exp168_test')
 env=NS(config=NS(timing=NS(control_hz=60)),episode=NS(terminal_cursors=torch.tensor([3]),trajectory_indices=torch.tensor([0])))
 try:
  for u in range(4):
   with r.scope(update=u,split='test'):
    for i in range(4):env.reference_cursors=torch.tensor([i]);r.observe(env)
   assert len(list((tmp_path/'evaluation_videos').glob('html_eval_*')))<=3
  cases=list((tmp_path/'evaluation_videos').glob('html_eval_*/case.json'));assert len(cases)==3
  case=json.loads(cases[-1].read_text());assert [f['t'] for f in case['frames']]==[0,1/60,2/60,3/60]
  with pytest.raises(RuntimeError):
   with r.scope(update=9,split='test'):raise RuntimeError('score error')
  assert r.active is None
 finally:r.close()

def test_gap_reports_failure_not_fake_html(tmp_path,monkeypatch):
 events=[];r=m.EvaluationStateReplay(source=None,run_dir=tmp_path,commit='test',port=0,event=lambda *a,**k:events.append(a[0]),experiment_name='smoke')
 try:
  with r.scope(update=0,split='test'):
   r.observe(NS(reference_cursors=torch.tensor([5])))
  assert events==['html_capture_failed']
  assert not list((tmp_path/'evaluation_videos').glob('html_eval_*'))
 finally:r.close()
