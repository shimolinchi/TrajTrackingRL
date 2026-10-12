"""One real primary-rank evaluation case per scope, replayed on simulation time.

Capture only observes the first local batch; no simulator calls, reset, or scoring
changes. The existing browser recorder separately provides its 60s wall-clock clip.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import re
from threading import Thread
from http.server import ThreadingHTTPServer
from trajectory_tracking.visualization.state_replay.failure_html_server import make_handler
from trajectory_tracking.visualization.evaluation_browser_recorder import write_json
from trajectory_tracking.visualization.viewers.trajectory_tracking import (build_trajectory_tracking_browser_frame,
    _build_trajectory_tracking_browser_scene, _TRAJECTORY_BROWSER_CAMERA_EYE, _TRAJECTORY_BROWSER_CAMERA_TARGET)

class EvaluationStateReplay:
    def __init__(self, *, source, run_dir, commit, port, event, experiment_name, test_sources=(), test_indices=()):
        self.source, self.run_dir, self.commit, self.event = source, Path(run_dir), commit, event
        self.parent = self.run_dir/'evaluation_videos'
        self.parent.mkdir(exist_ok=True)
        self.experiment = int(re.match(r'exp(\d+)', experiment_name).group(1)) if re.match(r'exp(\d+)', experiment_name) else 0
        self.test_sources, self.test_indices = test_sources, test_indices
        self.latest = self.parent/'html_latest_cases'
        self.latest.mkdir(exist_ok=True)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(self.latest))
        Thread(target=self.server.serve_forever,daemon=True).start()
        self.active=None

    @contextmanager
    def scope(self, **metadata):
        self.active=dict(metadata=metadata,frames=[],done=False,disabled=False)
        try:
            yield
        except BaseException:
            raise
        else:
            try:self.publish()
            except Exception as error:self.event('html_publish_failed',error=repr(error))
        finally:self.active=None

    def observe(self, env):
        s=self.active
        if s is None or s['done'] or s['disabled']:return
        try:
            cursor=int(env.reference_cursors[0].item())
            if s['frames']:
                previous=s['frames'][-1]['reference_cursor']
                if cursor<previous:s['done']=True;return
                if cursor==previous:return
                if cursor!=previous+1:raise ValueError('evaluation cursor gap')
            else:
                if cursor!=0:raise ValueError('HTML requires local0 evaluation')
                s['hz']=env.config.timing.control_hz
                s['terminal']=int(env.episode.terminal_cursors[0].item())
                s['trajectory_index']=int(env.episode.trajectory_indices[0].item())
                if s['terminal']>12000:raise ValueError('HTML capture bound12000 exceeded')
            frame=build_trajectory_tracking_browser_frame(env,env_index=0,include_reference_flow=False)
            s['frames'].append(dict(t=cursor/s['hz'],reference_cursor=cursor,
                actors={a.actor_id:a.as_payload() for a in frame.actor_states},
                actual_object_surface_points=frame.actual_object_surface_points))
            if cursor==s['terminal']:s['done']=True
        except Exception as error:
            s['disabled']=True
            self.event('html_capture_failed',error=repr(error))

    def publish(self):
        s=self.active
        if s['disabled'] or not s['frames']:return
        if not s['done'] or len(s['frames'])!=s['terminal']+1:raise ValueError('incomplete endpoint HTML')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        group=self.parent/f'html_eval_{stamp}'
        # Reserve one group before publishing; do not touch foreign or damaged directories.
        owned=[]
        for previous in sorted(self.parent.glob('html_eval_*')):
            if previous.is_symlink() or {p.name for p in previous.iterdir()} != {'case.json','owner.json'}:continue
            try:identity=json.loads((previous/'owner.json').read_text())
            except (OSError,ValueError):continue
            if identity==dict(owner='native_evaluation_html_v1',run=str(self.run_dir)):owned.append(previous)
        for previous in owned[:-2]:shutil.rmtree(previous)
        group.mkdir()
        source=self.source
        variant=source.object_variants[source.object_layout[0]] if source.object_variants else source
        visual=_build_trajectory_tracking_browser_scene(source.robot_urdf,flow_points=4,flow_horizon_steps=s['terminal']+1,
            object_flow_enabled=False,object_urdf=variant.object_urdf,object_size_m=variant.object_override.object_size_m)
        local=s['trajectory_index']
        index=self.test_indices[local] if s['metadata'].get('split')=='test' and self.test_indices else local
        identity=self.test_sources[local].entry.episode_uid if s['metadata'].get('split')=='test' and self.test_sources else None
        case_id=f'exp{self.experiment}_test{index}'
        case=dict(case_id=case_id,experiment=self.experiment,test_index=index,trajectory_identity=identity,checkpoint_update=s['metadata']['update'],
            checkpoint_sha256=None,policy_identity='live frozen deterministic evaluation weights at update; no checkpoint fabricated',
            runtime_source_commit=self.commit,original_run=str(self.run_dir),result='see full heldout report; local sampled case',
            local_trajectory_index=s['trajectory_index'],capture_protocol='actual first primary-rank evaluation slot; full endpoint',
            scene=visual.as_payload(),assets={url:dict(local_path=str(a.local_path),content_type=a.content_type,sha256=a.content_sha256) for url,a in visual.asset_files.items()},
            frames=s['frames'],camera=dict(eye=_TRAJECTORY_BROWSER_CAMERA_EYE,target=_TRAJECTORY_BROWSER_CAMERA_TARGET),
            duration_s=s['frames'][-1]['t'],control_hz=s['hz'],time_mapping='1 simulation second = 1 playback second; explicit looping; no wall-clock sampling',
            metadata=s['metadata'])
        write_json(group/'case.json',case)
        write_json(group/'owner.json',dict(owner='native_evaluation_html_v1',run=str(self.run_dir)))
        for old in self.latest.glob('exp*_test*.json'):old.unlink()
        write_json(self.latest/(case_id+'.json'),case)
        self.event('html_complete',path=str(group),duration_s=case['duration_s'],playback_speed=1)

    def close(self):
        self.server.shutdown();self.server.server_close()
