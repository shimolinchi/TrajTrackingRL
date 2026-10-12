/* Replays recorded actor poses only. Physics and policy remain offline. */
(() => {
  const meta = window.failureCase;
  const canvas = document.getElementById('scene'), context = canvas.getContext('2d');
  const pause = document.getElementById('pause'), clock = document.getElementById('clock');
  const result = typeof meta.result === 'object' && meta.result !== null ? meta.result : {};
  const objectLabel = typeof meta.object_id === 'string' ? meta.object_id.slice(0,12) : 'see trajectory identity';
  const policyLabel = meta.checkpoint_sha256 ? `checkpoint u${meta.checkpoint_update}` : `评估 u${meta.checkpoint_update} [live weights; no checkpoint]`;
  document.getElementById('identity').textContent = `实验 ${meta.experiment} · ${policyLabel} · 物体 ${objectLabel} · 案例 ${meta.test_index}`;
  const mean=result.mean_object_flow_error_m, final=result.final_object_flow_error_m;
  const reasons=[];
  if(Number.isFinite(mean))reasons.push(`平均物体跟踪误差 ${(mean*100).toFixed(2)} cm（阈值 3 cm）`);
  if(Number.isFinite(final))reasons.push(`终点误差 ${(final*100).toFixed(2)} cm（阈值 5 cm）`);
  if(result.object_departed)reasons.push('物体离开允许区域');
  document.getElementById('reason').textContent = meta.failure_reason ?? (typeof result.episode_success === 'boolean' ? `本次评估${result.episode_success?'成功':'失败'} · ${reasons.join(' · ')}` : 'Case outcome not attached; see the full heldout report. No success/failure inferred.');
  let frames = [], playing = true, ready = false, anchorWall = 0, anchorSim = 0, sim = 0, cycle = 0, seq = 0;
  let duration = 0, frameIndex = 0;
  const target = meta.camera?.target ?? [0, 0, 0.6];
  const eye = meta.camera?.eye ?? [2, -2, 1.8];
  const delta = eye.map((v, i) => v - target[i]);
  let radius = Math.hypot(...delta), azimuth = Math.atan2(delta[1], delta[0]), elevation = Math.asin(delta[2] / radius);
  function camera() {return {eye:[target[0]+radius*Math.cos(elevation)*Math.cos(azimuth),target[1]+radius*Math.cos(elevation)*Math.sin(azimuth),target[2]+radius*Math.sin(elevation)],target};}
  function current(now) {return anchorSim + (playing && ready ? (now - anchorWall) / 1000 : 0);}
  function toggle() {const now=performance.now();anchorSim=current(now);anchorWall=now;playing=!playing;pause.textContent=playing?'暂停':'播放';}
  pause.onclick = toggle;
  document.getElementById('restart').onclick=()=>{anchorSim=0;anchorWall=performance.now();sim=0;cycle=0;frameIndex=0;};
  document.getElementById('cases').onchange=e=>{location.href='?case='+encodeURIComponent(e.target.value);};
  let drag=null;
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointerup=()=>{drag=null;};
  canvas.onpointermove=e=>{if(!drag)return;azimuth-=(e.clientX-drag[0])*0.008;elevation=Math.max(0.05,Math.min(1.5,elevation+(e.clientY-drag[1])*0.008));drag=[e.clientX,e.clientY];};
  canvas.onwheel=e=>{e.preventDefault();radius=Math.max(.3,Math.min(8,radius*Math.exp(e.deltaY*.001)));};
  function lerp(a,b,u){return a.map((v,i)=>v+(b[i]-v)*u);}
  function quaternion(a,b,u){let dot=a.reduce((s,v,i)=>s+v*b[i],0), sign=dot<0?-1:1;dot=Math.abs(dot);let q;if(dot>.9995){q=a.map((v,i)=>v+(sign*b[i]-v)*u);}else{const theta=Math.acos(Math.min(1,dot)),s=Math.sin(theta);q=a.map((v,i)=>(Math.sin((1-u)*theta)*v+Math.sin(u*theta)*sign*b[i])/s);}const norm=Math.hypot(...q);return q.map(v=>v/norm);}
  function actorsAt(t){while(frameIndex+1<frames.length&&frames[frameIndex+1].t<=t)frameIndex++;while(frameIndex>0&&frames[frameIndex].t>t)frameIndex--;const a=frames[frameIndex],b=frames[Math.min(frameIndex+1,frames.length-1)],u=b.t>a.t?(t-a.t)/(b.t-a.t):0;const actors={};for(const [id,state] of Object.entries(a.actors)){const next=b.actors[id]??state;actors[id]={...state,position_m:lerp(state.position_m,next.position_m,u),joint_positions:lerp(state.joint_positions,next.joint_positions,u),quaternion_xyzw:quaternion(state.quaternion_xyzw,next.quaternion_xyzw,u)};}return actors;}
  function tick(now){
    if(frames.length&&window.visualizationThreeRenderer&&window.visualizationThreeAssetsReady){
      if(!ready){ready=true;anchorWall=now;}
      const elapsed=current(now);cycle=Math.floor(elapsed/duration);sim=elapsed%duration;
      window.visualizationPresentationSnapshot={sequence:++seq,render_revision:seq,actor_sequence:seq,view:camera(),actors:actorsAt(sim),trails:{},polyline_overlays:{}};
      const width=Math.round(canvas.clientWidth),height=Math.round(canvas.clientHeight);if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
      window.visualizationThreeRenderer.drawTo(context,width,height);
      clock.textContent=`${sim.toFixed(2)} / ${duration.toFixed(2)} 秒 · 第 ${cycle+1} 遍`;
    }
    requestAnimationFrame(tick);
  }
  window.failureReplay={status:()=>({ready,playing,sim,duration,cycle,frameIndex,elapsed:current(performance.now()),wall:performance.now(),caseId:meta.case_id,actors:window.visualizationPresentationSnapshot?.actors}),seek:t=>{anchorSim=Math.max(0,Math.min(duration-1e-6,t));anchorWall=performance.now();}};
  fetch('frames?case='+encodeURIComponent(meta.case_id)).then(r=>r.json()).then(data=>{frames=data;duration=frames.at(-1).t;if(frames[0].t!==0||duration<=0||frames.some((f,i)=>i&&f.t<=frames[i-1].t))throw Error('回放时间戳无效');requestAnimationFrame(tick);}).catch(e=>{clock.textContent=e.message;console.error(e);});
})();
