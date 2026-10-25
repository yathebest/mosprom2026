"""Live inference with latest-frame buffering and independent incident clips."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from dataclasses import replace
import logging
import os
import gc
import queue
import subprocess
import threading
import time
import uuid
import sys

import cv2
import numpy as np

from . import storage as db
from .core import Rules
from .inference import create_backend,PeopleTracker
from .streams import open_capture
from .video import MODEL,draw,ffmpeg_path,state_snapshot,finish_rules,persist_episode_updates
from .scenarios import ScenarioRules,base_config,annotate_objects,assessment_rows
from .preview_motion import PreviewMotion
from .access import current_scope
from .clip_groups import GroupPool

SESSIONS={}
LOCK=threading.Lock()
CLIPS=ThreadPoolExecutor(max_workers=2,thread_name_prefix='live-clip')
PRE_SECONDS=3.
POST_SECONDS=4.
SAMPLE_FPS=10.
DISPLAY_FPS=25.
MAX_LIVE_SESSIONS=2
ACTIVE_STATES=('starting','running','stopping')


def latest_only(target,item):
    """A slow consumer gets the newest frame, never a growing backlog."""
    dropped=0
    try:target.put_nowait(item)
    except queue.Full:
        try:target.get_nowait();dropped=1
        except queue.Empty:pass
        try:target.put_nowait(item)
        except queue.Full:dropped=1
    return dropped


def measured_fps(times):
    values=list(times)
    return round((len(values)-1)/max(values[-1]-values[0],.001),2) if len(values)>1 else 0.


class FramePacer:
    """Bound presentation rate without discarding every jittered early frame."""
    def __init__(self,fps):self.period=1/fps;self.due=None
    def accept(self,captured):
        if self.due is None:self.due=captured
        if captured+.002<self.due:return False
        self.due=max(self.due+self.period,captured)
        return True


class LiveSession:
    def __init__(self,config,source_type,url=None):
        cv2.setNumThreads(1)
        self.id=uuid.uuid4().hex;self.config=config;self.source_type=source_type;self.url=url
        scope=current_scope.get();self.organization_id=scope.organization_id if scope else 'legacy'
        self.folder=db.DATA/'runs'/self.id;self.folder.mkdir(parents=True)
        (self.folder/'clips').mkdir()
        self.stop_event=threading.Event();self.frames=queue.Queue(maxsize=1)
        self.state='starting';self.error=None;self.jpeg=None;self.version=0;self.stopped_at=None
        self.started=datetime.now(timezone.utc);self.first_mono=None;self.frame_origin=None
        self.last_ingest=time.monotonic();self.last_processed=None
        self.processed=0;self.dropped=0;self.elapsed=0.;self.events=0;self.pending=[]
        self.ring=deque();self.latencies=[];self.clip_futures=[];self.backend=None;self.backend_name=None
        self.created=time.monotonic()
        self.assessments=[]
        self.people_states={};self.scenario_states={};self.states_updated_s=0.
        self.effective_observation_gap_s=max(2.5,config.observation_gap)
        self.frame_lock=threading.RLock()
        self.clip_groups=GroupPool(self.id,config.camera,self.organization_id,pre_s=PRE_SECONDS,post_s=POST_SECONDS)
        self.display_frames=queue.Queue(maxsize=1);self.scenario_frames=queue.Queue(maxsize=1)
        self.record_frames=queue.Queue(maxsize=1);self.record_times=deque(maxlen=150)
        self.visual=None;self.scenario_visual=None
        self.capture_times=deque(maxlen=150);self.display_times=deque(maxlen=150)
        self.analysis_times=deque(maxlen=100);self.scenario_times=deque(maxlen=100)
        self.scenario_latencies=[];self.analysis_age_ms=None;self.scenario_age_ms=None
        self.capture_count=0;self.display_count=0;self.scenario_count=0
        self.render_latencies=deque(maxlen=150);self.jpeg_latencies=deque(maxlen=150)
        self.preview_age_ms=None
        self.pipeline_mode='starting'
        self.helpers=[];self.shape=None
        db.add_job(self.id,'Прямой эфир · '+config.camera,
                   {'live':True,'mode':'live','source_type':source_type,'duration_s':0,
                    'fps':SAMPLE_FPS,'width':0,'height':0,'frames':0})
        db.update_job(self.id,status='queued',config=config.model_dump())

    def start(self):
        self.worker=threading.Thread(target=self.run,name='live-'+self.id[:6],daemon=True)
        self.worker.start()
        return self

    def ingest(self,frame,encoded=None):
        if self.stop_event.is_set():return False
        captured=time.monotonic()
        # Original pixels go to inference. Only the display copy is resized.
        with self.frame_lock:
            if self.stop_event.is_set() or self.state not in ('starting','running'):return False
            self.last_ingest=captured
            if self.first_mono is None:
                self.first_mono=captured;self.frame_origin=datetime.now(timezone.utc)
                self.shape=frame.shape
            elif frame.shape!=self.shape:
                self.error='Источник изменил разрешение. Запустите новый эфир.';self.stop_event.set();return False
            self.elapsed=captured-self.first_mono
            self.capture_times.append(captured);self.capture_count+=1
        self.dropped+=latest_only(self.frames,(captured,frame))
        latest_only(self.display_frames,(captured,frame))
        latest_only(self.record_frames,(captured,frame,encoded))
        return True

    def stop(self):
        with self.frame_lock:
            if self.state in ('starting','running'):self.state='stopping'
            self.stop_event.set()

    def encode_recording(self):
        """Full-resolution evidence encoding never delays the live preview."""
        try:
            while not self.stop_event.is_set():
                try:captured,frame,original=self.record_frames.get(timeout=.2)
                except queue.Empty:continue
                now=captured-self.first_mono
                # Browser frames already arrive as a validated full-resolution
                # JPEG. Preserve those bytes rather than JPEG-encoding again.
                if original is None:
                    ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if not ok:continue
                    original=encoded.tobytes()
                with self.frame_lock:
                    self.ring.append((now,original))
                    history_s=PRE_SECONDS+min(15.,max(4.,(self.scenario_age_ms or 0)/1000+1.))
                    while self.ring and now-self.ring[0][0]>history_s:self.ring.popleft()
                    for clip in self.pending:
                        if clip.get('start',0)<=now<=clip['until']:clip['frames'].append((now,original))
                self.record_times.append(time.monotonic())
                self.queue_finished_clips(now)
        except Exception:
            logging.exception('Live recording failed: %s',self.id)
            self.error='Ошибка записи кадров происшествий.';self.stop_event.set()

    def render_frames(self):
        """Capture evidence and render independently of all neural networks."""
        pacer=FramePacer(DISPLAY_FPS)
        motion=PreviewMotion()
        try:
            while not self.stop_event.is_set():
                try:captured,frame=self.display_frames.get(timeout=.2)
                except queue.Empty:continue
                if not pacer.accept(captured):continue
                render_began=time.perf_counter()
                now=captured-self.first_mono
                if self.display_count==0:
                    h,w=frame.shape[:2];cv2.imwrite(str(self.folder/'preview.jpg'),frame)
                    db.update_job(self.id,meta={'live':True,'mode':'live','source_type':self.source_type,
                        'started_at':self.frame_origin.isoformat(),'fps':DISPLAY_FPS,'width':w,'height':h,'frames':0,'duration_s':0})
                h,w=frame.shape[:2];scale=min(1,960/max(h,w))
                preview=cv2.resize(frame,(round(w*scale),round(h*scale))) if scale<1 else frame
                visual=self.visual;annotated=preview.copy()
                if visual and now-visual['time']<=2.5:
                    # Flow follows current preview pixels. These blue projected
                    # boxes never enter tracking/rules or confirm an incident.
                    people=motion.project(visual,preview,now)
                    preview_details={tid:{**row,'tracking':'preview flow'} for tid,row in visual['details'].items()}
                    annotated=draw(preview,people,preview_details,self.config,now,
                                   visual['scene'],visual['scene_alerts'])
                    scenario=self.scenario_visual
                    if scenario and now-scenario['time']<=2:
                        # These are display-only cached assessments. Rules never
                        # receive them again as fresh measurements.
                        details={tid:{key:{**value,'fresh':False,'reason':'recent_observation'}
                                      for key,value in rows.items()} for tid,rows in scenario['details'].items()}
                        objects=[replace(o,box=tuple(v*scale for v in o.box)) for o in scenario['objects']] if now-scenario['time']<=.5 else []
                        annotated=annotate_objects(annotated,objects,scenario['status'],people,details,scenario.get('states'))
                label='ANALYSIS: WARMUP' if not visual else f"ANALYSIS AGE: {max(0,now-visual['time']):.2f}s"
                cv2.putText(annotated,label,(14,52),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,220,245),2)
                jpeg_began=time.perf_counter()
                ok,encoded=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,82])
                self.jpeg_latencies.append((time.perf_counter()-jpeg_began)*1000)
                if ok:
                    self.jpeg=encoded.tobytes();self.version+=1
                    self.display_count+=1;self.display_times.append(time.monotonic())
                    if self.display_count%25==0:(self.folder/'latest.jpg').write_bytes(self.jpeg)
                    self.preview_age_ms=round((time.monotonic()-captured)*1000,1)
                self.render_latencies.append((time.perf_counter()-render_began)*1000)
                self.queue_finished_clips(now)
        except Exception:
            logging.exception('Live rendering failed: %s',self.id)
            self.error='Ошибка видеопотока. Запустите эфир заново.';self.stop_event.set()

    def record_events(self,events,annotated,source_time):
        for event in events:
            arrived=time.monotonic()-self.first_mono
            event.update(live=True,clip_state='capturing',processing_delay_s=round(max(0,arrived-event['alert_s']),3),
                clip_scope='incident_excerpt',
                occurred_at=(self.frame_origin+timedelta(seconds=event['alert_s'])).isoformat(),
                priority=next((z.priority for z in self.config.zones if z.name==event.get('zone')),None))
            image=f'runs/{self.id}/evidence_{uuid.uuid4().hex}.jpg'
            cv2.imwrite(str(db.DATA/image),annotated)
            saved_id,created=db.add_event(self.id,self.config.camera,event,image,return_created=True)
            if not created:
                (db.DATA/image).unlink(missing_ok=True)
                continue
            with self.frame_lock:
                group=self.clip_groups.add(saved_id,event)
                clip=next((item for item in self.pending if item['id']==group.id),None)
                if group.state=='pending':
                    if clip is None:
                        clip={'id':group.id,'group':group,'frames':[]}
                        self.pending.append(clip)
                    clip.update(start=group.start_s,until=group.end_s,alert_s=group.first_alert_s)
                    merged={t:b for t,b in clip['frames']}
                    merged.update((t,b) for t,b in self.ring if group.start_s<=t<=group.end_s)
                    clip['frames']=sorted(merged.items())
                result={key:value for key,value in group.result.items() if key not in ('clip_path','clip_alert_offset_s')}
                updates={member_id:{**group.member_metadata(member_id,start_s=group.result.get('clip_start_s',group.start_s)),
                    'clip_state':'capturing' if group.state=='pending' else group.state,
                    **result} for member_id in group.members}
                db.set_group_clip(self.id,self.config.camera,updates,
                    clip=group.result.get('clip_path'),organization_id=self.organization_id)
                self.events+=1
        self.queue_finished_clips(self.elapsed)

    def analyze_scenarios(self):
        # Slower object models have their own latest-frame queue. Each result
        # enters the rule engine exactly once, at its source timestamp.
        cfg=self.config.model_copy(update={'observation_gap':self.effective_observation_gap_s})
        rules=ScenarioRules(cfg)
        now=observed_now=0.
        try:
            while not self.stop_event.is_set():
                try:now,frame,people,details=self.scenario_frames.get(timeout=.2)
                except queue.Empty:continue
                start=time.perf_counter();objects=self.backend.predict_scenarios(frame,people)
                self.scenario_latencies.append((time.perf_counter()-start)*1000)
                self.scenario_latencies=self.scenario_latencies[-300:]
                events,active=rules.step(people,objects,now,frame.shape[1],frame.shape[0],details)
                observed_now=now
                updates=list(getattr(rules,'event_updates',[]))
                with self.frame_lock:
                    self.scenario_states=state_snapshot(rules);self.states_updated_s=max(self.states_updated_s,now)
                self.assessments=assessment_rows(people,rules.ppe_details)
                self.scenario_visual={'time':now,'objects':objects,'details':rules.ppe_details,'status':rules.ppe_status,'states':state_snapshot(rules)}
                self.scenario_count+=1;self.scenario_times.append(time.monotonic())
                self.scenario_age_ms=round((time.monotonic()-self.first_mono-now)*1000,1)
                for tid,kinds in active.items():
                    if tid in details:details[tid]['active'].extend(kinds)
                if events or updates:
                    annotated=draw(frame,people,details,self.config,now)
                    annotated=annotate_objects(annotated,objects,rules.ppe_status,people,rules.ppe_details,rules.state_rows)
                    self.record_events(events,annotated,now)
                    persist_episode_updates(self.id,updates,annotated)
        except Exception:
            logging.exception('Live scenario inference failed: %s',self.id)
            self.error='Ошибка анализа СИЗ и объектов. Запустите эфир заново.';self.stop_event.set()
        finally:
            persist_episode_updates(self.id,finish_rules(rules,observed_now,'processing_failed' if self.error else 'stream_stopped'))
            with self.frame_lock:self.scenario_states=state_snapshot(rules)

    def read_network(self):
        cap=None
        try:
            cap=open_capture(self.url)
            while not self.stop_event.is_set():
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Камера перестала передавать кадры')
                self.ingest(frame)
        except Exception:
            if not self.stop_event.is_set():
                self.error='Потеряно соединение с камерой. Проверьте источник и запустите новый сеанс.'
                self.stop_event.set()
        finally:
            if cap is not None:cap.release()

    def report(self):
        current=time.monotonic()
        with self.frame_lock:
            states={tid:dict(rows) for tid,rows in self.people_states.items()}
            for tid,rows in self.scenario_states.items():states.setdefault(tid,{}).update(rows)
        return {'id':self.id,'camera':self.config.camera,'state':self.state,'error':self.error,'stopped_at':self.stopped_at,
                'current_states':states,'current_states_updated_s':self.states_updated_s,
                'effective_observation_gap_s':self.effective_observation_gap_s,
                'effective_people_observation_gap_s':self.effective_observation_gap_s,
                'effective_scenario_observation_gap_s':self.effective_observation_gap_s,
                'config':self.config.model_dump(),'job_id':self.id,
                'source_type':self.source_type,'started_at':self.started.isoformat(),
                'elapsed_s':round(self.elapsed,2),'processed_frames':self.processed,
                'capture_fps':measured_fps(self.capture_times),'display_fps':measured_fps(self.display_times),
                'recording_fps':measured_fps(self.record_times),
                'analysis_fps':measured_fps(self.analysis_times),'scenario_fps':measured_fps(self.scenario_times),
                'captured_frames':self.capture_count,'displayed_frames':self.display_count,'scenario_frames':self.scenario_count,
                'analysis_age_ms':self.analysis_age_ms,'scenario_age_ms':self.scenario_age_ms,
                'preview_age_ms':self.preview_age_ms,
                'capture_age_ms':round(max(0,current-self.last_ingest)*1000,1) if self.capture_count else None,
                'preview_frame_age_ms':round(max(0,current-self.display_times[-1])*1000,1) if self.display_times else None,
                'preview_version':self.version,'snapshot_url':f'/api/live/{self.id}/snapshot',
                'max_live_sessions':MAX_LIVE_SESSIONS,
                'render_p95_ms':round(float(np.percentile(self.render_latencies,95)),2) if self.render_latencies else None,
                'jpeg_p95_ms':round(float(np.percentile(self.jpeg_latencies,95)),2) if self.jpeg_latencies else None,
                'inference_pipeline':self.pipeline_mode,
                'fast_array_prediction':getattr(self.backend,'fast_array_prediction',False),
                'scenario_p95_ms':round(float(np.percentile(self.scenario_latencies,95)),2) if self.scenario_latencies else None,
                'dropped_frames':self.dropped,'events':self.events,
                'clips_pending':len(self.pending)+sum(not f.done() for f in self.clip_futures),
                'inference_p95_ms':round(float(np.percentile(self.latencies,95)),2) if self.latencies else None,
                'view_url':f'/api/live/{self.id}/view','clip_pre_s':PRE_SECONDS,'clip_post_s':POST_SECONDS,
                'timing':'Server frame-arrival time; network/camera latency is not measured',
                'model':getattr(self.backend,'name',self.backend_name),'assessments':self.assessments}

    def finalize_clip(self,clip):
        event_id=clip['id'];frames=sorted(dict(clip['frames']).items());group=clip.get('group')
        def publish(patch,path=None):
            if group is None:
                if path is not None:db.set_clip(event_id,path)
                db.patch_payload(event_id,patch)
            else:
                with self.frame_lock:
                    updates={member_id:{**group.member_metadata(member_id,start_s=patch.get('clip_start_s',group.start_s)),
                                        **patch} for member_id in group.members}
                    # Offset is per event, not the first actor's alert offset.
                    for member_id,update in updates.items():
                        update['clip_alert_offset_s']=group.member_metadata(member_id,start_s=patch.get('clip_start_s',group.start_s))['clip_alert_offset_s']
                    db.set_group_clip(self.id,self.config.camera,updates,clip=path,organization_id=self.organization_id)
                    if patch.get('clip_state') in ('ready','failed','cancelled'):
                        self.clip_groups.finish(group.id,patch['clip_state'],{**patch,**({'clip_path':path} if path else {})})
        if not frames:
            publish({'clip_state':'failed'})
            return
        publish({'clip_state':'exporting'})
        target=self.folder/'clips'/f'{event_id}.mp4'
        process=None
        try:
            start,end=frames[0][0],frames[-1][0]
            executable=ffmpeg_path()
            if not executable:raise RuntimeError('H264 encoder unavailable')
            # Decode the JPEG sequence and encode H264 once in native FFmpeg.
            # Avoid Python JPEG decode -> MPEG4 encode -> decode -> H264 encode.
            process=subprocess.Popen([executable,'-hide_banner','-loglevel','error','-y',
                '-f','image2pipe','-vcodec','mjpeg','-framerate',str(DISPLAY_FPS),'-i','pipe:0',
                '-an','-c:v','libx264','-threads','2','-preset','ultrafast','-crf','23',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            index=0
            for t in np.arange(start,end+.0001,1/DISPLAY_FPS):
                while index+1<len(frames) and frames[index+1][0]<=t:index+=1
                process.stdin.write(frames[index][1])
            process.stdin.close();process.stdin=None
            _,error=process.communicate(timeout=60)
            if process.returncode or not target.exists():raise RuntimeError('H264 export failed')
            from .clip_metadata import probe_clip
            metadata=probe_clip(target,data_dir=db.DATA) or {}
            path=f'runs/{self.id}/clips/{target.name}'
            patch={'clip_state':'ready','clip_start_s':round(start,3),
                'clip_end_s':round(end,3),'clip_truncated_after':end<clip['until']-.2,
                'clip_truncated_before':start>max(0,clip['alert_s']-PRE_SECONDS)+.2,
                'clip_alert_offset_s':round(clip['alert_s']-start,3),**metadata}
            publish(patch,path)
        except Exception:
            logging.exception('Clip export failed: %s',event_id)
            publish({'clip_state':'failed'})
        finally:
            if process is not None and process.poll() is None:
                process.kill();process.communicate()

    def queue_finished_clips(self,now,force=False):
        with self.frame_lock:
            remaining=[]
            for clip in self.pending:
                if force or now>=clip['until']:
                    if 'group' not in clip or self.clip_groups.reserve(clip['group'].id):
                        self.clip_futures.append(CLIPS.submit(self.finalize_clip,clip))
                else:remaining.append(clip)
            self.pending=remaining
            self.clip_futures=[f for f in self.clip_futures if not f.done()]

    def run(self):
        rules=scenarios=None;separated=False;now=0.;rule_time=scenario_time=0.
        try:
            renderer=threading.Thread(target=self.render_frames,name='display-'+self.id[:6],daemon=True)
            renderer.start();self.helpers.append(renderer)
            recorder=threading.Thread(target=self.encode_recording,name='record-'+self.id[:6],daemon=True)
            recorder.start();self.helpers.append(recorder)
            if self.source_type=='network':
                threading.Thread(target=self.read_network,name='camera-'+self.id[:6],daemon=True).start()
            self.backend=create_backend(MODEL,self.config.confidence,self.config.fall_detection,config=self.config)
            self.backend_name=self.backend.name
            tracker=PeopleTracker(SAMPLE_FPS,self.config.confidence,max_gap=3.)
            # LIVE sampling follows actual inference cadence, not the requested
            # 10 FPS ceiling. Three source-timed measurements are still required;
            # projected/predicted boxes never become evidence for this allowance.
            cfg=self.config.model_copy(update={'observation_gap':self.effective_observation_gap_s})
            rules=Rules(base_config(cfg))
            staged=callable(getattr(self.backend,'predict_people',None)) and callable(getattr(self.backend,'predict_scenarios',None))
            # One CUDA device processes the same source frame end-to-end.
            # Two workers previously interleaved many model stages while PPE
            # waited behind people inference, so results arrived seconds old.
            # Camera display/recording remain independent in either mode.
            split_default='0' if str(getattr(self.backend,'device','')).startswith('cuda') else '1'
            separated=staged and os.getenv('SAFETY_LIVE_SPLIT_INFERENCE',split_default)=='1'
            self.pipeline_mode='split' if separated else 'serial'
            scenarios=ScenarioRules(cfg)
            if separated:
                helper=threading.Thread(target=self.analyze_scenarios,name='objects-'+self.id[:6],daemon=True)
                helper.start();self.helpers.append(helper)
            while not self.stop_event.is_set():
                try:captured,frame=self.frames.get(timeout=.25)
                except queue.Empty:
                    if time.monotonic()-self.last_ingest>20:
                        self.error='Нет кадров больше 20 секунд. Подключите источник заново.';break
                    continue
                if self.last_processed is not None and captured-self.last_processed<1/SAMPLE_FPS-.005:
                    self.dropped+=1;continue
                self.last_processed=captured;now=captured-self.first_mono;h,w=frame.shape[:2]
                began=time.perf_counter()
                people=self.backend.predict_people(frame) if staged else self.backend.predict(frame)
                if callable(getattr(self.backend,'predict_temporal',None)):
                    self.backend.predict_temporal(frame,now)
                self.latencies.append((time.perf_counter()-began)*1000)
                self.latencies=self.latencies[-300:]
                people=tracker.update(people,frame.shape,now)
                scene=dict(getattr(self.backend,'scene_state',None) or {}) or None
                events,details=rules.step(people,now,w,h,scene)
                rule_time=now
                updates=list(getattr(rules,'event_updates',[]))
                with self.frame_lock:
                    self.people_states=state_snapshot(rules);self.states_updated_s=max(self.states_updated_s,now)
                scene_alerts=[key[1] for key,gate in rules.gates.items() if key[0]==0 and gate.active]
                if separated:
                    # Clone mutable arrays/details before passing ownership to
                    # another worker. Never reuse cached inference as evidence.
                    copied=[replace(p,keypoints=p.keypoints.copy() if p.keypoints is not None else None) for p in people]
                    copied_details={tid:{**d,'active':list(d.get('active',[]))} for tid,d in details.items()}
                    latest_only(self.scenario_frames,(now,frame,copied,copied_details))
                else:
                    scenario_began=time.perf_counter()
                    objects=self.backend.predict_scenarios(frame,people) if staged else getattr(self.backend,'objects_state',[])
                    self.scenario_latencies.append((time.perf_counter()-scenario_began)*1000)
                    self.scenario_latencies=self.scenario_latencies[-300:]
                    extra,active=scenarios.step(people,objects,now,w,h,details)
                    scenario_time=now
                    updates.extend(getattr(scenarios,'event_updates',[]))
                    with self.frame_lock:self.scenario_states=state_snapshot(scenarios)
                    events.extend(extra);self.assessments=assessment_rows(people,scenarios.ppe_details)
                    for tid,kinds in active.items():
                        if tid in details:details[tid]['active'].extend(kinds)
                    self.scenario_visual={'time':now,'objects':objects,'details':scenarios.ppe_details,'status':scenarios.ppe_status,'states':state_snapshot(scenarios)}
                    self.scenario_count+=1;self.scenario_times.append(time.monotonic())
                    self.scenario_age_ms=round((time.monotonic()-captured)*1000,1)
                preview_scale=min(1,960/max(h,w))
                small=cv2.resize(frame,(round(w*preview_scale),round(h*preview_scale))) if preview_scale<1 else frame
                self.visual={'time':now,'people':people,'details':details,'scene':scene,'scene_alerts':scene_alerts,
                             'gray':cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),'width':w}
                if events or updates:
                    annotated=draw(frame,people,details,self.config,now,scene,scene_alerts)
                    if not separated:annotated=annotate_objects(annotated,objects,scenarios.ppe_status,people,scenarios.ppe_details,scenarios.state_rows)
                    self.record_events(events,annotated,now)
                    persist_episode_updates(self.id,updates,annotated)
                if self.state=='starting':
                    self.state='running';db.update_job(self.id,status='running')
                self.processed+=1;self.analysis_times.append(time.monotonic())
                self.analysis_age_ms=round((time.monotonic()-captured)*1000,1)
        except Exception as exc:
            logging.exception('Live inference failed: %s',self.id)
            self.error='Ошибка прямого эфира: '+type(exc).__name__
        finally:
            self.state='stopping';self.stop_event.set();self.url=None
            reason='processing_failed' if self.error else 'stream_stopped'
            persist_episode_updates(self.id,finish_rules(rules,rule_time,reason)+([] if separated else finish_rules(scenarios,scenario_time,reason)))
            with self.frame_lock:
                self.people_states=state_snapshot(rules)
                if not separated:self.scenario_states=state_snapshot(scenarios)
            for helper in self.helpers:helper.join(timeout=30)
            self.queue_finished_clips(self.elapsed,force=True)
            for future in list(self.clip_futures):
                try:future.result(timeout=30)
                except Exception:pass
            self.state='failed' if self.error else 'stopped'
            self.stopped_at=datetime.now(timezone.utc).isoformat()
            stats=self.report();stats['backend']=getattr(self.backend,'name','unknown')
            db.update_job(self.id,status='failed' if self.error else 'completed',stats=stats,progress=1,error=self.error)
            self.backend=None;self.ring.clear()
            for pending_frames in (self.frames,self.display_frames,self.record_frames,self.scenario_frames):
                while True:
                    try:pending_frames.get_nowait()
                    except queue.Empty:break
            # CUDA's unused allocator cache otherwise stays reserved by this
            # server after Stop and competes with other applications/processes.
            gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()



def create(config,source_type,url):
    with LOCK:
        scope=current_scope.get();organization_id=scope.organization_id if scope else 'legacy'
        camera=config.camera.strip()
        if scope and scope.cameras is not None and camera not in scope.cameras:
            raise PermissionError('Эта камера не назначена вашему аккаунту')
        if not camera:raise ValueError('Укажите название камеры')
        config=config.model_copy(update={'camera':camera})
        active=[s for s in SESSIONS.values() if s.state in ACTIVE_STATES]
        if any(s.organization_id==organization_id and s.config.camera.casefold()==camera.casefold() for s in active):
            raise ValueError('Эта камера уже в эфире. Откройте существующую карточку камеры.')
        if source_type=='network' and any(s.organization_id==organization_id and s.url==url for s in active):
            raise ValueError('Этот источник уже подключён. Откройте существующую карточку камеры.')
        if len(active)>=MAX_LIVE_SESSIONS:
            raise ValueError('Локальный стенд поддерживает до 2 одновременных эфиров из-за ресурсов оборудования. Сначала остановите один эфир.')
        for old in list(SESSIONS):
            if len(SESSIONS)<10:break
            if SESSIONS[old].state in ('stopped','failed'):del SESSIONS[old]
        session=LiveSession(config,source_type,url);SESSIONS[session.id]=session
        try:return session.start()
        except BaseException:
            SESSIONS.pop(session.id,None);session.stop();raise
