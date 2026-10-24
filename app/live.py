"""Live inference with latest-frame buffering and independent incident clips."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from . import storage as db
from .core import Rules
from .inference import create_backend,PeopleTracker
from .streams import open_capture
from .video import MODEL,draw,transcode
from .scenarios import ScenarioRules,base_config,annotate_objects,assessment_rows

SESSIONS={}
LOCK=threading.Lock()
CLIPS=ThreadPoolExecutor(max_workers=2,thread_name_prefix='live-clip')
PRE_SECONDS=3.
POST_SECONDS=4.
SAMPLE_FPS=10.


class LiveSession:
    def __init__(self,config,source_type,url=None):
        self.id=uuid.uuid4().hex;self.config=config;self.source_type=source_type;self.url=url
        self.folder=db.DATA/'runs'/self.id;self.folder.mkdir(parents=True)
        (self.folder/'clips').mkdir()
        self.stop_event=threading.Event();self.frames=queue.Queue(maxsize=1)
        self.state='starting';self.error=None;self.jpeg=None;self.version=0
        self.started=datetime.now(timezone.utc);self.first_mono=None;self.frame_origin=None
        self.last_ingest=time.monotonic();self.last_processed=None
        self.processed=0;self.dropped=0;self.elapsed=0.;self.events=0;self.pending=[]
        self.ring=deque();self.latencies=[];self.clip_futures=[];self.backend=None
        self.created=time.monotonic()
        self.assessments=[]
        db.add_job(self.id,'Прямой эфир · '+config.camera,
                   {'live':True,'mode':'live','source_type':source_type,'duration_s':0,
                    'fps':SAMPLE_FPS,'width':0,'height':0,'frames':0})
        db.update_job(self.id,status='queued',config=config.model_dump())

    def start(self):
        self.worker=threading.Thread(target=self.run,name='live-'+self.id[:6],daemon=True)
        self.worker.start()
        return self

    def ingest(self,frame):
        if self.stop_event.is_set():return
        h,w=frame.shape[:2]
        scale=min(1,1280/max(h,w))
        if scale<1:frame=cv2.resize(frame,(round(w*scale/2)*2,round(h*scale/2)*2))
        captured=time.monotonic();self.last_ingest=captured
        try:self.frames.put_nowait((captured,frame))
        except queue.Full:
            try:self.frames.get_nowait();self.dropped+=1
            except queue.Empty:pass
            try:self.frames.put_nowait((captured,frame))
            except queue.Full:self.dropped+=1

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
        return {'id':self.id,'camera':self.config.camera,'state':self.state,'error':self.error,
                'config':self.config.model_dump(),'job_id':self.id,
                'source_type':self.source_type,'started_at':self.started.isoformat(),
                'elapsed_s':round(self.elapsed,2),'processed_frames':self.processed,
                'dropped_frames':self.dropped,'events':self.events,
                'clips_pending':len(self.pending)+sum(not f.done() for f in self.clip_futures),
                'inference_p95_ms':round(float(np.percentile(self.latencies,95)),2) if self.latencies else None,
                'view_url':f'/api/live/{self.id}/view','clip_pre_s':PRE_SECONDS,'clip_post_s':POST_SECONDS,
                'timing':'Server frame-arrival time; network/camera latency is not measured',
                'model':getattr(self.backend,'name',None),'assessments':self.assessments}

    def finalize_clip(self,clip):
        event_id=clip['id'];frames=clip['frames']
        if not frames:
            db.patch_payload(event_id,{'clip_state':'failed'});return
        db.patch_payload(event_id,{'clip_state':'exporting'})
        raw=self.folder/'clips'/f'{event_id}-raw.mp4'
        target=self.folder/'clips'/f'{event_id}.mp4'
        writer=None
        try:
            start,end=frames[0][0],frames[-1][0]
            first=cv2.imdecode(np.frombuffer(frames[0][1],np.uint8),cv2.IMREAD_COLOR)
            h,w=first.shape[:2]
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),SAMPLE_FPS,(w,h))
            if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
            index=0;decoded=first
            for t in np.arange(start,end+.0001,1/SAMPLE_FPS):
                while index+1<len(frames) and frames[index+1][0]<=t:
                    index+=1;decoded=cv2.imdecode(np.frombuffer(frames[index][1],np.uint8),cv2.IMREAD_COLOR)
                writer.write(decoded)
            writer.release();writer=None
            if not transcode(raw,target):raise RuntimeError('H264 export failed')
            db.set_clip(event_id,f'runs/{self.id}/clips/{target.name}')
            db.patch_payload(event_id,{'clip_state':'ready','clip_start_s':round(start,3),
                'clip_end_s':round(end,3),'clip_truncated_after':end<clip['until']-.2,
                'clip_alert_offset_s':round(clip['alert_s']-start,3)})
            raw.unlink(missing_ok=True)
        except Exception:
            db.patch_payload(event_id,{'clip_state':'failed'})
        finally:
            if writer is not None:writer.release()

    def queue_finished_clips(self,now,force=False):
        remaining=[]
        for clip in self.pending:
            if force or now>=clip['until']:
                self.clip_futures.append(CLIPS.submit(self.finalize_clip,clip))
            else:remaining.append(clip)
        self.pending=remaining
        self.clip_futures=[f for f in self.clip_futures if not f.done()]

    def run(self):
        try:
            self.backend=create_backend(MODEL,self.config.confidence,self.config.fall_detection,config=self.config)
            tracker=PeopleTracker(SAMPLE_FPS,self.config.confidence);rules=Rules(base_config(self.config));scenario_rules=ScenarioRules(self.config)
            self.state='running';db.update_job(self.id,status='running')
            if self.source_type=='network':
                threading.Thread(target=self.read_network,name='camera-'+self.id[:6],daemon=True).start()
            while not self.stop_event.is_set():
                try:captured,frame=self.frames.get(timeout=.25)
                except queue.Empty:
                    if time.monotonic()-self.last_ingest>20:
                        self.error='Нет кадров больше 20 секунд. Подключите источник заново.';break
                    self.queue_finished_clips(self.elapsed)
                    continue
                if self.last_processed is not None and captured-self.last_processed<1/SAMPLE_FPS-.005:
                    self.dropped+=1;continue
                if self.first_mono is None:
                    self.first_mono=captured;self.frame_origin=datetime.now(timezone.utc)
                    h,w=frame.shape[:2]
                    cv2.imwrite(str(self.folder/'preview.jpg'),frame)
                    db.update_job(self.id,meta={'live':True,'mode':'live','source_type':self.source_type,
                        'started_at':self.frame_origin.isoformat(),'fps':SAMPLE_FPS,'width':w,'height':h,'frames':0,'duration_s':0})
                self.last_processed=captured;self.elapsed=captured-self.first_mono
                now=self.elapsed;h,w=frame.shape[:2]
                began=time.perf_counter();people=self.backend.predict(frame)
                if callable(getattr(self.backend,'predict_temporal',None)):
                    self.backend.predict_temporal(frame,now)
                self.latencies.append((time.perf_counter()-began)*1000)
                if len(self.latencies)>3000:self.latencies=self.latencies[-3000:]
                people=tracker.update(people,frame.shape,now)
                events,details=rules.step(people,now,w,h,getattr(self.backend,'scene_state',None))
                observations=getattr(self.backend,'objects_state',[])
                extra_events,extra_active=scenario_rules.step(people,observations,now,w,h,details)
                self.assessments=assessment_rows(people,scenario_rules.ppe_details)
                events.extend(extra_events)
                for tid,kinds in extra_active.items():
                    if tid in details:details[tid]['active'].extend(kinds)
                scene_alerts=[key[1] for key,gate in rules.gates.items() if key[0]==0 and gate.active]
                annotated=draw(frame,people,details,self.config,now,getattr(self.backend,'scene_state',None),scene_alerts)
                annotated=annotate_objects(annotated,observations,scenario_rules.ppe_status,people,scenario_rules.ppe_details)
                ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                original=encoded.tobytes()
                self.ring.append((now,original))
                while self.ring and now-self.ring[0][0]>PRE_SECONDS+1:self.ring.popleft()
                for clip in self.pending:clip['frames'].append((now,original))
                for event in events:
                    event.update(live=True,clip_state='capturing',
                        occurred_at=(self.frame_origin+timedelta(seconds=event['alert_s'])).isoformat(),
                        priority=next((z.priority for z in self.config.zones if z.name==event.get('zone')),None))
                    event_id=uuid.uuid4().hex
                    image=f'runs/{self.id}/evidence_{event_id}.jpg'
                    cv2.imwrite(str(db.DATA/image),annotated)
                    saved_id=db.add_event(self.id,self.config.camera,event,image)
                    self.pending.append({'id':saved_id,'frames':[(t,b) for t,b in self.ring if t>=now-PRE_SECONDS],
                                         'until':now+POST_SECONDS,'alert_s':event['alert_s']})
                    self.events+=1
                self.jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,85])[1].tobytes()
                self.version+=1;self.processed+=1
                if self.processed%10==0:
                    cv2.imwrite(str(self.folder/'latest.jpg'),annotated)
                self.queue_finished_clips(now)
        except Exception as exc:
            self.error='Ошибка прямого эфира: '+type(exc).__name__
        finally:
            self.state='stopping';self.stop_event.set();self.url=None
            self.queue_finished_clips(self.elapsed,force=True)
            for future in self.clip_futures:
                try:future.result(timeout=30)
                except Exception:pass
            self.state='failed' if self.error else 'stopped'
            stats=self.report()
            stats['backend']=getattr(self.backend,'name','unknown')
            db.update_job(self.id,status='failed' if self.error else 'completed',stats=stats,
                          progress=1,error=self.error)
            self.backend=None;self.ring.clear()


def create(config,source_type,url):
    with LOCK:
        if sum(s.state in ('starting','running','stopping') for s in SESSIONS.values())>=2:
            raise ValueError('Максимум 2 прямых эфира одновременно')
        for old in list(SESSIONS):
            if len(SESSIONS)<10:break
            if SESSIONS[old].state in ('stopped','failed'):del SESSIONS[old]
        session=LiveSession(config,source_type,url);SESSIONS[session.id]=session
        return session.start()
