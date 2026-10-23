import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from datetime import datetime,timedelta,timezone

import cv2
import numpy as np

from . import storage as db
from .core import Rules, RunConfig
from .inference import PeopleTracker, create_backend
from . import streams
from .scenarios import ScenarioRules,base_config,annotate_objects,assessment_rows

MODEL = db.DATA / "models" / "yolov8n-pose.pt"
CANCEL: dict[str,threading.Event] = {}


def ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def probe(source: Path, folder: Path):
    cap = cv2.VideoCapture(str(source))
    try:
        ok,frame = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or not 1<=fps<=240 or count<1:
            raise ValueError("Не удалось прочитать видео или временную шкалу. Нужен MP4/AVI/MOV/WebM с постоянной частотой кадров.")
        h,w = frame.shape[:2]
        if w*h>4096*2160 or count/fps>3600:
            raise ValueError("Для прототипа: не более 4K и 60 минут на файл")
        cv2.imwrite(str(folder/"preview.jpg"),frame)
        return {"width":w,"height":h,"fps":fps,"frames":count,"duration_s":round(count/fps,3)}
    finally:
        cap.release()


def draw(frame, people, details, config, now, scene_state=None, scene_alerts=()):
    output = frame.copy()
    h,w = frame.shape[:2]
    overlay = output.copy()
    for idx,zone in enumerate(config.zones):
        pts = (np.asarray(zone.points)*[w,h]).astype(np.int32)
        cv2.fillPoly(overlay,[pts],(27,117,185))
        cv2.polylines(output,[pts],True,(30,190,245),2)
        cv2.putText(output,f"ZONE {idx+1}",tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,190,245),2)
    output = cv2.addWeighted(overlay,.18,output,.82,0)
    for p in people:
        d = details.get(p.track_id,{})
        color = (90,220,90) if not d.get("active") else (55,55,245)
        if not p.alert_eligible:color=(190,160,80)
        x1,y1,x2,y2 = map(int,p.box)
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        label = f"#{p.track_id} {d.get('speed',0):.2f} h/s {d.get('posture','')} {d.get('tracking','')}"
        cv2.putText(output,label,(max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        if p.keypoints is not None:
            for x,y,c in p.keypoints:
                if c>=.4:
                    cv2.circle(output,(int(x),int(y)),3,color,-1)
    cv2.putText(output,f"VIDEO {now:.1f}s",(14,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2)
    if scene_state is not None:
        probability=float(scene_state.get('lying_probability',0))
        label='ALERT: FALL' if 'possible_fall' in scene_alerts else 'ALERT: PERSON DOWN' if 'person_down' in scene_alerts else 'LYING CANDIDATE' if probability>=config.body_threshold else 'BODY: MONITORING'
        color=(70,70,250) if scene_alerts else (100,215,235)
        cv2.putText(output,label,(14,max(48,h-15)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        if scene_alerts:cv2.rectangle(output,(2,2),(w-3,h-3),color,3)
    return output


def transcode(source,target):
    executable = ffmpeg_path()
    if not executable:
        return False
    result = subprocess.run([executable,"-hide_banner","-loglevel","error","-y","-i",str(source),
                             "-an","-c:v","libx264","-preset","veryfast","-crf","24","-pix_fmt","yuv420p",
                             "-movflags","+faststart",str(target)],capture_output=True,timeout=600)
    return result.returncode==0 and target.exists()


def run(job_id: str, config: RunConfig, cancel: threading.Event):
    folder = db.DATA/"runs"/job_id
    source = folder/"source.video"
    cap,writer,source_writer = None,None,None
    try:
        if cancel.is_set():
            db.update_job(job_id,status="cancelled")
            return
        db.update_job(job_id,status="running",config=config.model_dump(),progress=0,error=None)
        setup_started = time.perf_counter()
        backend = create_backend(MODEL,config.confidence,config.fall_detection,config=config)
        meta = db.job(job_id)["meta"]
        live=meta.get('live',False)
        cap=streams.open_capture(streams.SOURCES[job_id]) if live else cv2.VideoCapture(str(source))
        fps,w,h,count = meta["fps"],meta["width"],meta["height"],meta["frames"]
        stride = max(1,round(fps/config.sample_fps))
        tracker = PeopleTracker(fps/stride,config.confidence)
        rules = Rules(base_config(config))
        scenario_rules=ScenarioRules(config)
        raw = folder/"annotated_raw.mp4"
        # Analyzed frames retain their original video time at fps/stride.
        writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"mp4v"),fps/stride,(w,h))
        if not writer.isOpened():
            raise RuntimeError("Видеокодек MP4V недоступен")
        if live:
            source=folder/'source_capture.mp4'
            source_writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            if not source_writer.isOpened():raise RuntimeError('Не удалось создать запись потока')
            if not config.recorded_at:config.recorded_at=datetime.now(timezone.utc).isoformat()
        ok,warm_frame = cap.read()
        if not ok:
            raise RuntimeError("Первый кадр не читается")
        backend.predict(warm_frame)
        if not live:cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        setup_seconds = time.perf_counter()-setup_started
        latencies,total_latencies = [],[]
        inference_events = []
        brightness=[];sharpness=[];density=np.zeros((18,32),np.int32);latest_assessments=[]
        processed,decoded,alerts = 0,0,0
        started = time.perf_counter()
        for index in range(count):
            if cancel.is_set():
                break
            ok,frame = cap.read()
            if not ok:
                break
            decoded += 1
            if source_writer is not None:source_writer.write(frame)
            if index%stride:
                continue
            frame_start = time.perf_counter()
            now = index/fps
            inference_start = time.perf_counter()
            people = backend.predict(frame)
            latencies.append((time.perf_counter()-inference_start)*1000)
            people = tracker.update(people,frame.shape,now)
            if processed%10==0:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean()));sharpness.append(float(cv2.Laplacian(gray,cv2.CV_64F).var()))
            for person in people:
                if not person.observed:continue
                x,y=person.foot
                density[min(17,max(0,int(y/h*18))),min(31,max(0,int(x/w*32)))]+=1
            events,details = rules.step(people,now,w,h,getattr(backend,"scene_state",None))
            observations=getattr(backend,'objects_state',[])
            extra_events,extra_active=scenario_rules.step(people,observations,now,w,h,details)
            latest_assessments=assessment_rows(people,scenario_rules.ppe_details)
            events.extend(extra_events)
            for tid,kinds in extra_active.items():
                if tid in details:details[tid]['active'].extend(kinds)
            scene_alerts=[key[1] for key,gate in rules.gates.items() if key[0]==0 and gate.active]
            annotated = draw(frame,people,details,config,now,getattr(backend,'scene_state',None),scene_alerts)
            annotated=annotate_objects(annotated,observations,scenario_rules.ppe_status,people,scenario_rules.ppe_details)
            writer.write(annotated)
            if processed%3==0:
                temp = folder/"latest.tmp.jpg"
                cv2.imwrite(str(temp),annotated)
                try:
                    temp.replace(folder/"latest.jpg")
                except PermissionError:
                    pass  # A Windows browser may briefly hold the prior JPEG open.
            for event in events:
                event.update(clip_state='waiting_for_video_end',clip_start_s=max(0,event['alert_s']-3),
                             clip_end_s=min(meta['duration_s'],event['alert_s']+4),
                             clip_alert_offset_s=min(3,event['alert_s']))
                event['priority']=next((z.priority for z in config.zones if z.name==event.get('zone')),None)
                if config.recorded_at:
                    event['occurred_at']=(datetime.fromisoformat(config.recorded_at)+timedelta(seconds=event['alert_s'])).isoformat()
                alerts += 1
                rel_image = f"runs/{job_id}/evidence_{alerts}.jpg"
                cv2.imwrite(str(db.DATA/rel_image),annotated)
                event_id = db.add_event(job_id,config.camera,event,rel_image)
                inference_events.append((event_id,event))
            processed += 1
            total_latencies.append((time.perf_counter()-frame_start)*1000)
            if processed%5==0:
                db.update_job(job_id,progress=round(decoded/count,4),stats={'latest_assessments':latest_assessments})
        wall = time.perf_counter()-started
        cap.release();cap=None
        if source_writer is not None:
            source_writer.release();source_writer=None
            source.replace(folder/'source.video');source=folder/'source.video'
        writer.release();writer=None
        if not processed:
            raise RuntimeError("Видео не содержит читаемых кадров")
        db.update_job(job_id,status="exporting",progress=round(decoded/count,4))
        media_wall_start = time.perf_counter()
        browser_video = folder/"annotated.mp4"
        converted = transcode(raw,browser_video)
        clip_failures = 0
        executable = ffmpeg_path()
        # Clip generation is separate from inference timing. Original video,
        # without overlays; the accompanying evidence image contains annotations.
        for event_id,event in inference_events:
            if cancel.is_set():
                db.patch_payload(event_id,{'clip_state':'cancelled'})
                continue
            clip = folder/f"clip_{event_id}.mp4"
            begin = max(0,event["alert_s"]-3)
            length = min(7,meta["duration_s"]-begin)
            if not executable:
                clip_failures += 1
                db.patch_payload(event_id,{'clip_state':'failed'})
                continue
            result = subprocess.run([executable,"-hide_banner","-loglevel","error","-y","-ss",str(begin),
                                     "-i",str(source),"-t",str(length),"-an","-c:v","libx264",
                                     "-preset","veryfast","-crf","26","-pix_fmt","yuv420p",
                                     "-movflags","+faststart",str(clip)],capture_output=True,timeout=90)
            if result.returncode==0:
                db.set_clip(event_id,f"runs/{job_id}/{clip.name}")
                db.patch_payload(event_id,{'clip_state':'ready','clip_start_s':begin,'clip_end_s':begin+length,
                                           'clip_alert_offset_s':round(event['alert_s']-begin,3)})
            else:
                clip_failures += 1
                db.patch_payload(event_id,{'clip_state':'failed'})
        truncated = decoded<count and not cancel.is_set()
        stats = {"backend":backend.name,"model":getattr(backend,"model_id","external-adapter"),"input_frames":count,
                 "latest_assessments":latest_assessments,
                 "decoded_frames":decoded,"processed_frames":processed,"stride":stride,
                 "sample_fps":fps/stride,"inference_p50_ms":round(float(np.median(latencies)),2),
                 "live_source":live,"source_timing":"CFR frame_index / declared_fps",
                 "model_setup_and_warmup_s":round(setup_seconds,3),
                 "inference_p95_ms":round(float(np.percentile(latencies,95)),2),
                 "pipeline_p95_ms":round(float(np.percentile(total_latencies,95)),2),
                 "pipeline_wall_s":round(wall,3),"processing_fps":round(processed/max(wall,1e-9),2),
                 "media_export_s":round(time.perf_counter()-media_wall_start,3),
                 "events":alerts,"clip_failures":clip_failures,"truncated":truncated,
                 "annotated_video":f"runs/{job_id}/annotated.mp4" if converted else None,
                 "raw_video":f"runs/{job_id}/annotated_raw.mp4",
                 "occupancy_grid":density.tolist(),
                 "signal_checks":{"mean_brightness":round(float(np.mean(brightness)),2),
                     "median_laplacian_variance":round(float(np.median(sharpness)),2),
                     "dark_frame_fraction":round(float(np.mean(np.asarray(brightness)<30)),3),
                     "note":"Heuristic diagnostics only; not a validated camera-quality classifier or recognition metric."},
                 "scope":"Uploaded-file benchmark after one unmeasured warmup frame. Includes decode, tracking, rules, JPEG evidence and MP4V writing. Model setup/warmup and H.264/clip export are reported separately. Not LinQ measurements."}
        (folder/"metrics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
        (folder/"config.json").write_text(config.model_dump_json(indent=2),encoding="utf-8")
        (folder/"events.json").write_text(json.dumps(db.events(job_id),ensure_ascii=False,indent=2),encoding="utf-8")
        db.update_job(job_id,status="cancelled" if cancel.is_set() else "completed",
                      progress=round(decoded/count,4),stats=stats,
                      error="Чтение завершилось раньше заявленного конца файла" if truncated else None)
    except Exception as exc:
        db.update_job(job_id,status="failed",error=f"{type(exc).__name__}: {str(exc)[:600]}")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if source_writer is not None:source_writer.release()
        with streams.SOURCE_LOCK:streams.SOURCES.pop(job_id,None)
        CANCEL.pop(job_id,None)
