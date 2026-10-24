"""Run the production fall/fire rules on source-grouped iSafety clips.

Author clip-level action labels are not precise event-onset annotations.
"""
from pathlib import Path
import json
import sys
import time
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core import RunConfig,Rules
from app.inference import create_backend,PeopleTracker
from app.video import MODEL
from app.scenarios import base_config,ScenarioRules


def main():
    root=ROOT/'data/datasets/isafety-inspection';manifest=json.loads((root/'manifest.json').read_text())
    cfg=RunConfig(fall_detection=True,fire_detection=True,fast_motion=False)
    backend=create_backend(MODEL,cfg.confidence,True,config=cfg);rows=[]
    for source in manifest['files']:
        temporal=getattr(backend,'temporal_fall',None)
        if temporal:temporal.frames.clear();temporal.last_sample=None
        cap=cv2.VideoCapture(str(root/source['file']));fps=cap.get(cv2.CAP_PROP_FPS)
        tracker=PeopleTracker(8,cfg.confidence);rules=Rules(base_config(cfg));scenarios=ScenarioRules(cfg)
        index=0;next_time=0.;events=[];frames=0;probabilities=[];start=time.perf_counter()
        while True:
            ok,frame=cap.read()
            if not ok:break
            now=index/fps;index+=1
            if now<next_time:continue
            next_time=now+.125-1e-6
            people=backend.predict(frame);backend.predict_temporal(frame,now)
            people=tracker.update(people,frame.shape,now)
            observed,details=rules.step(people,now,frame.shape[1],frame.shape[0],backend.scene_state)
            extra,_=scenarios.step(people,backend.objects_state,now,frame.shape[1],frame.shape[0],details)
            events+=observed+extra;frames+=1
            probabilities.append(dict(backend.scene_state or {}))
        cap.release();actions=set(source['gt_actions'])
        record={'file':source['file'],'source_group':source['source_group'],'caption':source['caption'],
            'declared_actions':source['gt_actions'],'fall_label':bool(actions&{'person falling down','falling down'}),
            'fire_label':'fire incident' in actions,'fall_alert':any(e['kind'] in {'possible_fall','person_down'} for e in events),
            'fire_alert':any(e['kind']=='fire' for e in events),'events':events,'processed_frames':frames,
            'wall_seconds':round(time.perf_counter()-start,2),'scene_scores':probabilities}
        rows.append(record)
        report={'source':manifest['source'],'revision':manifest['revision'],'license':manifest['license'],
            'config':cfg.model_dump(),'sample_fps':8,'rows':rows,
            'warning':'Author clip-level actions, not complete event intervals. Human onset/visibility review required. Mixed industrial/construction/workplace video; not all fixed factory CCTV.'}
        (ROOT/'docs/isafety-clip-debug.json').write_text(json.dumps(report,indent=2))
        print('ISAFETY_EVAL',source['file'],'fall',record['fall_label'],record['fall_alert'],'fire',record['fire_label'],record['fire_alert'],'frames',frames,flush=True)


if __name__=='__main__':main()
