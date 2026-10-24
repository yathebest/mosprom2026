"""Check rotated pose-only fall rules without modifying production decisions."""
import json
import sys
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core import RunConfig,Rules
from app.inference import create_backend,PeopleTracker
from app.video import MODEL


def main():
    root=ROOT/'data/datasets/isafety-inspection';source=json.loads((root/'manifest.json').read_text())
    cfg=RunConfig(fall_detection=True,fast_motion=False)
    backend=create_backend(MODEL,cfg.confidence,True,config=cfg);backend.rotated_search=True;rows=[]
    for clip in source['files']:
        if backend.temporal_fall:
            backend.temporal_fall.frames.clear();backend.temporal_fall.last_sample=None
        tracker=PeopleTracker(8,cfg.confidence);pose_rules=Rules(cfg);rgb_rules=Rules(cfg)
        cap=cv2.VideoCapture(str(root/clip['file']));fps=cap.get(5);index=0;due=0;events={'pose':[],'rgb':[]};states=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            now=index/fps;index+=1
            if now+1e-6<due:continue
            due=now+.125;people=backend.predict(frame);backend.predict_temporal(frame,now)
            people=tracker.update(people,frame.shape,now)
            measured,details=pose_rules.step(people,now,frame.shape[1],frame.shape[0],None)
            events['pose'].extend(measured)
            rgb,_=rgb_rules.step(people,now,frame.shape[1],frame.shape[0],backend.scene_state);events['rgb'].extend(rgb)
            states.append({'time':now,'people':[{'id':p.track_id,'box':p.box,'confidence':p.confidence,
                'observed':p.observed,'state':details.get(p.track_id)} for p in people]})
        cap.release()
        row={'file':clip['file'],'actions':clip['gt_actions'],'fall_label':bool(set(clip['gt_actions'])&{'person falling down','falling down'}),
            'events':events,'states':states};rows.append(row)
        (ROOT/'docs/fall-rotated-pose-ablation.json').write_text(json.dumps({'config':cfg.model_dump(),'rows':rows,
            'warning':'Debug candidate only. Clip labels do not establish correct onset; no production fall rules changed.'},indent=2))
        print('FALL_ROTATED',clip['file'],row['fall_label'],{k:[(e['kind'],e['alert_s']) for e in v] for k,v in events.items()},flush=True)


if __name__=='__main__':main()
