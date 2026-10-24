"""Fixed clip-level fire comparison. Clip action labels are not box/onset GT."""
import json
import sys
import time
import argparse
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.scenario_models import OnnxDetector
from app.core import RunConfig
from app.scenarios import ScenarioRules


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/datasets/isafety-inspection')
    p.add_argument('--output',type=Path,default=ROOT/'docs/fire-candidate-clip-comparison.json');a=p.parse_args()
    root=a.data
    source=json.loads((root/'manifest.json').read_text())
    cfg=RunConfig(fire_detection=True,smoke_detection=True,fast_motion=False)
    report={'source':source['source'],'revision':source['revision'],'license':source['license'],
        'config':cfg.model_dump(),'sample_fps':5,'models':{},
        'warning':'Debug clips, author action labels only. Fire incident can mean smoke/sparks without visible flame. Not event precision or a blind independent test. Candidates remain undeployed.'}
    for name,path in [('baseline',ROOT/'data/models/fire-smoke.onnx'),
                      ('dfire_yolov5s',ROOT/'data/cache/dfire-models/yolov5s.onnx')]:
        model=OnnxDetector(path,.5);rows=[]
        for clip in source['files']:
            cap=cv2.VideoCapture(str(root/clip['file']));fps=cap.get(cv2.CAP_PROP_FPS)
            rules=ScenarioRules(cfg);index=0;due=0.;events=[];seen={'fire':0,'smoke':0};frames=0;start=time.monotonic()
            while True:
                ok,frame=cap.read()
                if not ok:break
                now=index/fps;index+=1
                if now+1e-6<due:continue
                due=now+.2;objects=model.predict(frame);frames+=1
                for kind in seen:seen[kind]+=any(o.label==kind for o in objects)
                extra,_=rules.step([],objects,now,frame.shape[1],frame.shape[0],{})
                events.extend(extra)
            cap.release()
            row={'file':clip['file'],'source_group':clip['source_group'],
                'declared_fire':'fire incident' in clip['gt_actions'],'declared_actions':clip['gt_actions'],
                'fire_alert':any(e['kind']=='fire' for e in events),'smoke_alert':any(e['kind']=='smoke' for e in events),
                'events':events,'frames':frames,'frames_with_objects':seen,'seconds':round(time.monotonic()-start,2)}
            rows.append(row);report['models'][name]=rows
            a.output.write_text(json.dumps(report,indent=2))
            print('FIRE_COMPARE',name,clip['file'],row['declared_fire'],row['fire_alert'],row['smoke_alert'],flush=True)


if __name__=='__main__':main()
