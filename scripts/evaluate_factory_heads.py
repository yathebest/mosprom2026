"""Full-frame proposal/classifier check on explicitly reviewed factory heads.

This is conditional on reviewed visible crops, not exhaustive person recall or
event accuracy. Source video groups and exclusions remain in the output.
"""
import json
from pathlib import Path
import sys
from collections import Counter,defaultdict
import time

import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core import RunConfig
from app.inference import create_backend
from app.video import MODEL
from app.scenarios import ScenarioRules
from app.scenario_models import iou
import app.scenario_models as scenarios


def main():
    root=ROOT/'data/datasets/factory-review'
    reviewed=json.loads((root/'reviewed-head-crops.json').read_text())
    groups=defaultdict(list)
    for row in reviewed['rows']:
        if row['label']=='no_hardhat':groups[row['frame_file']].append(row)
    cfg=RunConfig(helmet_detection=True,fast_motion=False,confidence=.35)
    backend=create_backend(MODEL,.35,config=cfg)
    original=scenarios.pose_head_proposal;result=[]
    output=ROOT/'data/cache/factory-adaptation/full-frame-evaluation.json'
    for filename,rows in groups.items():
        cap=cv2.VideoCapture(str(ROOT/'data/datasets/factory-expansion'/rows[0]['source_video']))
        cap.set(cv2.CAP_PROP_POS_FRAMES,rows[0]['frame_index']);ok,frame=cap.read();cap.release()
        if not ok:raise RuntimeError('Cannot decode original source frame: '+filename)
        began=time.perf_counter();people=backend.predict_people(frame);people_ms=(time.perf_counter()-began)*1000
        people=[p for p in people if p.confidence>=cfg.confidence]
        for i,p in enumerate(people):p.track_id=i+1
        checks={}
        for enabled in [False,True]:
            scenarios.pose_head_proposal=original if enabled else lambda *args:None
            start=time.perf_counter();objects=backend.predict_scenarios(frame,people)
            rules=ScenarioRules(cfg);rules.step(people,objects,0,frame.shape[1],frame.shape[0])
            checks['pose_fallback' if enabled else 'detector_only']={'latency_ms':(time.perf_counter()-start)*1000,
                'assessments':{p.track_id:rules.ppe_details.get(p.track_id,{}).get('helmet',{}) for p in people}}
        scenarios.pose_head_proposal=original
        for row in rows:
            match=max(people,key=lambda p:iou(p.box,row['person_box']),default=None)
            matched=match is not None and iou(match.box,row['person_box'])>=.5
            record={'id':row['id'],'source_video':row['source_video'],'source_split':row['source_split'],
                'frame_file':filename,'truth':'missing','person_matched':matched}
            record['people_latency_ms']=people_ms
            for name,check in checks.items():
                record[name]=check['assessments'].get(match.track_id,{'state':'person_missing'}) if matched else {'state':'person_missing'}
            result.append(record)
        counts={name:dict(Counter(r[name].get('state','unknown') for r in result)) for name in checks}
        report={'annotation_scope':reviewed['method'],'rows':result,'counts':counts,'backend':backend.name,
            'warning':'Reviewed head subset; not exhaustive person recall, not event metrics. Test source crops have been inspected; do not use as unseen deployment evidence.'}
        output.write_text(json.dumps(report,indent=2))
        print('FULL_FRAME',filename,'reviewed',len(result),json.dumps(counts),flush=True)


if __name__=='__main__':main()
