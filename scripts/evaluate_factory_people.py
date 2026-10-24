"""Small visually reviewed person ablation; not factory-wide person mAP."""
import json
import sys
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core import RunConfig
from app.inference import create_backend
from app.video import MODEL
from app.scenario_models import iou


def main():
    review=json.loads((ROOT/'docs/factory-person-review.json').read_text())
    cfg=RunConfig(confidence=.4,fast_motion=False)
    backend=create_backend(MODEL,cfg.confidence,config=cfg)
    ensemble=backend.person_models[:];results={}
    for mode in ['adapted_only_640','ensemble_640','ensemble_native_tiles']:
        backend.person_models=ensemble[:1] if mode=='adapted_only_640' else ensemble
        backend.distant_people=mode=='ensemble_native_tiles';rows=[]
        for sample in review['rows']:
            path=ROOT/'data/datasets/factory-expansion/test'/(sample['clip']+'.mp4')
            cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,sample['frame']);ok,frame=cap.read();cap.release()
            if not ok:raise ValueError(str(path))
            truth=[tuple(v*2 for v in box) for box in sample['boxes']]
            people=[p for p in backend.predict_people(frame) if p.confidence>=cfg.confidence]
            stats={}
            for threshold in [.3,.5]:
                used=set();tp=0
                for p in sorted(people,key=lambda p:p.confidence,reverse=True):
                    overlap,index=max(((iou(p.box,box),i) for i,box in enumerate(truth) if i not in used),default=(0,None))
                    if overlap>=threshold:used.add(index);tp+=1
                stats[str(threshold)]={'matched':tp,'unmatched_predictions':len(people)-tp,'missed_reviewed':len(truth)-tp}
            rows.append({**sample,'predictions':[{'box':p.box,'score':p.confidence} for p in people],'overlap':stats})
        results[mode]=rows
    report={'source':review['source'],'review':review,'config':cfg.model_dump(),'models':results,
        'warning':'Coarse assistant-reviewed boxes, eight debug frames, not expert benchmark or a blind test. Unmatched predictions need manual review before being called false positives. Adapted-only is an ablation; COCO ensemble already existed before this upgrade.'}
    (ROOT/'docs/factory-person-evaluation.json').write_text(json.dumps(report,indent=2))
    for mode,rows in results.items():
        print(mode,{t:{k:sum(row['overlap'][t][k] for row in rows) for k in ['matched','unmatched_predictions','missed_reviewed']} for t in ['0.3','0.5']},flush=True)


if __name__=='__main__':main()
