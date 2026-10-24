"""Guard a cigarette candidate using validation before touching test/runtime."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.scenario_models import OnnxDetector,iou


def main():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(1)
    folder=ROOT/'data/cache/smoking-yolo12m';candidate=YOLO(str(folder/'training/weights/best.pt'))
    baseline=OnnxDetector(ROOT/'data/models/smoking.onnx',.5)
    root=folder/'dataset';results={}
    def evaluate(split):
        stats={name:{'tp':0,'fp':0,'fn':0,'latency_ms':[]} for name in ['baseline','candidate']}
        for file in sorted((root/'images'/split).glob('*.jpg')):
            frame=cv2.imread(str(file));h,w=frame.shape[:2];truth=[]
            for line in (root/'labels'/split/(file.stem+'.txt')).read_text().splitlines():
                if not line.strip():continue
                _,x,y,bw,bh=map(float,line.split());truth.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
            for name in stats:
                start=time.perf_counter()
                if name=='baseline':boxes=[o.box for o in baseline.predict(frame) if o.label in {'smooking','smoking','cigarette'}]
                else:
                    pred=candidate.predict(frame,conf=.5,imgsz=640,rect=False,device=0,verbose=False)[0]
                    boxes=[tuple(map(float,b[:4])) for b in pred.boxes.data.cpu().numpy()]
                stats[name]['latency_ms'].append((time.perf_counter()-start)*1000)
                used=set();tp=0
                for box in boxes:
                    overlap,index=max(((iou(box,t),i) for i,t in enumerate(truth) if i not in used),default=(0,None))
                    if overlap>=.5:used.add(index);tp+=1
                stats[name]['tp']+=tp;stats[name]['fp']+=len(boxes)-tp;stats[name]['fn']+=len(truth)-tp
        for s in stats.values():
            s['precision']=s['tp']/max(1,s['tp']+s['fp']);s['recall']=s['tp']/max(1,s['tp']+s['fn'])
            s['p95_ms']=float(np.percentile(s.pop('latency_ms'),95))
        return stats
    results['validation']=evaluate('val');old,new=results['validation']['baseline'],results['validation']['candidate']
    passed=(new['recall']>old['recall']+.01 and new['precision']>=max(.95,old['precision']-.005))
    results['validation_guard_passed']=passed
    results['selection_policy']='Recall must improve >0.01, precision >=0.95 and not regress >0.005, measured at conf=0.5/IoU=0.5. Test evaluated only after validation passes.'
    if passed:results['test']=evaluate('test')
    results['deployed']=False
    results['warning']='CigDet object boxes, not factory event metrics. Upstream baseline training overlap is unknown. Factory event controls and LinQ compilation still required.'
    (ROOT/'docs/smoking-candidate-comparison.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
