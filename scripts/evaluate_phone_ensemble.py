"""Validate supplemental COCO phone detections on FPI-Det image boxes.

This compares full-frame objects, not temporal phone-use events. Author split
and possible video/source overlap need separate audit before training claims.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
from fpi_labels import read_labels

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.scenario_models import OnnxDetector,iou


def score(boxes,truth):
    used=set();tp=0
    for box,confidence in sorted(boxes,key=lambda x:x[1],reverse=True):
        overlap,index=max(((iou(box,t),i) for i,t in enumerate(truth) if i not in used),default=(0,None))
        if overlap>=.5:used.add(index);tp+=1
    return tp,len(boxes)-tp,len(truth)-tp


def merge(first,extra):
    output=list(first)
    for candidate in extra:
        if not any(iou(candidate[0],box)>.45 for box,_ in output):output.append(candidate)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=600);p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(1)
    root=ROOT/'data/datasets/fpi-det/reorganized_dataset'/a.split
    files=sorted((root/'images').glob('*.jpg'));random.Random(42).shuffle(files);files=files[:a.count]
    baseline=OnnxDetector(ROOT/'data/models/phone-objects.onnx',.3)
    coco=YOLO(str(ROOT/'data/models/yolo12m.pt'))
    thresholds=[.3,.5,.7,.85,.95];totals={name:{'tp':0,'fp':0,'fn':0} for name in ['baseline']+[str(v) for v in thresholds]}
    rows=[]
    for index,file in enumerate(files):
        frame=cv2.imread(str(file));h,w=frame.shape[:2];truth=[]
        for label,x,y,bw,bh in read_labels(root/'labels'/(file.stem+'.txt')):
            if int(label)==0:truth.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
        old=[(o.box,o.confidence) for o in baseline.predict(frame) if o.label=='cell_phone']
        r=coco.predict(frame,classes=[67],conf=.1,imgsz=960 if w>=1600 else 640,device=0,verbose=False)[0]
        extra=[(tuple(map(float,b[:4])),float(b[4])) for b in r.boxes.data.cpu().numpy()]
        predictions={'baseline':old,**{str(t):merge(old,[(box,conf) for box,conf in extra if conf>=t]) for t in thresholds}}
        for name,boxes in predictions.items():
            tp,fp,fn=score(boxes,truth)
            for k,v in zip(['tp','fp','fn'],[tp,fp,fn]):totals[name][k]+=v
        rows.append({'file':file.name,'truth_boxes':truth,'baseline':old,'supplemental':extra})
        if (index+1)%50==0:print('PHONE_EVAL',index+1,flush=True)
    for s in totals.values():
        s['precision']=s['tp']/max(1,s['tp']+s['fp']);s['recall']=s['tp']/max(1,s['tp']+s['fn'])
    old=totals['baseline'];choices=[t for t in thresholds if totals[str(t)]['precision']>=old['precision']-.005 and totals[str(t)]['recall']>old['recall']]
    chosen=max(choices,key=lambda t:totals[str(t)]['recall']) if choices else None
    report={'source':'https://github.com/KvCgRv/FPI-Det','split':a.split,'images':len(files),'confidence_baseline':.3,
        'iou':.5,'totals':totals,'chosen_auxiliary_threshold':chosen,'rows':rows,
        'scope':'Full-frame phone object detection only. Not phone-use event precision/recall. Source split and source/video leakage not yet audited.'}
    (ROOT/f'docs/phone-ensemble-{a.split}.json').write_text(json.dumps(report,indent=2))
    print('PHONE_RESULT',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
