"""Smoke-detector stress test on CigDet boxes, not factory-event metrics."""
import json
from pathlib import Path
import sys
import time
import argparse
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.scenario_models import OnnxDetector,iou


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,default=ROOT/'data/models/smoking.onnx')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/cigarette-object-stress.json');args=parser.parse_args()
    root=ROOT/'data/datasets/cigdet/CigDet_dataset/test'
    model=OnnxDetector(args.model,.25)
    totals={scale:{'tp':0,'fp':0,'fn':0,'images':0,'latency_ms':[]} for scale in [1.,.5,.25]}
    rows=[]
    for file in sorted(root.glob('*.jpg')):
        frame=cv2.imread(str(file));h,w=frame.shape[:2];truth=[]
        for line in file.with_suffix('.txt').read_text().splitlines():
            if not line.strip():continue
            _,x,y,bw,bh=map(float,line.split())
            truth.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
        for scale,stats in totals.items():
            image=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale)))) if scale!=1 else frame
            start=time.perf_counter();objects=model.predict(image);elapsed=(time.perf_counter()-start)*1000
            predictions=[o for o in objects if o.label in {'smooking','smoking','cigarette'} and o.confidence>=.5]
            used=set();matched=0
            for o in predictions:
                overlaps=[(iou(tuple(v/scale for v in o.box),box),i) for i,box in enumerate(truth) if i not in used]
                overlap,index=max(overlaps,default=(0,None))
                if overlap>=.5:used.add(index);matched+=1
            stats['tp']+=matched;stats['fp']+=len(predictions)-matched;stats['fn']+=len(truth)-matched
            stats['images']+=1;stats['latency_ms'].append(elapsed)
            rows.append({'image':file.name,'scale':scale,'truth_boxes':len(truth),'matched':matched,'proposals':len(predictions)})
    import numpy as np
    for stats in totals.values():
        stats['precision']=stats['tp']/max(1,stats['tp']+stats['fp'])
        stats['recall']=stats['tp']/max(1,stats['tp']+stats['fn'])
        stats['p95_ms']=float(np.percentile(stats.pop('latency_ms'),95))
    report={'source':'https://data.mendeley.com/datasets/6hyrr8typ7/1','author':'Ali Khan','license':'CC-BY-4.0',
        'model':str(args.model.relative_to(ROOT) if args.model.is_absolute() and args.model.is_relative_to(ROOT) else args.model),'confidence':.5,'iou':.5,'totals':totals,'rows':rows,
        'limitations':['Web-image object boxes, not industrial smoking events.',
            'Overlap with upstream model training is unknown: not an independent generalization score.',
            'Downscaled images are synthetic resolution stress tests, not camera footage.',
            'Person association and temporal alert confirmation are not evaluated here.']}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
