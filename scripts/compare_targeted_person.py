"""Fixed post-training person comparison; does not install weights or tune on test."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__
from app.core import RunConfig
from app.inference import YoloBackend, PeopleTracker, semantic_person_evidence, validate_person_class
from app.scenario_models import iou

OUTPUT = ROOT / 'docs/targeted-person-comparison-20261025.json'
WEIGHTS = {
    'baseline': ROOT / 'data/models/person-yolo12m.pt',
    'candidate': ROOT / 'data/cache/targeted-completed-20261025/person/person/person/weights/best.pt',
    'stock': ROOT / 'data/models/yolo12m.pt',
}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def json_value(value):
    if isinstance(value,np.generic): return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')

def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False,
        default=json_value), encoding='utf-8')

def read_frame(path, index):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(str(path))
    return frame

def scored(boxes, truth):
    result = {}
    for threshold in (.3, .5):
        used = set(); matched = []
        for box, score in sorted(boxes, key=lambda row: row[1], reverse=True):
            overlap, index = max(((iou(box, target), n) for n, target in enumerate(truth) if n not in used), default=(0, None))
            if overlap >= threshold:
                used.add(index); matched.append(index)
        small = [n for n, b in enumerate(truth) if b[3]-b[1] < 150]
        result[str(threshold)] = {'matched': len(used), 'reviewed': len(truth),
            'unmatched_predictions': len(boxes)-len(used), 'missed_reviewed': len(truth)-len(used),
            'small_matched': len(set(small)&used), 'small_reviewed': len(small)}
    return result

def totals(rows):
    return {threshold: {key: sum(row['overlap'][threshold][key] for row in rows)
        for key in rows[0]['overlap'][threshold]} for threshold in ('0.3','0.5')}

def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=WEIGHTS['baseline'])
    parser.add_argument('--candidate',type=Path,default=WEIGHTS['candidate'])
    parser.add_argument('--stock',type=Path,default=WEIGHTS['stock'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUTPUT=args.output.resolve(); OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    WEIGHTS.update({name:getattr(args,name).resolve() for name in ('baseline','candidate','stock')})
    torch.set_num_threads(2); cv2.setNumThreads(1)
    if not torch.cuda.is_available(): raise RuntimeError('This comparison expects the reserved CUDA slot')
    review = json.loads((ROOT/'docs/factory-person-review.json').read_text(encoding='utf-8'))
    frames = [(sample, read_frame(ROOT/'data/datasets/factory-expansion/test'/f"{sample['clip']}.mp4", sample['frame'])) for sample in review['rows']]
    report = {'status':'running','deployment_decision':'reject','reason':'Comparison incomplete',
        'ultralytics':__version__, 'device':torch.cuda.get_device_name(0),
        'weights':{name:{'path':str(path),'sha256':digest(path)} for name,path in WEIGHTS.items()},
        'factory_review':review, 'limitations':[
            'Eight debug frames were reviewed after previous predictions; not blind/exhaustive factory ground truth.',
            'Unmatched predictions are not necessarily false positives; hidden/omitted people need review.',
            'Small means reviewed source-frame box height below 150px, fixed before this comparison.',
            'Held-out test is post-training evaluation only; no threshold/checkpoint tuning or training here.',
            'Test/source splits are inherited; independent factory cameras and zero false positives are not established.'],
        'direct_confidence':.4, 'direct_imgsz':640, 'direct_factory':{}, 'heldout_test':{}, 'production_backend':{}}
    if args.resume and OUTPUT.exists():
        previous=json.loads(OUTPUT.read_text(encoding='utf-8'))
        if previous['weights'] != report['weights']:
            raise RuntimeError('Cannot reuse comparison after weight paths/hashes changed')
        report=previous
    save(report)
    dataset = ROOT/'data/cache/targeted-training/prepared-inputs/person/people'
    testfiles = sorted((dataset/'images/test').glob('*'))
    if len(testfiles)!=610: raise RuntimeError(f'Expected fixed 610 test images, got {len(testfiles)}')
    yaml_path = ROOT/'data/cache/targeted-training/person-comparison-data.yaml'
    yaml_path.write_text('path: '+dataset.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n', encoding='utf-8')
    report['heldout_protocol']={'images':len(testfiles),'manifest_sha256':digest(dataset/'manifest.jsonl'),
        'split':'test','classes':[0],'imgsz':640,'batch':8,'workers':0,'ap_confidence_floor':.001,
        'ap_note':'Standard YOLO AP integrates confidence >=.001; reported precision/recall are library operating-point summaries, not fixed-conf .4. Factory comparisons use fixed .4.'}
    for name,path in WEIGHTS.items():
        if name in report['direct_factory'] and (name=='stock' or name in report['heldout_test']):
            continue
        model=YOLO(str(path),task='detect'); validate_person_class(model)
        model.predict(frames[0][1],classes=[0],conf=.4,imgsz=640,device=0,verbose=False)
        rows=[]
        for sample,frame in frames:
            tick=time.perf_counter()
            pred=model.predict(frame,classes=[0],conf=.4,imgsz=640,device=0,verbose=False)[0].cpu()
            elapsed=time.perf_counter()-tick
            boxes=[(list(map(float,b[:4])),float(b[4])) for b in pred.boxes.data.numpy()]
            truth=[[value*2 for value in box] for box in sample['boxes']]
            rows.append({'clip':sample['clip'],'frame':sample['frame'],'elapsed_s':elapsed,
                'predictions':[{'box':b,'score':s} for b,s in boxes], 'overlap':scored(boxes,truth)})
        report['direct_factory'][name]={'rows':rows,'totals':totals(rows),'median_s':float(np.median([r['elapsed_s'] for r in rows]))}
        save(report); print('DIRECT',name,totals(rows),flush=True)
        if name!='stock':
            tick=time.perf_counter()
            metrics=model.val(data=str(yaml_path),split='test',classes=[0],device=0,imgsz=640,
                batch=8,workers=0,plots=False,verbose=False,conf=.001,
                project=str(ROOT/'data/cache/targeted-training/person-comparison-val'),name=name,exist_ok=True)
            report['heldout_test'][name]={'elapsed_s':time.perf_counter()-tick,
                'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
                'speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},
                'per_class_ids':list(map(int,metrics.box.ap_class_index)), 'test_evaluated':True}
            save(report); print('HELDOUT',name,report['heldout_test'][name],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    config=RunConfig(confidence=.4,fast_motion=False,fall_detection=False,distant_people=True,helmet_detection=True)
    for name in ('baseline','candidate'):
        if name in report['production_backend']: continue
        # Same stock ensemble, pose model, refinement, source pixels and thresholds.
        # Explicit assignment avoids environment/file-name-dependent ensemble changes.
        backend=YoloBackend(ROOT/'data/models/yolov8n-pose.pt',.4,config=None)
        backend.person_models=[YOLO(str(WEIGHTS[name]),task='detect'),YOLO(str(WEIGHTS['stock']),task='detect')]
        backend.person_model=backend.person_models[0]; backend.distant_people=True; backend.refine_head_pose=True
        for detector in backend.person_models: validate_person_class(detector)
        tick=time.perf_counter(); backend.predict_people(frames[0][1]); warm=time.perf_counter()-tick
        rows=[]
        for sample,frame in frames:
            tick=time.perf_counter(); people=backend.predict_people(frame); elapsed=time.perf_counter()-tick
            boxes=[(list(p.box),p.confidence) for p in people if p.confidence>=.4]
            truth=[[value*2 for value in box] for box in sample['boxes']]
            rows.append({'clip':sample['clip'],'elapsed_s':elapsed,'overlap':scored(boxes,truth),
                'predictions':[{'box':list(p.box),'score':p.confidence,'sources':list(p.detection_sources),
                    'semantic_person':semantic_person_evidence(p)} for p in people if p.confidence>=.4]})
        source=ROOT/'data/datasets/factory-expansion/test/6_te9.mp4'
        cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS)
        tracker=PeopleTracker(fps/5,.4); sequence=[]; index=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index%5==0:
                tick=time.perf_counter(); people=backend.predict_people(frame)
                tracked=tracker.update(people,frame.shape,index/fps)
                sequence.append({'at_s':index/fps,'elapsed_s':time.perf_counter()-tick,
                    'people':[{'box':list(p.box),'score':p.confidence,'track_id':p.track_id,
                        'observed':p.observed,'alert_eligible':p.alert_eligible,'tracking_state':p.tracking_state,
                        'sources':list(p.detection_sources),'semantic_person':semantic_person_evidence(p)} for p in tracked]})
            index+=1
        cap.release()
        machine=[{'at_s':row['at_s'],**p} for row in sequence for p in row['people'] if p['box'][0]<300 and p['box'][1]>300]
        report['production_backend'][name]={'config':config.model_dump(),'scenario_models_excluded':'Person comparison only; same head-pose refinement enabled, no PPE inference needed.',
            'warmup_s':warm,'rows':rows,'totals':totals(rows),'median_s':float(np.median([r['elapsed_s'] for r in rows])),
            'fp_sequence':{'source':str(source),'sha256':digest(source),'measurements':sequence,
                'machine_region_is_reporting_only':True,'machine_samples':machine,
                'machine_confirmed_count':sum(p['track_id']>0 for p in machine),
                'confirmed_ids':sorted({p['track_id'] for row in sequence for p in row['people'] if p['track_id']>0})}}
        save(report); print('BACKEND',name,totals(rows),'machine confirmed',sum(p['track_id']>0 for p in machine),flush=True)
        del backend; gc.collect(); torch.cuda.empty_cache()
    base=report['heldout_test']['baseline']['metrics']; candidate=report['heldout_test']['candidate']['metrics']
    old=report['production_backend']['baseline']; new=report['production_backend']['candidate']
    ap_gain=candidate['metrics/mAP50-95(B)']-base['metrics/mAP50-95(B)']
    regression=old['totals']['0.3']['matched']-new['totals']['0.3']['matched']
    machine_ok=new['fp_sequence']['machine_confirmed_count']==0
    accepted=ap_gain>=0 and regression<=1 and machine_ok
    report.update(status='completed',deployment_decision='accept' if accepted else 'reject',
        reason=f'Held-out mAP50-95 delta {ap_gain:+.6f}; reviewed production recall delta {-regression:+d}/25; machine confirmed samples {new["fp_sequence"]["machine_confirmed_count"]}. Parent review required before installation.',
        decision_gate={'heldout_ap_non_regression':ap_gain>=0,'factory_drop_at_most_one':regression<=1,'no_confirmed_machine_on_fixed_sequence':machine_ok},
        test_evaluated=True)
    save(report); print('DECISION',report['deployment_decision'],report['reason'],flush=True)

if __name__=='__main__': main()
