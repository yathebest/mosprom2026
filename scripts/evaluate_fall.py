"""Sequence-level evaluation; select thresholds on validation, test once frozen.

python scripts/evaluate_fall.py --split validation
python scripts/evaluate_fall.py --split test
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.body_classifier import BodyClassifier
from app.core import Rules,RunConfig
from scripts.evaluate_events import evaluate
from scripts.train_fall_classifier import annotations


def score(records,threshold,hold):
    predictions=[];truth=[];delays=[];errors=[]
    for sequence,record in records.items():
        config=RunConfig(fast_motion=False,fall_detection=True,body_threshold=threshold,body_hold=hold)
        rules=Rules(config);found=[]
        for point in record['probabilities']:
            events,_=rules.step([],point['time'],320,240,{'lying_probability':point['probability']})
            for event in events:
                if event['kind']!='possible_fall':
                    continue  # Sustained lying is a different event, not a predicted fall.
                predictions.append({'video':sequence,'kind':'fall','alert_s':event['alert_s']})
                found.append(event['alert_s'])
        if record['onset_s'] is not None:
            truth.append({'video':sequence,'kind':'fall','start_s':record['onset_s'],'end_s':record['duration_s']})
            if not any(record['onset_s']<=t<=record['duration_s'] for t in found):errors.append({'video':sequence,'error':'miss','alerts':found})
        elif found:
            errors.append({'video':sequence,'error':'false_alarm','alerts':found})
    result=evaluate(truth,predictions)
    result.update(threshold=threshold,hold_s=hold,sequences=len(records),errors=errors,predictions=predictions,truth=truth)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    protocol=json.loads((ROOT/'docs/evaluation_protocol.json').read_text())
    folder=ROOT/'data/datasets/urfd';labels=annotations(folder)
    weights=ROOT/'data/models/fall-resnet18.pt';sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    config_path=ROOT/'data/models/fall-operating-point.json'
    if args.split=='test':
        operating=json.loads(config_path.read_text())
        if operating['checkpoint_sha256']!=sha:raise ValueError('Checkpoint changed after validation freeze')
        if (ROOT/'docs/quality_report.json').exists():raise ValueError('A final test already exists; do not tune on it or overwrite it silently')
    model=BodyClassifier(weights,'cuda:0' if torch.cuda.is_available() else 'cpu')
    cache=ROOT/'data/cache';cache.mkdir(exist_ok=True)
    records={};began=time.time()
    for sequence,split in protocol['split'].items():
        if split!=args.split:continue
        path=folder/f'{sequence}-cam0-rgb.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seq_labels=labels.get(sequence,{})
        onset=min((i for i,l in seq_labels.items() if l in (0,1)),default=None) if sequence.startswith('fall') else None
        probabilities=[]
        for index in range(n):
            ok,frame=cap.read()
            if not ok:raise ValueError('Early decode end '+sequence)
            if index%3:continue
            probabilities.append({'frame':index,'time':index/fps,'probability':model.predict(frame),
                                  'label':seq_labels.get(index)})
        cap.release()
        records[sequence]={'probabilities':probabilities,'duration_s':n/fps,'onset_s':onset/fps if onset is not None else None}
        print(sequence,len(probabilities),'sampled RGB frames',flush=True)
    (cache/f'fall-{args.split}-{sha[:12]}.json').write_text(json.dumps(records),encoding='utf-8')
    if args.split=='validation':
        options=[score(records,t,h) for t in [.5,.65,.8,.9,.95] for h in [.1,.2,.3,.4,.5]]
        # Prefer F1, then precision, then longer confirmation on equal scores.
        best=max(options,key=lambda r:(r['f1'] or 0,r['precision'] or 0,r['hold_s'],r['threshold']))
        operating={'threshold':best['threshold'],'hold_s':best['hold_s'],'checkpoint_sha256':sha,
                   'selected_on':'validation sequences only','selection':'event F1, then precision, longer hold and higher threshold'}
        config_path.write_text(json.dumps(operating,indent=2),encoding='utf-8')
        report={'split':'validation','best':best,'selection_grid':options,'checkpoint_sha256':sha}
        (ROOT/'docs/validation_fall.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('VALIDATION',json.dumps(best),flush=True)
    else:
        result=score(records,operating['threshold'],operating['hold_s'])
        labeled=[p for r in records.values() for p in r['probabilities'] if p['label'] in (-1,1)]
        tp=sum(p['probability']>=operating['threshold'] and p['label']==1 for p in labeled)
        fp=sum(p['probability']>=operating['threshold'] and p['label']==-1 for p in labeled)
        fn=sum(p['probability']<operating['threshold'] and p['label']==1 for p in labeled)
        frame={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'sampled_labeled_frames':len(labeled)}
        normal_hours=sum(r['duration_s'] for k,r in records.items() if k.startswith('adl'))/3600
        normal_false=sum(1 for p in result['predictions'] if p['video'].startswith('adl'))
        report={'status':'measured','dataset':'URFD RGB cam0 previews','split':'frozen whole-sequence test',
                'model':'ResNet18 RGB scene body-state classifier + temporal events','checkpoint_sha256':sha,
                'test_sequences':list(records),'operating_point':operating,'events':result,'frames':frame,
                'normal_video_hours':normal_hours,'normal_false_alarms_per_hour':normal_false/max(normal_hours,1e-9),
                'elapsed_evaluation_s':round(time.time()-began,2),'linq_measured':False,
                'limitations':['20 short controlled indoor sequences; not an industrial deployment benchmark',
                    'Person-disjointness cannot be verified from supplied preview metadata',
                    'No quality claims for running, PPE, phone, smoking, harness or machinery',
                    'Frame metrics exclude transition/unknown author labels; event onset uses first transition/fallen label',
                    'Threshold/model selected on validation; test not used for fitting or threshold selection']}
        (ROOT/'docs/quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('FINAL TEST',json.dumps(report),flush=True)


if __name__=='__main__':main()
