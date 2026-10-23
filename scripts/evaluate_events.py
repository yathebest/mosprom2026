"""Match predicted events to annotated intervals one-to-one.

python scripts/evaluate_events.py --truth labels.json --predictions events.json
truth: [{"video":"id", "kind":"danger_zone", "start_s":2,"end_s":5}]
predictions: [{"video":"id", "kind":"danger_zone", "alert_s":3}]
Compute false alarms/hour on a separately evaluated normal set with known duration.
"""
import argparse
import json
from pathlib import Path


def evaluate(truth,predictions,tolerance=0.0):
    # Sort by time. An extra alarm for an already matched event counts as FP.
    used=set();tp=0;delays=[]
    for p in sorted(predictions,key=lambda e:e['alert_s']):
        candidates=[(i,t) for i,t in enumerate(truth) if i not in used
                    and t['video']==p['video'] and t['kind']==p['kind']
                    and t['start_s']-tolerance<=p['alert_s']<=t['end_s']+tolerance]
        if candidates:
            i,t=min(candidates,key=lambda item:abs(p['alert_s']-item[1]['start_s']))
            used.add(i);tp+=1;delays.append(p['alert_s']-t['start_s'])
    fp,fn=len(predictions)-tp,len(truth)-tp
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'matched_alert_delays_s':delays,'tolerance_s':tolerance}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--truth',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=0)
    args=parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.truth.read_text(encoding='utf-8')),
                              json.loads(args.predictions.read_text(encoding='utf-8')),args.tolerance),indent=2))
