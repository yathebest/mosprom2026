"""Quarantine low-resolution classifier crops without changing raw labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
from pathlib import Path
import shutil

import cv2

try:
    from scripts.prepare_ppe_data import crop_quality
except ModuleNotFoundError:
    from prepare_ppe_data import crop_quality


def curate(data,rejected,summary_path):
    data=Path(data).resolve();rejected=Path(rejected).resolve()
    if rejected.is_relative_to(data):raise ValueError('Quarantine must be outside packed dataset')
    summary_path=Path(summary_path);summary=json.loads(summary_path.read_text())
    counts={};removed={};cv2.setNumThreads(1)
    def inspect(p,move):
        crop=cv2.imread(str(p))
        reason='unreadable' if crop is None else crop_quality(crop)
        if reason and move:
            dest=rejected/p.relative_to(data);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(p),str(dest))
        return reason,p.parent.parent.name+':'+p.parent.name
    def batches(paths):
        iterator=iter(paths)
        while batch:=list(islice(iterator,512)):yield batch
    for task in ['head_ppe','vest_ppe']:
        remaining=Counter();bad=Counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Resume after interruption without losing quarantine statistics.
            for batch in batches((rejected/(task+'_classifier')).rglob('*.jpg')):
                for reason,_ in pool.map(lambda p:inspect(p,False),batch):bad[reason or 'prior_filter']+=1
            paths=sorted((data/(task+'_classifier')).rglob('*.jpg'))
            processed=0
            for batch in batches(paths):
                for reason,key in pool.map(lambda p:inspect(p,True),batch):
                    if reason:bad[reason]+=1
                    else:remaining[key]+=1
                processed+=len(batch)
                if processed%10240==0:print(task,processed,'/',len(paths),flush=True)
        counts[task]=dict(remaining);removed[task]=dict(bad)
        print(task,counts[task],removed[task],flush=True)
    summary['classification_crops']=counts
    summary['classifier_quality_gate']={'min_side':24,'min_pixels':1024,'aspect_range':[.2,4],
        'laplacian_variance_min':15,'quarantined':removed,
        'warning':'Automatic clarity filtering does not establish label correctness or independent accuracy.'}
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    try:
        from scripts.finalize_ppe_data import finalize
    except ModuleNotFoundError:
        from finalize_ppe_data import finalize
    return finalize(data,summary_path,rejected/'unused-manifest.jsonl')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/datasets/ppe-expanded')
    p.add_argument('--rejected',default='data/datasets/ppe-rejected-crops')
    p.add_argument('--summary',default='docs/ppe-dataset-summary.json');a=p.parse_args()
    curate(a.data,a.rejected,a.summary)
