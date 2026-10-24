"""Train a LinQ-family YOLO12m cigarette candidate, never replace production.

CigDet contains web-image cigarette boxes, not factory smoking-event labels.
Near-duplicate groups stay together. Original test stays out of selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=10)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'data/datasets/cigdet/CigDet_dataset'
    output=root/'data/cache/smoking-yolo12m';data=output/'dataset';output.mkdir(parents=True,exist_ok=True)
    files=sorted((source/'train').glob('*.jpg'));test=sorted((source/'test').glob('*.jpg'))
    def perceptual(path):
        image=cv2.resize(cv2.imread(str(path),0),(32,32)).astype(np.float32)
        dct=cv2.dct(image)[:8,:8].reshape(-1)[1:]
        return sum(int(v>np.median(dct))<<i for i,v in enumerate(dct))
    hashes={f:perceptual(f) for f in files+test};parent=list(range(len(files)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    excluded=[];clean=[]
    for f in files:
        if any((hashes[f]^hashes[t]).bit_count()<=4 for t in test):excluded.append(f)
        else:clean.append(f)
    files=clean;parent=list(range(len(files)))
    for i,f in enumerate(files):
        for j in range(i):
            if (hashes[f]^hashes[files[j]]).bit_count()<=4:parent[find(i)]=find(j)
    groups={}
    for i,f in enumerate(files):groups.setdefault(find(i),[]).append(f)
    ordered=sorted(groups);random.Random(42).shuffle(ordered)
    val_groups=set(ordered[:max(1,round(len(ordered)*.2))]);splits={'train':[],'val':[],'test':test}
    for group,rows in groups.items():splits['val' if group in val_groups else 'train'].extend(rows)
    manifest={'source':'https://data.mendeley.com/datasets/6hyrr8typ7/1','author':'Ali Khan','license':'CC-BY-4.0',
        'scope':'Web cigarette boxes, not industrial smoking events','perceptual_hash_max_distance':4,
        'excluded_train_test_near_duplicates':[f.name for f in excluded],'splits':{}}
    for split,rows in splits.items():
        for kind in ['images','labels']:(data/kind/split).mkdir(parents=True,exist_ok=True)
        manifest['splits'][split]=[]
        for f in rows:
            text=f.with_suffix('.txt').read_text()
            for line in text.splitlines():
                values=line.split()
                if len(values)!=5 or int(values[0])!=0 or any(not 0<=float(v)<=1 for v in values[1:]):raise ValueError('Invalid annotation')
            shutil.copyfile(f,data/'images'/split/f.name)
            (data/'labels'/split/(f.stem+'.txt')).write_text(text)
            manifest['splits'][split].append({'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    (output/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2))
    import yaml
    config=data/'data.yaml';config.write_text(yaml.safe_dump({'path':str(data.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'cigarette'}}))
    print('PREPARED',json.dumps({k:len(v) for k,v in splits.items()}),'excluded',len(excluded),flush=True)
    if a.prepare_only:return
    import torch
    from ultralytics import YOLO,settings
    from training_checkpoint import atomic_copy
    torch.set_num_threads(2);cv2.setNumThreads(1)
    dict.update(settings,{k:False for k in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if k in settings})
    model=YOLO(str(root/'data/models/yolo12m.pt'))
    clock={}
    def begin(trainer):clock['deadline']=time.monotonic()+a.minutes*60
    def limit(trainer):
        if time.monotonic()>=clock['deadline']:trainer.stop=True
    def checkpoint(trainer):
        if trainer.last.exists():
            path=trainer.last.with_name('resume-last.pt');atomic_copy(trainer.last,path)
            print('CHECKPOINT_VERIFIED',int(trainer.epoch)+1,path.stat().st_size,flush=True)
    model.add_callback('on_train_start',begin);model.add_callback('on_train_batch_end',limit);model.add_callback('on_model_save',checkpoint)
    model.train(data=str(config),epochs=50,imgsz=640,batch=2,device=0,workers=2,amp=True,
        project=str(output),name='training',exist_ok=True,seed=42,patience=12,close_mosaic=10,
        optimizer='AdamW',lr0=.0002,save_period=1,cache=False,verbose=False)
    best=output/'training/weights/best.pt';candidate=YOLO(str(best))
    metrics=candidate.val(data=str(config),split='val',device=0,verbose=False)
    report={'source':manifest['source'],'weights':str(best.relative_to(root)),'validation':metrics.results_dict,
        'selection':'Ultralytics validation fitness only; test not used for epoch selection',
        'status':'candidate_not_deployed','time_budget_minutes':a.minutes,
        'warning':'Requires original-vs-candidate test and non-smoking factory controls; not factory event metrics.'}
    (output/'candidate.json').write_text(json.dumps(report,indent=2,default=float))
    print('CANDIDATE_READY',json.dumps(report,default=float),flush=True)


if __name__=='__main__':main()
