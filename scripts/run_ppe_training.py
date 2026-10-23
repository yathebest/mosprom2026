"""Run independent tasks on up to two GPUs, keeping logs and resume state."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import gc


def run(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Training requires a CUDA GPU')
    devices=list(range(min(torch.cuda.device_count(),2)))
    queues=[[('detect','people'),('detect','vest_ppe'),('classify','head_ppe')],
            [('detect','head_ppe'),('classify','vest_ppe')]]
    if len(devices)==1:queues=[queues[0]+queues[1]]
    if args.jobs:
        requested=set(args.jobs)
        queues=[[job for job in queue if '-'.join(job) in requested] for queue in queues]
    weights=None
    if any(mode=='detect' for queue in queues for mode,_ in queue):
        from ultralytics import YOLO
        print('Preparing YOLO12m initialization once before parallel workers',flush=True)
        initial=YOLO('yolo12m.pt');weights=str(Path(initial.ckpt_path).resolve())
        del initial;gc.collect()
        # Ultralytics' AMP sanity check loads this tiny official checkpoint.
        # Fetch it once so concurrent trainers cannot overwrite one download.
        initial=YOLO('yolo26n.pt');del initial;gc.collect()
    if any(mode=='classify' for queue in queues for mode,_ in queue):
        from torchvision import models
        print('Preparing classifier initialization once before parallel workers',flush=True)
        enum=models.ResNet152_Weights if args.classifier=='resnet152' else models.ResNet50_Weights
        initial=enum.DEFAULT.get_state_dict(progress=True,check_hash=True)
        del initial;gc.collect()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('train_ppe.py')
    manifest=Path(args.data)/'manifest.jsonl'
    identity={'data':str(Path(args.data).resolve()),'epochs':args.epochs,'classifier_epochs':args.classifier_epochs,
              'classifier':args.classifier,'batch':args.batch,'classifier_batch':args.classifier_batch,'imgsz':args.imgsz,
              'trainer_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
              'data_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else 'bounded-smoke-subset'}
    progress=output/'training-progress.json';lock=threading.Lock()
    prior=json.loads(progress.read_text()) if progress.exists() else {}
    state={'identity':identity,'jobs':prior.get('jobs',{}) if prior.get('identity')==identity else {},
           'gpus':[torch.cuda.get_device_name(i) for i in devices]}
    def save():
        temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(progress)
    def queue(device,jobs):
        for mode,task in jobs:
            key=mode+'-'+task
            if state['jobs'].get(key,{}).get('status')=='done':
                print('SKIP completed',key,flush=True);continue
            log=output/(key+'.log')
            cmd=[sys.executable,str(script),'--data',args.data,'--output',args.output,'--task',task,
                 '--mode',mode,'--device',str(device),'--epochs',str(args.epochs if mode=='detect' else args.classifier_epochs),
                 '--batch',str(args.batch if mode=='detect' else args.classifier_batch),'--imgsz',str(args.imgsz),'--resume']
            if mode=='classify':cmd+=['--classifier',args.classifier]
            elif weights:cmd+=['--weights',weights]
            with lock:
                state['jobs'][key]={'status':'running','gpu':device,'log':log.name};save()
            print('START',key,'GPU',device,flush=True)
            env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            with log.open('a',encoding='utf-8') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            with lock:
                state['jobs'][key].update(status='done' if result.returncode==0 else 'failed',returncode=result.returncode);save()
            print('DONE' if result.returncode==0 else 'FAILED',key,'GPU',device,flush=True)
    with lock:save()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pending={pool.submit(queue,device,jobs) for device,jobs in zip(devices,queues)}
        while pending:
            done,pending=wait(pending,timeout=20)
            for future in done:future.result()
            with lock:running=[(key,dict(job)) for key,job in state['jobs'].items() if job['status']=='running']
            for key,job in running:
                try:
                    with (output/job['log']).open('rb') as f:
                        f.seek(0,2);f.seek(max(0,f.tell()-3000));tail=f.read().decode('utf-8',errors='replace')
                    lines=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',tail).replace('\r','\n').splitlines()
                    print(key,'GPU',job['gpu'],next((s.strip()[:220] for s in reversed(lines) if s.strip()),'initializing'),flush=True)
                except OSError:pass
    if any(job['status']=='failed' for job in state['jobs'].values()):
        raise RuntimeError('One or more tasks failed; see training-progress.json and per-task logs')
    print('All requested tasks completed. Independent event quality remains unmeasured.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--classifier-epochs',type=int,default=30)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--classifier-batch',type=int,default=16)
    p.add_argument('--classifier',choices=['resnet50','resnet152'],default='resnet152')
    p.add_argument('--jobs',nargs='+',choices=['detect-people','detect-head_ppe','detect-vest_ppe','classify-head_ppe','classify-vest_ppe'])
    run(p.parse_args())
