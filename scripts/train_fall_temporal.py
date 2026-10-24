"""Continue the evaluated RGB encoder with clip-supervised temporal fall training.

All model/threshold selection uses validation. Test is read only for final scoring.
Figshare has action labels, not onset times: do not call its clip F1 event F1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.fall_temporal import TemporalFallNet
from training_checkpoint import atomic_torch_save


class Clips(Dataset):
    def __init__(self,root,rows,augment=False):self.root=root;self.rows=rows;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];frames=[]
        flip=self.augment and torch.rand(()).item()<.5
        brightness=(.85+.3*torch.rand(()).item()) if self.augment else 1
        # The same transform is used throughout one clip; never shuffle its frames.
        for path in sorted((self.root/row['frames']).glob('*.jpg')):
            with Image.open(path) as image:x=np.asarray(image.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
            if flip:x=x[:,:,::-1].copy()
            x=np.clip(x*brightness,0,1)
            frames.append((x-np.array([.485,.456,.406],np.float32)[:,None,None])/
                          np.array([.229,.224,.225],np.float32)[:,None,None])
        if len(frames)!=16:raise ValueError('Every clip must contain exactly 16 RGB frames')
        return torch.from_numpy(np.stack(frames)),row['label']


def stats(scores,labels,threshold=.5):
    predicted=np.asarray(scores)>=threshold;truth=np.asarray(labels).astype(bool)
    tp=int((predicted&truth).sum());fp=int((predicted&~truth).sum());fn=int((~predicted&truth).sum());tn=int((~predicted&~truth).sum())
    return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn}


def train(args):
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(74291)
    device=torch.device(args.device);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    data=Path(args.data);manifest=data/'fall-manifest.jsonl';manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    parts={name:[r for r in rows if r['split']==name] for name in ['train','val','test']}
    if not all(len(parts[k])>=8 for k in parts):raise ValueError('Insufficient independent clips')
    model=TemporalFallNet()
    if args.fresh:
        from torchvision.models import ResNet18_Weights
        initial_state=ResNet18_Weights.DEFAULT.get_state_dict(progress=True,check_hash=True)
        initial_encoder='official ImageNet ResNet18; no previous fall checkpoint'
    else:
        initial_state=torch.load(data/'initial-fall-resnet18.pt',map_location='cpu',weights_only=True)['state_dict']
        initial_encoder=hashlib.sha256((data/'initial-fall-resnet18.pt').read_bytes()).hexdigest()
    model.encoder.load_state_dict({k:v for k,v in initial_state.items() if not k.startswith('fc.')},strict=True)
    model.to(device)
    last=output/'fall-temporal-last.pt';best_path=output/'fall-temporal.pt'
    prior=torch.load(last,map_location='cpu',weights_only=True) if args.resume and last.exists() else None
    if prior:
        if prior['data_manifest_sha256']!=manifest_sha:raise ValueError('Resume corpus/split identity changed')
        model.load_state_dict(prior['state_dict']);torch.set_rng_state(prior['torch_rng'])
    start=time.monotonic();deadline=start+args.max_minutes*60
    best=float(prior.get('best_validation_f1',-1)) if prior else -1
    loaders={part:DataLoader(Clips(data,part_rows),batch_size=args.batch,shuffle=False,num_workers=args.workers)
             for part,part_rows in parts.items() if part!='test'}

    def evaluate(loader,features=False):
        model.eval();scores=[];labels=[]
        with torch.inference_mode():
            for x,y in loader:
                logits=model.classify_features(x.to(device)) if features else model(x.to(device))
                scores.extend(logits.softmax(1)[:,1].cpu().tolist());labels.extend(y.tolist())
        return np.asarray(scores),np.asarray(labels)

    def checkpoint(stage,epoch,optimizer,partial=False):
        state={'architecture':'resnet18-gru-v1','state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
            'stage':stage,'epoch':epoch,'partial_epoch':partial,'best_validation_f1':best,
            'data_manifest_sha256':manifest_sha,'initial_encoder':initial_encoder,
            'torch_rng':torch.get_rng_state(),'classes':['no_fall','fall'],'frames':16,'window_seconds':2.0,
            'threshold':.5,'evaluation_unit':'action clip, not incident onset'}
        atomic_torch_save(state,last,previous=True)
        print('FALL_CHECKPOINT_SAVED',stage,'epoch',epoch,'bytes',last.stat().st_size,flush=True)
        return state

    # Freeze and cache RGB features: the temporal stage is cheap even on CPU.
    feature_loaders={};model.eval()
    for part in ['train','val']:
        feature_path=output/('features-'+part+'.pt')
        if feature_path.exists():
            cached=torch.load(feature_path,map_location='cpu',weights_only=True)
            if cached['data_manifest_sha256']!=manifest_sha:raise ValueError('Feature cache identity differs')
            vectors,targets=cached['features'],cached['labels']
        else:
            features=[];targets=[]
            with torch.inference_mode():
                for batch,(x,y) in enumerate(loaders[part]):
                    b,t,c,h,w=x.shape
                    encoded=model.encode(x.reshape(b*t,c,h,w).to(device)).reshape(b,t,512)
                    features.append(encoded.cpu());targets.append(y)
                    if batch%25==0:print('FALL_FEATURES',part,batch,'/',len(loaders[part]),flush=True)
                    if time.monotonic()>deadline-15*60:raise TimeoutError('Feature extraction consumed the budget; partial caches/checkpoints retained')
            vectors=torch.cat(features);targets=torch.cat(targets)
            atomic_torch_save({'features':vectors,'labels':targets,'data_manifest_sha256':manifest_sha},feature_path)
        dataset=TensorDataset(vectors,targets)
        counts=torch.bincount(targets,minlength=2).float()
        sampler=WeightedRandomSampler((1/counts)[targets],len(targets),replacement=True) if part=='train' else None
        feature_loaders[part]=DataLoader(dataset,batch_size=32,sampler=sampler)

    history=[];criterion=nn.CrossEntropyLoss()
    for stage,epochs,phase_minutes in [('head',args.head_epochs,12),('finetune',args.finetune_epochs,args.finetune_minutes)]:
        if prior and prior['stage']=='finetune' and stage=='head':continue
        for name,p in model.named_parameters():p.requires_grad=not name.startswith('encoder.') or (stage=='finetune' and name.startswith('encoder.layer4.'))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4 if stage=='head' else 1e-5,weight_decay=1e-3)
        first=0
        if prior and prior['stage']==stage:
            optimizer.load_state_dict(prior['optimizer']);first=prior['epoch'];print('FALL_RESTORED',stage,first,flush=True)
        loader=feature_loaders['train'] if stage=='head' else DataLoader(Clips(data,parts['train'],True),
            batch_size=args.batch,shuffle=True,num_workers=args.workers)
        phase_deadline=min(deadline-10*60,time.monotonic()+phase_minutes*60)
        for epoch in range(first,epochs):
            if time.monotonic()>=phase_deadline:break
            model.train();model.encoder.eval()
            if stage=='finetune':model.encoder.layer4.train()
            loss_sum=0;partial=False
            for batch,(x,y) in enumerate(loader):
                optimizer.zero_grad(set_to_none=True);x,y=x.to(device),y.to(device)
                logits=model.classify_features(x) if stage=='head' else model(x)
                loss=criterion(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                loss_sum+=float(loss.detach())
                if time.monotonic()>=phase_deadline:partial=batch+1<len(loader);break
            scores,labels=evaluate(feature_loaders['val'] if stage=='head' else loaders['val'],features=stage=='head')
            metrics=stats(scores,labels);improved=metrics['f1']>best
            if improved:best=metrics['f1']
            state=checkpoint(stage,epoch if partial else epoch+1,optimizer,partial)
            if improved:atomic_torch_save(state,best_path)
            row={'stage':stage,'epoch':epoch+1,'partial_epoch':partial,'train_loss':loss_sum/max(batch+1,1),
                 'validation_clip':metrics,'elapsed_seconds':time.monotonic()-start}
            history.append(row);(output/'fall-training-history.json').write_text(json.dumps(history,indent=2))
            print('FALL_VALIDATION',json.dumps(row),flush=True)
            if partial:break
    saved=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);model.to(device)
    scores,labels=evaluate(loaders['val'])
    candidates=[(stats(scores,labels,float(t)),float(t)) for t in np.linspace(.05,.95,91)]
    _,threshold=max(candidates,key=lambda pair:(pair[0]['f1'],pair[0]['precision'],pair[1]))
    saved['threshold']=threshold;atomic_torch_save(saved,best_path)
    test_loader=DataLoader(Clips(data,parts['test']),batch_size=args.batch,num_workers=args.workers)
    test_scores,test_labels=evaluate(test_loader)
    report={'architecture':'resnet18-gru-v1','threshold_selected_on_validation':threshold,
        'validation_clip':stats(scores,labels,threshold),'test_clip':stats(test_scores,test_labels,threshold),
        'test_by_source':{source:stats(test_scores[[i for i,r in enumerate(parts['test']) if r['source']==source]],
            test_labels[[i for i,r in enumerate(parts['test']) if r['source']==source]],threshold) for source in {r['source'] for r in parts['test']}},
        'data_manifest_sha256':manifest_sha,'split_counts':{k:len(v) for k,v in parts.items()},
        'elapsed_minutes':(time.monotonic()-start)/60,'event_timing_metrics':'not measured on Figshare: onset labels unavailable',
        'linq_encoder':'ResNet18 family; new weights require compilation and hardware test',
        'temporal_head':'small GRU on CPU; no LinQ GRU support claimed',
        'deployment':'candidate only; do not replace the evaluated baseline automatically'}
    (output/'fall-temporal-metrics.json').write_text(json.dumps(report,indent=2))
    print('FALL_TEST',json.dumps(report),flush=True)
    # Split export keeps the supported ResNet encoder separate from temporal CPU inference.
    model.eval().cpu()
    torch.onnx.export(model.encoder,torch.zeros(1,3,224,224),str(output/'fall-temporal-encoder.onnx'),
        input_names=['images'],output_names=['features'],opset_version=12,dynamo=False)
    class Head(nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,features):return self.network.classify_features(features)
    torch.onnx.export(Head(model),torch.zeros(1,16,512),str(output/'fall-temporal-head.onnx'),
        input_names=['features'],output_names=['logits'],opset_version=12,dynamo=False)
    print('FALL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--head-epochs',type=int,default=30);p.add_argument('--finetune-epochs',type=int,default=4)
    p.add_argument('--finetune-minutes',type=float,default=25);p.add_argument('--max-minutes',type=float,default=90)
    p.add_argument('--resume',action='store_true');p.add_argument('--fresh',action='store_true');train(p.parse_args())
