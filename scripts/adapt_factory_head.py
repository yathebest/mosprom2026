"""Guarded crop-classifier adaptation. Reviewed crops are not event/bbox GT.

Freeze ResNet50 features, fit only its final layer. Whole source videos stay
within one split. Test labels never select an epoch. No installed file changes.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from training_checkpoint import atomic_torch_save

VAL_VIDEOS={'train/2_tr21.mp4','train/4_tr24.mp4','train/6_tr15.mp4','train/1_tr39.mp4'}
EXCLUDED='8ab6abebd308d47cead1c3b6403e4bc767c288766d4113dde197c3609373a0c3'


class Crops(Dataset):
    def __init__(self,rows):
        self.rows=rows
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['path']) as im:im=im.convert('RGB')
        if r.get('degrade'):
            # Deterministic train-only scale augmentation; cannot invent pixels.
            side=r['degrade'];im=im.resize((side,side),Image.Resampling.BILINEAR)
        return self.transform(im),r['label']


def metric(logits,y):
    prob=logits.softmax(1);pred=prob.argmax(1);selected=prob.max(1).values>=.9
    out={'n':len(y),'coverage':float(selected.float().mean()),'accuracy':float((pred==y).float().mean())}
    for k,name in enumerate(['hardhat','no_hardhat']):
        tp=int(((pred==k)&(y==k)&selected).sum());fp=int(((pred==k)&(y!=k)&selected).sum())
        fn=int(((y==k)&((pred!=k)|~selected)).sum())
        out[name]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/datasets/ppe-prepared/head_ppe_classifier')
    p.add_argument('--review',default='data/datasets/factory-review')
    p.add_argument('--base',default='data/cache/factory-adaptation/head-base.pt')
    p.add_argument('--output',default='data/cache/factory-adaptation');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    saved=torch.load(a.base,map_location='cpu',weights_only=True)
    if saved['classes']!=['hardhat','no_hardhat']:raise ValueError('Unexpected class order')
    rows={}
    for split,count in [('train',1024),('val',512),('test',512)]:
        group=[]
        for label,name in enumerate(saved['classes']):
            files=sorted(x for x in (Path(a.root)/split/name).glob('*') if x.is_file() and not x.stem.startswith(EXCLUDED))
            random.Random(42).shuffle(files)
            for i,f in enumerate(files[:count]):
                group.append({'path':str(f),'label':label,'degrade':([24,32,48,64][i%4] if split=='train' and i%2 else 0)})
        rows['original_'+split]=group
    review=json.loads((Path(a.review)/'reviewed-head-crops.json').read_text())
    for split in ['train','val','test']:rows['factory_'+split]=[]
    for r in review['rows']:
        if r['label']!='no_hardhat':continue
        split='test' if r['source_split']=='test' else 'val' if r['source_video'] in VAL_VIDEOS else 'train'
        row={'path':str(Path(a.review)/r['crop_file']),'label':1,'id':r['id'],'source_video':r['source_video']}
        rows['factory_'+split].append(row)
    sets={k:{r['source_video'] for r in v} for k,v in rows.items() if k.startswith('factory_')}
    assert not (sets['factory_train']&sets['factory_val'] or sets['factory_train']&sets['factory_test'] or sets['factory_val']&sets['factory_test'])
    (out/'split-manifest.json').write_text(json.dumps({'rows':rows,'warning':review['method']},indent=2))
    print('SPLITS',json.dumps({k:len(v) for k,v in rows.items()}),flush=True)
    model=models.resnet50(weights=None);model.fc=torch.nn.Linear(2048,2);model.load_state_dict(saved['state_dict'])
    initial={k:v.detach().clone() for k,v in model.fc.state_dict().items()}
    model.fc=torch.nn.Identity();model.eval().cuda()
    features={}
    for name,group in rows.items():
        xs=[];ys=[]
        with torch.inference_mode():
            for x,y in DataLoader(Crops(group),batch_size=32,num_workers=2):
                xs.append(model(x.cuda()).cpu());ys.append(y)
        features[name]=(torch.cat(xs),torch.cat(ys))
        print('FEATURES',name,len(group),flush=True)
    torch.save(features,out/'features.pt');model.cpu();torch.cuda.empty_cache()
    head=torch.nn.Linear(2048,2);head.load_state_dict(initial)
    baseline={k:metric(head(x).detach(),y) for k,(x,y) in features.items() if not k.endswith('train')}
    print('BASELINE',json.dumps(baseline),flush=True)
    x=torch.cat([features['original_train'][0],features['factory_train'][0].repeat(8,1)])
    y=torch.cat([features['original_train'][1],features['factory_train'][1].repeat(8)])
    optimizer=torch.optim.AdamW(head.parameters(),lr=2e-5,weight_decay=.01)
    best_score=baseline['factory_val']['no_hardhat']['recall'];best=None;history=[]
    for epoch in range(1,81):
        head.train();order=torch.randperm(len(y))
        for ids in order.split(128):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids]),y[ids])
            # Penalize drifting from the original classifier on sparse target data.
            loss=loss+.05*sum((v-initial[k]).square().sum() for k,v in head.state_dict(keep_vars=True).items())
            loss.backward();optimizer.step()
        head.eval()
        original=metric(head(features['original_val'][0]).detach(),features['original_val'][1])
        factory=metric(head(features['factory_val'][0]).detach(),features['factory_val'][1])
        guard=(original['hardhat']['recall']>=baseline['original_val']['hardhat']['recall']-.005 and
               original['no_hardhat']['fp']<=baseline['original_val']['no_hardhat']['fp']+2 and
               original['accuracy']>=baseline['original_val']['accuracy']-.005)
        score=factory['no_hardhat']['recall']
        history.append({'epoch':epoch,'guard_passed':guard,'original_val':original,'factory_val':factory})
        atomic_torch_save({'fc':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},out/'last.pt')
        if guard and score>best_score:
            best_score=score;best={k:v.detach().clone() for k,v in head.state_dict().items()}
            atomic_torch_save({'fc':best,'epoch':epoch},out/'best-fc.pt')
        if epoch%10==0:print('EPOCH',epoch,'factory_val_recall',score,'guard',guard,'CHECKPOINT_SAVED',flush=True)
    report={'method':'Frozen ResNet50 features; final-layer target-camera adaptation',
        'annotation_scope':review['method'],'threshold':.9,'selection':'whole-video validation only; original hardhat regression guards',
        'base_sha256':hashlib.sha256(Path(a.base).read_bytes()).hexdigest(),'baseline':baseline,'history':history,
        'accepted_on_validation':best is not None,'test_warning':'Crop tests are conditional on proposed visible heads, not person/event recall'}
    if best is not None:
        head.load_state_dict(best)
        report['candidate_test']={k:metric(head(v[0]).detach(),v[1]) for k,v in features.items() if k.endswith('test')}
        state=saved['state_dict'];state['fc.weight']=best['weight'];state['fc.bias']=best['bias']
        atomic_torch_save({'state_dict':state,'classes':saved['classes']},out/'head-camera-candidate.pt')
    (out/'adaptation-report.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
