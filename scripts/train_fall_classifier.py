"""Train an RGB-only body-state classifier on a frozen, sequence-level split.

Only train/validation sequences are read. Test inference is a separate command.
Author-provided depth-derived labels are used as supervision only; columns
with depth/accelerometer features are never inputs to the network.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def annotations(folder):
    labels={}
    for name in ('urfall-cam0-falls.csv','urfall-cam0-adls.csv'):
        with (folder/name).open() as f:
            for row in csv.reader(f):
                if row:
                    labels.setdefault(row[0],{})[int(row[1])-1]=int(row[2])
    return labels


def prepare(split,stride=4):
    folder=ROOT/'data/datasets/urfd'
    labels=annotations(folder)
    cache=ROOT/'data/cache/urfd-frames'
    cache.mkdir(parents=True,exist_ok=True)
    samples={'train':[],'validation':[]}
    for sequence,part in split.items():
        if part not in samples:
            continue
        path=folder/f'{sequence}-cam0-rgb.mp4'
        cap=cv2.VideoCapture(str(path))
        index=0
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            label=labels.get(sequence,{}).get(index)
            if index%stride==0 and label in (-1,1):
                target=cache/f'{sequence}-{index:04d}.jpg'
                if not target.exists():
                    cv2.imwrite(str(target),cv2.resize(frame,(256,192)))
                samples[part].append((str(target),int(label==1),sequence,index))
            index+=1
        cap.release()
    return samples


class Frames(Dataset):
    def __init__(self,rows,train):
        self.rows=rows
        augment=[transforms.ColorJitter(.25,.25,.15,.04),transforms.RandomGrayscale(.12),
                 transforms.RandomHorizontalFlip(),transforms.RandomAffine(8,translate=(.04,.04),scale=(.9,1.1))] if train else []
        self.transform=transforms.Compose([transforms.Resize((224,224)),*augment,
            transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        path,label,_,_=self.rows[i]
        with Image.open(path) as im:x=self.transform(im.convert('RGB'))
        return x,label


def run():
    args=argparse.ArgumentParser()
    args.add_argument('--epochs',type=int,default=18)
    args.add_argument('--batch',type=int,default=32)
    args.add_argument('--overwrite-checkpoint',action='store_true',help='Explicitly replace the checkpoint; previous quality report becomes stale')
    args=args.parse_args()
    if (ROOT/'data/models/fall-resnet18.pt').exists() and not args.overwrite_checkpoint:
        raise SystemExit('Checkpoint already exists. Preserve this evaluated version; train in a separate project copy, or explicitly use --overwrite-checkpoint.')
    seed=74291
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.hub.set_dir(str(ROOT/'data/models/torch-cache'))
    protocol=json.loads((ROOT/'docs/evaluation_protocol.json').read_text())
    samples=prepare(protocol['split'])
    print('Samples',{k:len(v) for k,v in samples.items()},'device',device,flush=True)
    train=Frames(samples['train'],True);valid=Frames(samples['validation'],False)
    targets=np.array([r[1] for r in samples['train']]);counts=np.bincount(targets,minlength=2)
    sampler=WeightedRandomSampler(torch.as_tensor(1/counts[targets],dtype=torch.double),len(train),replacement=True)
    loader=DataLoader(train,batch_size=args.batch,sampler=sampler,num_workers=0)
    val_loader=DataLoader(valid,batch_size=args.batch,shuffle=False,num_workers=0)
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc=nn.Linear(model.fc.in_features,2)
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    criterion=nn.CrossEntropyLoss()
    best=-1;history=[];started=time.time()
    out=ROOT/'data/models/fall-resnet18.pt'
    for epoch in range(args.epochs):
        model.train();loss_sum=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=device=='cuda'):
                loss=criterion(model(x),y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(x)
        model.eval();probs=[];truth=[]
        with torch.inference_mode():
            for x,y in val_loader:
                with torch.amp.autocast('cuda',enabled=device=='cuda'):
                    p=model(x.to(device)).float().softmax(1)[:,1]
                probs.extend(p.cpu().tolist());truth.extend(y.tolist())
        pred=np.array(probs)>=.5;y=np.array(truth).astype(bool)
        tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
        f1=2*tp/max(2*tp+fp+fn,1)
        row={'epoch':epoch+1,'train_loss':loss_sum/len(train),'validation_frame_f1':f1,
             'tp':tp,'fp':fp,'fn':fn,'elapsed_s':round(time.time()-started,1)}
        history.append(row);print(json.dumps(row),flush=True)
        if f1>best:
            best=f1
            torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                        'architecture':'resnet18','classes':['not_lying','lying'],
                        'input_size':224,'normalization':'ImageNet mean/std','selected_epoch':epoch+1,
                        'selection':'highest validation frame F1 at threshold 0.5; test was not accessed',
                        'protocol':protocol,'seed':seed},out)
        (ROOT/'docs/training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print('BEST',best,'checkpoint',out,flush=True)
    manifest={'file':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'training':'RGB frames from URFD training sequences; whole-video validation',
              'test_accessed':False,'dataset_license':'CC BY-NC-SA 4.0',
              'warning':'Indoor controlled dataset. No verified subject-disjoint split or industrial-generalization claim.'}
    (ROOT/'data/models/fall-resnet18-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':run()
