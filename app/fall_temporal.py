"""RGB temporal model. ResNet18 frame encoder + a small CPU-capable GRU head.

Inputs are RGB frames in chronological order over a two-second window.
Clip classification is not itself validated incident onset localization.
"""
from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class TemporalFallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=resnet18(weights=None);self.encoder.fc=nn.Identity()
        self.temporal=nn.GRU(1024,128,batch_first=True)
        self.classifier=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))

    def encode(self,frames):return self.encoder(frames)

    def classify_features(self,features):
        features=F.layer_norm(features,(512,))
        delta=torch.cat([torch.zeros_like(features[:,:1]),features[:,1:]-features[:,:-1]],dim=1)
        output,_=self.temporal(torch.cat([features,delta],dim=-1))
        return self.classifier(torch.cat([output[:,-1],output.mean(1)],dim=-1))

    def forward(self,frames):
        b,t,c,h,w=frames.shape
        return self.classify_features(self.encode(frames.reshape(b*t,c,h,w)).reshape(b,t,512))


class TemporalFallClassifier:
    def __init__(self,path:Path,device='cpu'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('architecture')!='resnet18-gru-v1':raise ValueError('Unsupported temporal fall model')
        self.model=TemporalFallNet();self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device).eval();self.device=device
        self.threshold=float(checkpoint['threshold']);self.frames=deque(maxlen=16)
        self.last_sample=None;self.window_s=2.0
        self.mean=np.array([.485,.456,.406],np.float32)[:,None,None]
        self.std=np.array([.229,.224,.225],np.float32)[:,None,None]

    @torch.inference_mode()
    def predict(self,bgr,now=None):
        now=time.monotonic() if now is None else now
        if self.last_sample is not None and (now<self.last_sample or now-self.last_sample>.5):
            self.frames.clear();self.last_sample=None
        if self.last_sample is not None and now-self.last_sample<.125:return None
        rgb=Image.fromarray(np.ascontiguousarray(bgr[:,:,::-1])).resize((224,224),Image.Resampling.BILINEAR)
        x=np.asarray(rgb,dtype=np.float32).transpose(2,0,1)/255
        x=torch.from_numpy(np.ascontiguousarray((x-self.mean)/self.std))[None].to(self.device)
        self.frames.append(self.model.encode(x)[0]);self.last_sample=now
        if len(self.frames)<16:return None
        logits=self.model.classify_features(torch.stack(list(self.frames))[None])
        return float(logits.softmax(1)[0,1].cpu())
