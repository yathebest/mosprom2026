"""RGB-only scene-level body-state model; no depth inputs or track identity."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18


class BodyClassifier:
    def __init__(self,path: Path,device='cpu'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        self.model=resnet18(weights=None)
        self.model.fc=nn.Linear(self.model.fc.in_features,2)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.device=device
        self.model.to(device).eval()
        self.selected_epoch=checkpoint.get('selected_epoch')
        self.mean=np.array([.485,.456,.406],np.float32)[:,None,None]
        self.std=np.array([.229,.224,.225],np.float32)[:,None,None]

    @torch.inference_mode()
    def predict(self,bgr):
        rgb=Image.fromarray(np.ascontiguousarray(bgr[:,:,::-1])).resize((224,224),Image.Resampling.BILINEAR)
        x=np.asarray(rgb,dtype=np.float32).transpose(2,0,1)/255
        x=np.ascontiguousarray((x-self.mean)/self.std)
        logits=self.model(torch.from_numpy(x)[None].to(self.device))
        return float(logits.softmax(1)[0,1].cpu())
