"""Only this boundary changes when porting inference from PyTorch to LinQ."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from .core import Person


class Backend(Protocol):
    name: str
    def predict(self, bgr: np.ndarray) -> list[Person]: ...


class YoloBackend:
    def __init__(self, model: Path, confidence: float, fall_detection=False,config=None):
        import torch
        from ultralytics import YOLO
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.name = f"PyTorch / {self.device} / {model.name}"
        self.model = YOLO(str(model),task="pose")
        self.confidence = confidence
        self.model_id = model.stem
        self.rotated_search = os.getenv("SAFETY_ROTATED_SEARCH","0")=="1"
        self.scene_state = None
        self.body = None
        self.objects_state=[]
        self.objects=None
        if config is not None:
            from .scenario_models import ScenarioModels
            self.objects=ScenarioModels(model.parent,config)
            if self.objects.models:
                self.name+=' + ONNX CPU scenarios'
                self.model_id+='+'+'+'.join(m.name for m in self.objects.models)
        body_path=model.parent/'fall-resnet18.pt'
        if fall_detection and body_path.exists():
            from .body_classifier import BodyClassifier
            self.body=BodyClassifier(body_path,self.device)
            self.name += " + RGB ResNet18"
            self.model_id += "+fall-resnet18"

    def predict(self, bgr):
        if self.body:
            self.scene_state={"lying_probability":self.body.predict(bgr),"scope":"scene"}
        # Low confidence detections are retained for ByteTrack's second pass.
        rotations = [0,1,3] if self.rotated_search else [0]
        views = [np.ascontiguousarray(np.rot90(bgr,k)) for k in rotations]
        results = self.model.predict(views,conf=.1,imgsz=640,rect=False,device=self.device,verbose=False)
        h,w = bgr.shape[:2]
        candidates = []
        for k,result in zip(rotations,results):
            boxes = result.boxes.data.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for i,b in enumerate(boxes):
                if int(b[5])!=0:
                    continue
                corners = np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
                corners = inverse_rotation(corners,k,w,h)
                box = (float(corners[:,0].min()),float(corners[:,1].min()),
                       float(corners[:,0].max()),float(corners[:,1].max()))
                box=tuple(np.clip(box,[0,0,0,0],[w,h,w,h]).tolist())
                points = keypoints[i].copy() if keypoints is not None else None
                if points is not None:
                    points[:,:2] = inverse_rotation(points[:,:2],k,w,h)
                candidates.append(Person(box,float(b[4]),points))
        candidates=suppress_duplicates(candidates)
        self.objects_state=self.objects.predict(bgr,candidates) if self.objects else []
        from .scenario_models import iou
        for obj in self.objects_state:
            if obj.label=='person' and not any(iou(p.box,obj.box)>.3 for p in candidates):
                candidates.append(Person(obj.box,obj.confidence))
        return candidates


def inverse_rotation(points,k,width,height):
    out = np.asarray(points).copy()
    if k==1:
        out[:,0],out[:,1] = width-1-points[:,1],points[:,0]
    elif k==3:
        out[:,0],out[:,1] = points[:,1],height-1-points[:,0]
    return out


def suppress_duplicates(people,threshold=.45):
    output=[]
    for p in sorted(people,key=lambda p:p.confidence,reverse=True):
        a=np.asarray(p.box)
        duplicate=False
        for other in output:
            b=np.asarray(other.box)
            inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
            if inter/max(union,1e-9)>threshold:
                duplicate=True
                break
        if not duplicate:
            output.append(p)
    return output


def create_backend(model,confidence,fall_detection=False,config=None):
    adapter = os.getenv("LINQ_ADAPTER","")
    if adapter:
        # Explicit local configuration only. Never populated from uploaded files.
        module,factory = adapter.split(":",1)
        backend = getattr(importlib.import_module(module),factory)(confidence=confidence)
        if not callable(getattr(backend,"predict",None)) or not getattr(backend,"name",""):
            raise ValueError("LinQ adapter must expose name and predict(BGR)->list[Person]")
        if config is not None:
            from .scenario_models import required_models
            if required_models(config) and not callable(getattr(backend,'configure_scenarios',None)):
                raise ValueError('Адаптер LinQ не поддерживает выбранные дополнительные модели сценариев')
            if callable(getattr(backend,'configure_scenarios',None)):
                backend.configure_scenarios(config)
        return backend
    if not model.exists():
        raise FileNotFoundError("Нет весов модели. Выполните python scripts/download_model.py")
    return YoloBackend(model,confidence,fall_detection,config)


class StableTracks:
    """Conservative short-gap spatial relinking, without biometric identity."""
    def __init__(self,fps):
        self.fps=fps;self.frame=0;self.next_id=1;self.mapping={};self.recent={}

    def update(self,people):
        self.frame+=1
        occupied=set();pending=[]
        for p in sorted(people,key=lambda p:p.confidence,reverse=True):
            raw=p.track_id;canonical=self.mapping.get(raw)
            if canonical in self.recent and canonical not in occupied:
                p.track_id=canonical;occupied.add(canonical)
            else:
                pending.append((raw,p))
        for raw,p in pending:
            matches=[]
            a=np.asarray(p.box)
            for canonical,(last,bbox) in self.recent.items():
                if canonical in occupied or self.frame-last>max(1,round(self.fps*1.2)):
                    continue
                b=np.asarray(bbox)
                inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                aa=np.maximum(0,a[2:]-a[:2]).prod();bb=np.maximum(0,b[2:]-b[:2]).prod()
                iou=inter/max(aa+bb-inter,1e-9)
                if iou>=.3 and .3<aa/max(bb,1)<3:
                    matches.append((iou,canonical))
            if matches:
                canonical=max(matches)[1]
            else:
                canonical=self.next_id;self.next_id+=1
            self.mapping[raw]=canonical;p.track_id=canonical;occupied.add(canonical)
        for p in people:
            self.recent[p.track_id]=(self.frame,p.box)
        for canonical,(last,_) in list(self.recent.items()):
            if self.frame-last>self.fps*15:
                del self.recent[canonical]
        self.mapping={k:v for k,v in self.mapping.items() if v in self.recent}
        return people


class PeopleTracker:
    def __init__(self, fps, confidence):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=confidence,track_low_thresh=.1,new_track_thresh=confidence,
            track_buffer=max(5,round(fps*1.5)),match_thresh=.8,fuse_score=True))
        self.confidence = confidence
        self.stable=StableTracks(fps)

    def update(self, people, shape):
        from ultralytics.engine.results import Boxes
        array = np.asarray([(*p.box,p.confidence,0) for p in people],np.float32).reshape(-1,6)
        tracks = self.tracker.update(Boxes(array,shape[:2]))
        output = []
        for row in tracks:
            index = int(row[-1])
            if not 0<=index<len(people):
                continue
            p = people[index]
            # Low-score observations help tracking but never confirm an alert.
            if p.confidence < self.confidence:
                continue
            p.track_id = int(row[4])
            output.append(p)
        return self.stable.update(output)
