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
        preferred=model.parent/'person-yolo12m.pt'
        if not preferred.exists():preferred=model.parent/'yolo12m.pt'
        people_path=Path(os.getenv('SAFETY_PERSON_MODEL',str(preferred)))
        if os.getenv('SAFETY_PERSON_MODEL') and not people_path.exists():
            raise FileNotFoundError('Configured person model does not exist')
        self.person_model=YOLO(str(people_path),task='detect') if people_path.exists() else None
        self.person_models=[self.person_model] if self.person_model else []
        stock_path=model.parent/'yolo12m.pt'
        if people_path.name=='person-yolo12m.pt' and stock_path.exists():
            self.person_models.append(YOLO(str(stock_path),task='detect'))
        if self.person_model:
            self.name+=' + '+people_path.name
        if len(self.person_models)>1:self.name+=' + COCO person ensemble'
        self.imgsz=int(os.getenv('SAFETY_IMAGE_SIZE','640'))
        self.reflections=[z for z in config.zones if z.kind=='reflection'] if config else []
        self.confidence = confidence
        self.model_id = model.stem+('+'+people_path.stem if self.person_model else '')
        if len(self.person_models)>1:self.model_id+='+yolo12m-coco-ensemble'
        self.rotated_search = os.getenv("SAFETY_ROTATED_SEARCH","0")=="1"
        self.scene_state = None
        self.body = None
        self.temporal_fall = None
        self.objects_state=[]
        self.objects=None
        if config is not None:
            from .scenario_models import ScenarioModels
            self.objects=ScenarioModels(model.parent,config)
            if self.objects.models:
                devices={m.provider.replace('ExecutionProvider','') for m in self.objects.models}
                self.name+=' + ONNX '+('/'.join(sorted(devices)))+' scenarios'
                self.model_id+='+'+'+'.join(m.name for m in self.objects.models)
            if self.objects.classifiers:
                self.name+=' + trained PPE classifiers'
                self.model_id+='+'+'+'.join(m.name for m in self.objects.classifiers)
        body_path=model.parent/'fall-resnet18.pt'
        if fall_detection and body_path.exists():
            from .body_classifier import BodyClassifier
            self.body=BodyClassifier(body_path,self.device)
            self.name += " + RGB ResNet18"
            self.model_id += "+fall-resnet18"
        temporal_path=model.parent/'fall-temporal.pt'
        if fall_detection and temporal_path.exists():
            from .fall_temporal import TemporalFallClassifier
            self.temporal_fall=TemporalFallClassifier(temporal_path,self.device)
            self.name += ' + temporal ResNet18/GRU'
            self.model_id += '+fall-temporal'

    def predict_temporal(self,bgr,now):
        if self.temporal_fall is None:return
        probability=self.temporal_fall.predict(bgr,now)
        state=dict(self.scene_state or {})
        state.update(fall_probability=probability,
                     fall_threshold=self.temporal_fall.threshold,
                     temporal_ready=probability is not None)
        self.scene_state=state

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
        if self.person_model:
            from .scenario_models import iou
            strong=[]
            for detector in self.person_models:
                detected=detector.predict(bgr,classes=[0],conf=.1,imgsz=self.imgsz,
                                          device=self.device,verbose=False)[0]
                for row in detected.boxes.data.cpu().numpy():
                    box=tuple(map(float,row[:4]));matches=[p for p in candidates if iou(p.box,box)>.3]
                    pose=max(matches,key=lambda p:iou(p.box,box)).keypoints if matches else None
                    strong.append(Person(box,float(row[4]),pose))
            candidates=suppress_duplicates(strong+[p for p in candidates if not any(iou(p.box,s.box)>.3 for s in strong)])
        if self.reflections:
            from .scenarios import contains
            # A bbox overlapping a mirror may belong to someone standing in
            # front of it. Only the annotated reflection foot region is masked.
            candidates=[p for p in candidates if not any(contains(z,p.foot,w,h) for z in self.reflections)]
        self.objects_state=self.objects.predict(bgr,candidates) if self.objects else []
        from .scenario_models import iou
        for obj in self.objects_state:
            if obj.label=='person' and not any(iou(p.box,obj.box)>.3 for p in candidates):
                candidate=Person(obj.box,obj.confidence)
                if not any(contains(z,candidate.foot,w,h) for z in self.reflections):
                    candidates.append(candidate)
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

    def update(self,people,now=None):
        self.frame+=1
        now=self.frame/self.fps if now is None else float(now)
        occupied=set();pending=[]
        for p in sorted(people,key=lambda p:p.confidence,reverse=True):
            raw=p.track_id;canonical=self.mapping.get(raw)
            if (canonical in self.recent and canonical not in occupied
                    and now-self.recent[canonical][0] <= 1.2):
                p.track_id=canonical;occupied.add(canonical)
            else:
                pending.append((raw,p))
        for raw,p in pending:
            matches=[]
            a=np.asarray(p.box)
            for canonical,(last,bbox) in self.recent.items():
                if canonical in occupied or now-last>1.2:
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
            self.recent[p.track_id]=(now,p.box)
        for canonical,(last,_) in list(self.recent.items()):
            if now-last>15:
                del self.recent[canonical]
        self.mapping={k:v for k,v in self.mapping.items() if v in self.recent}
        return people


class PeopleTracker:
    def __init__(self, fps, confidence):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=confidence,track_low_thresh=.1,new_track_thresh=confidence,
            track_buffer=max(5,round(fps*1.5)),match_thresh=.7,fuse_score=False))
        # In the pinned Ultralytics release score fusion is also applied in
        # ByteTrack's low-score stage (distance threshold .5). A .2-confidence
        # box then cannot match even at perfect IoU. Spatial matching rescues
        # existing IDs; the separate alert threshold remains unchanged.
        self.confidence = confidence
        self.stable=StableTracks(fps)
        self.fps=fps;self.frame=0;self.recent={}

    def update(self, people, shape, now=None):
        from ultralytics.engine.results import Boxes
        array = np.asarray([(*p.box,p.confidence,0) for p in people],np.float32).reshape(-1,6)
        tracks = self.tracker.update(Boxes(array,shape[:2]))
        self.frame+=1
        now=self.frame/self.fps if now is None else float(now)
        output = []
        for row in tracks:
            index = int(row[-1])
            if not 0<=index<len(people):
                continue
            p = people[index]
            # ByteTrack already associates these with a confirmed track. Keep the
            # measured box visible; alert rules use the separate eligibility flag.
            p.alert_eligible=p.confidence >= self.confidence
            p.track_id = int(row[4])
            output.append(p)
        output=self.stable.update(output,now)
        visible={p.track_id for p in output}
        for p in output:
            self.recent[p.track_id]=(now,tuple(p.box),p.confidence)
        for tid,(last,box,score) in list(self.recent.items()):
            if now-last > .5:
                del self.recent[tid]
            elif tid not in visible:
                output.append(Person(box,score,track_id=tid,observed=False,alert_eligible=False))
        return output
