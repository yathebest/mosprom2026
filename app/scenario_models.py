"""Third-party ONNX detectors. No uploaded Python or pickle code is executed."""
import ast
from dataclasses import dataclass,replace
from pathlib import Path
import re
import os
import sys

import cv2
import numpy as np

_DLL_HANDLES=[]


def onnx_session(path,threads=4):
    local=Path(__file__).resolve().parents[1]/'data/cache/ort-gpu-cu12'
    if os.name=='nt' and local.exists() and 'onnxruntime' not in sys.modules:
        sys.path.insert(0,str(local))
    import onnxruntime as ort
    providers=['CPUExecutionProvider']
    if os.getenv('SAFETY_ONNX_DEVICE','auto')!='cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
        import torch
        if torch.cuda.is_available():
            if os.name=='nt' and not _DLL_HANDLES:
                _DLL_HANDLES.append(os.add_dll_directory(str(Path(torch.__file__).parent/'lib')))
            providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC',
                        'cudnn_conv_use_max_workspace':'0','gpu_mem_limit':1024**3}),*providers]
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=options,providers=providers)


@dataclass
class Observation:
    box: tuple[float,float,float,float]
    confidence: float
    label: str
    model: str
    owner_box: tuple | None = None
    region: str | None = None


def letterbox_tensor(bgr,size,dtype=np.float32):
    h,w=bgr.shape[:2];tw,th=size;r=min(tw/w,th/h)
    nw,nh=round(w*r),round(h*r)
    left,top=round((tw-nw)/2-.1),round((th-nh)/2-.1)
    padded=np.full((th,tw,3),114,np.uint8)
    padded[top:top+nh,left:left+nw]=cv2.resize(bgr,(nw,nh))
    return np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=dtype)/255,(r,left,top)


def person_regions(person,width,height):
    """Context crops, not a claim that every head/torso is visible."""
    a,b,c,d=person.box;h=person.height;pad=h*.06
    regions={'head':(a-pad,b-pad,c+pad,b+h*.42),
             'torso':(a-pad,b+h*.15,c+pad,b+h*.82)}
    k=person.keypoints
    if k is not None and k.shape==(17,3):
        head=k[:5][k[:5,2]>=.5,:2]
        if len(head)>=2:
            center=head.mean(0);span=max(np.ptp(head[:,0])*1.8,h*.22)
            regions['head']=(center[0]-span/2,center[1]-span*.65,center[0]+span/2,center[1]+span*.65)
    return {key:tuple(map(int,np.clip(box,[0,0,0,0],[width,height,width,height]))) for key,box in regions.items()}


def normalized(label):
    value=re.sub(r'[^a-z0-9]+','_',str(label).lower()).strip('_')
    return 'cell_phone' if value in ('mobile_phone','phone') else value


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/max(union,1e-9))


class OnnxDetector:
    def __init__(self,path: Path,confidence=.5):
        if not path.exists():raise FileNotFoundError(f'Нет модели сценария: {path.name}')
        self.session=onnx_session(path)
        self.provider=self.session.get_providers()[0]
        inp=self.session.get_inputs()[0];self.input_name=inp.name
        self.size=(int(inp.shape[3]),int(inp.shape[2]))
        self.dtype=np.float16 if inp.type=='tensor(float16)' else np.float32
        metadata=self.session.get_modelmeta().custom_metadata_map
        names=ast.literal_eval(metadata.get('names','{}'))
        self.names={int(k):normalized(v) for k,v in (names.items() if isinstance(names,dict) else enumerate(names))}
        if not self.names:raise ValueError('ONNX needs explicit class names')
        self.name=path.stem;self.confidence=confidence

    def predict(self,bgr):
        h,w=bgr.shape[:2]
        tensor,(r,left,top)=letterbox_tensor(bgr,self.size,self.dtype)
        output=self.session.run(None,{self.input_name:tensor})[0]
        raw=np.asarray(output)[0]
        if raw.shape[0]==4+len(self.names):raw=raw.T
        if raw.shape[1]!=4+len(self.names):raise ValueError('Unsupported ONNX detection output shape')
        class_ids=raw[:,4:].argmax(1);scores=raw[np.arange(len(raw)),class_ids+4]
        keep=np.where(scores>=self.confidence)[0]
        keep=keep[np.argsort(scores[keep])[::-1]][:1500]
        results=[]
        for index in keep:
            x,y,bw,bh=raw[index,:4]
            box=np.array([(x-bw/2-left)/r,(y-bh/2-top)/r,(x+bw/2-left)/r,(y+bh/2-top)/r])
            box=tuple(np.clip(box,[0,0,0,0],[w,h,w,h]).tolist())
            if box[2]-box[0]<2 or box[3]-box[1]<2:continue
            label=self.names[int(class_ids[index])]
            if any(o.label==label and iou(o.box,box)>.45 for o in results):continue
            results.append(Observation(box,float(scores[index]),label,self.name))
            if len(results)>=150:break
        return results


class CropClassifier:
    """Optional locally trained classifier, with preprocessing in ONNX metadata."""
    def __init__(self,path,region):
        self.session=onnx_session(path,2)
        meta=self.session.get_modelmeta().custom_metadata_map
        import json
        self.labels=json.loads(meta['class_names']);self.region=region;self.name=path.stem
        self.mean=np.asarray(json.loads(meta['rgb_mean']),np.float32)[None,:,None,None]
        self.std=np.asarray(json.loads(meta['rgb_std']),np.float32)[None,:,None,None]
        inp=self.session.get_inputs()[0];self.input_name=inp.name;self.size=int(inp.shape[-1])
        self.threshold=float(meta.get('abstain_threshold','.8'))
        if meta.get('crop_policy')!='proposal_context_1.3':
            raise ValueError('PPE classifier must declare its training crop policy')
        if meta.get('resize_algorithm','pil_bilinear')!='pil_bilinear':
            raise ValueError('Unsupported PPE classifier resize algorithm')
        if set(self.labels)-{'hardhat','no_hardhat','vest','no_vest','unknown'}:
            raise ValueError('Unsupported PPE classifier classes')

    def predict(self,frame,person,proposals=()):
        # Training uses annotated PPE boxes with 30% context, not arbitrary
        # person-derived regions. Reproduce that geometry on detector proposals.
        labels={'hardhat','helmet','no_hardhat','no_helmet'} if self.region=='head' else {'vest','safety_vest','no_vest','no_safety_vest'}
        candidates=[o for o in proposals if o.label in labels]
        if not candidates:return None
        proposal=max(candidates,key=lambda o:o.confidence)
        a,b,c,d=proposal.box;bw,bh=c-a,d-b;h,w=frame.shape[:2]
        x1,y1=max(0,int(a-bw*.15)),max(0,int(b-bh*.15))
        x2,y2=min(w,int(c+bw*.15)),min(h,int(d+bh*.15))
        if min(x2-x1,y2-y1)<20:return None
        from PIL import Image
        crop=np.asarray(Image.fromarray(frame[y1:y2,x1:x2,::-1]).resize(
            (self.size,self.size),Image.Resampling.BILINEAR))
        tensor=np.ascontiguousarray(crop.transpose(2,0,1)[None],dtype=np.float32)/255
        logits=np.asarray(self.session.run(None,{self.input_name:(tensor-self.mean)/self.std})[0])[0]
        values=np.exp(logits-logits.max());scores=values/values.sum();index=int(scores.argmax())
        label=self.labels[index]
        if scores[index]<self.threshold or label=='unknown':return None
        return Observation((x1,y1,x2,y2),float(scores[index]),label,self.name,tuple(person.box),self.region)


def required_models(config):
    selected=[]
    if any((config.helmet_detection,config.vest_detection,config.mask_detection,config.harness_detection)):
        selected.append('ppe-safetyvision.onnx')
    if config.fire_detection or config.smoke_detection:selected.append('fire-smoke.onnx')
    if config.smoking_detection:selected.append('smoking.onnx')
    if config.phone_detection:selected.append('phone-objects.onnx')
    if config.load_posture_detection:selected.append('coco-objects.onnx')
    if config.proximity_detection:selected.append('construction-machinery.onnx')
    return selected


class ScenarioModels:
    def __init__(self,folder,config):
        self.phone_crops=config.phone_detection
        self.ppe_crops=config.ppe_crops
        self.max_people=config.ppe_max_people
        self.ppe_labels={'head':set(),'torso':set()}
        if config.helmet_detection:self.ppe_labels['head'].update({'hardhat','helmet','no_hardhat','no_helmet','bare_head'})
        if config.mask_detection:self.ppe_labels['head'].update({'mask','face_mask','no_mask','no_face_mask'})
        if config.vest_detection:self.ppe_labels['torso'].update({'vest','safety_vest','no_vest','no_safety_vest'})
        self.allowed={'person'}
        for flag,labels in {
            'helmet':{'hardhat','helmet','no_hardhat','no_helmet'},
            'vest':{'safety_vest','vest','no_safety_vest','no_vest'},
            'mask':{'mask','face_mask','no_mask','no_face_mask'},
            'harness':{'harness','safety_harness','no_harness'},
            'phone':{'cell_phone'},'load_posture':{'handbag','suitcase'},
            'fire':{'fire','flame','flames'},'smoke':{'smoke'},
            'smoking':{'smooking','smoking','cigarette'},
            'proximity':{'excavator','excavators','wheel_loader','dump_truck','truck','mini_van','vehicle','machinery','car','bus','motorcycle'},
        }.items():
            if getattr(config,flag+'_detection'):self.allowed.update(labels)
        self.models=[OnnxDetector(folder/name,config.object_confidence) for name in required_models(config)]
        self.classifiers=[CropClassifier(folder/name,region) for name,region,enabled in [
            ('ppe-head-classifier.onnx','head',config.helmet_detection),
            ('ppe-torso-classifier.onnx','torso',config.vest_detection)] if enabled and (folder/name).exists()]
        for name,enabled in [('ppe-head.onnx',config.helmet_detection),('ppe-vest.onnx',config.vest_detection)]:
            if enabled and (folder/name).exists():self.models.append(OnnxDetector(folder/name,config.object_confidence))

    def predict(self,frame,people=()):
        result=[]
        for model in self.models:
            result.extend(o for o in model.predict(frame) if o.label in self.allowed)
            if self.ppe_crops and model.name.startswith('ppe-'):
                h,w=frame.shape[:2]
                # Prioritize small visible people; bound inference work per frame.
                for person in sorted(people,key=lambda p:p.height)[:self.max_people]:
                    # Large foreground people already have detailed head/torso
                    # pixels in the 640px full-frame detector input. Repeating
                    # every PPE model on their crops adds live latency without
                    # rescuing a small object. Classifier proposal crops remain.
                    if person.height>=h*.55:continue
                    for region,(x1,y1,x2,y2) in person_regions(person,w,h).items():
                        if model.name=='ppe-head' and region!='head':continue
                        if model.name=='ppe-vest' and region!='torso':continue
                        labels=self.ppe_labels[region]
                        if not labels or x2-x1<20 or y2-y1<20:continue
                        for obs in model.predict(frame[y1:y2,x1:x2]):
                            if obs.label not in labels:continue
                            box=(obs.box[0]+x1,obs.box[1]+y1,obs.box[2]+x1,obs.box[3]+y1)
                            candidate=Observation(box,obs.confidence,obs.label,obs.model+':'+region+'_crop',tuple(person.box),region)
                            duplicates=[o for o in result if o.label==candidate.label and iou(o.box,box)>.45]
                            if duplicates and max(o.confidence for o in duplicates)>=candidate.confidence:continue
                            result=[o for o in result if o not in duplicates]
                            result.append(candidate)
            # A distant phone may occupy only a few pixels in the full-frame input.
            # Re-run the COCO detector on up to four largest visible people.
            if self.phone_crops and model.name=='phone-objects':
                h,w=frame.shape[:2]
                for person in sorted(people,key=lambda p:p.height,reverse=True)[:4]:
                    a,b,c,d=person.box;pad=person.height*.12
                    x1,y1=max(0,int(a-pad)),max(0,int(b-pad));x2,y2=min(w,int(c+pad)),min(h,int(d+pad))
                    if x2-x1<48 or y2-y1<80:continue
                    for obs in model.predict(frame[y1:y2,x1:x2]):
                        if obs.label!='cell_phone':continue
                        box=(obs.box[0]+x1,obs.box[1]+y1,obs.box[2]+x1,obs.box[3]+y1)
                        if not any(o.label=='cell_phone' and iou(o.box,box)>.45 for o in result):
                            result.append(Observation(box,obs.confidence,obs.label,obs.model+':person_crop'))
        # predict runs before tracking: incoming track_id values may all be 0.
        # Give this frame local unique IDs solely for spatial crop ownership.
        indexed_people=[replace(p,track_id=i+1) for i,p in enumerate(people)]
        for classifier in self.classifiers:
            from .scenarios import person_for
            for person in sorted(indexed_people,key=lambda p:p.height)[:self.max_people]:
                proposals=[o for o in result if person_for(o,indexed_people,'head' if classifier.region=='head' else 'body')==person.track_id]
                observation=classifier.predict(frame,person,proposals)
                if observation is not None:result.append(observation)
        return result
