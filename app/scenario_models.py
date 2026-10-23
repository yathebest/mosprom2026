"""Third-party ONNX detectors. No uploaded Python or pickle code is executed."""
import ast
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass
class Observation:
    box: tuple[float,float,float,float]
    confidence: float
    label: str
    model: str


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
        import onnxruntime as ort
        if not path.exists():raise FileNotFoundError(f'Нет модели сценария: {path.name}')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        inp=self.session.get_inputs()[0];self.input_name=inp.name
        self.size=(int(inp.shape[3]),int(inp.shape[2]))
        self.dtype=np.float16 if inp.type=='tensor(float16)' else np.float32
        metadata=self.session.get_modelmeta().custom_metadata_map
        names=ast.literal_eval(metadata.get('names','{}'))
        self.names={int(k):normalized(v) for k,v in (names.items() if isinstance(names,dict) else enumerate(names))}
        if not self.names:raise ValueError('ONNX needs explicit class names')
        self.name=path.stem;self.confidence=confidence

    def predict(self,bgr):
        h,w=bgr.shape[:2];tw,th=self.size;r=min(tw/w,th/h)
        nw,nh=round(w*r),round(h*r)
        left,top=round((tw-nw)/2-.1),round((th-nh)/2-.1)
        resized=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
        padded=np.full((th,tw,3),114,np.uint8);padded[top:top+nh,left:left+nw]=resized
        tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=self.dtype)/255
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

    def predict(self,frame,people=()):
        result=[]
        for model in self.models:
            result.extend(o for o in model.predict(frame) if o.label in self.allowed)
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
        return result
