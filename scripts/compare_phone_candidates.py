"""Fixed phone validation and optional PRIVATE local clip regression. Never installs weights."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from app.core import RunConfig
from app.inference import YoloBackend,PeopleTracker
from app.scenario_models import OnnxDetector,ScenarioModels,iou
from app.scenarios import ScenarioRules,person_for,point_box_distance
from fpi_labels import read_labels


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def js(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,default=js),encoding='utf-8');temporary.replace(path)


class ReplaySession:
    """Compare decoding on EXACT same NN tensors, with no extra inference."""
    def __init__(self,session,phone_id,nc):self.actual=session;self.phone_id=phone_id;self.nc=nc;self.values=[];self.index=None
    def __getattr__(self,name):return getattr(self.actual,name)
    def record(self):self.values=[];self.index=None
    def replay(self):self.index=0
    def run(self,*args,**kwargs):
        if self.index is None:
            values=self.actual.run(*args,**kwargs);self.values.append(values);return values
        values=self.values[self.index];self.index+=1
        raw=values[0].copy()
        if raw.shape[1]==4+self.nc:
            scores=raw[:,4:,:].copy();raw[:,4:,:]=-np.inf;raw[:,4+self.phone_id,:]=scores[:,self.phone_id,:]
        elif raw.shape[-1]==4+self.nc:
            scores=raw[:,:,4:].copy();raw[:,:,4:]=-np.inf;raw[:,:,4+self.phone_id]=scores[:,:,self.phone_id]
        else:raise ValueError('Unsupported replay shape')
        return [raw,*values[1:]]


def score(predictions,truth):
    matched=set()
    for prediction in sorted(predictions,key=lambda o:o.confidence,reverse=True):
        overlap,index=max(((iou(prediction.box,box),i) for i,box in enumerate(truth) if i not in matched),default=(0,None))
        if overlap>=.5:matched.add(index)
    return {'tp':len(matched),'fp':len(predictions)-len(matched),'fn':len(truth)-len(matched),
        'small_tp':sum((b[2]-b[0])*(b[3]-b[1])<1024 for i,b in enumerate(truth) if i in matched),
        'small_truth':sum((b[2]-b[0])*(b[3]-b[1])<1024 for b in truth)}


def fixed_manifest(dataset):
    groups={True:[],False:[]}
    for path in sorted((dataset/'images').glob('*.jpg')):
        labels=read_labels(dataset/'labels'/f'{path.stem}.txt')
        groups[any(row[0]==0 for row in labels)].append(path)
    rng=random.Random(42);selection=[]
    for positive in (True,False):
        rng.shuffle(groups[positive])
        if len(groups[positive])<64:raise ValueError('Need 64 positive and 64 negative images')
        for path in groups[positive][:64]:selection.append({'file':path.name,'phone_positive':positive,'sha256':sha(path)})
    return selection,{str(key):len(value) for key,value in groups.items()}


def private_source(job_id):
    conn=sqlite3.connect(ROOT/'data/events.sqlite3');conn.row_factory=sqlite3.Row
    job=conn.execute('SELECT config FROM jobs WHERE id=?',(job_id,)).fetchone()
    event=conn.execute("SELECT clip,payload FROM events WHERE job_id=? AND kind='possible_phone_use' ORDER BY alert_s LIMIT 1",(job_id,)).fetchone();conn.close()
    if not job or not event or not event['clip']:raise ValueError('Private job needs a saved phone clip')
    return ROOT/'data'/event['clip'],json.loads(job['config']),json.loads(event['payload'])


def private_measurements(path,original_config):
    import torch
    torch.set_num_threads(2)
    backend=YoloBackend(ROOT/'data/models/yolov8n-pose.pt',original_config['confidence'],config=None)
    backend.distant_people=original_config['distant_people'];backend.refine_phone_pose=True
    backend.refine_head_pose=original_config.get('helmet_detection',False);backend.refine_torso_pose=original_config.get('vest_detection',False)
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);stride=max(1,round(fps/2.5))
    tracker=PeopleTracker(fps/stride,original_config['confidence']);rows=[];index=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if index%stride==0:
            tick=time.perf_counter();raw=backend.predict_people(frame);original=deepcopy(raw)
            tracked=tracker.update(raw,frame.shape,index/fps)
            rows.append({'at_s':index/fps,'frame':frame,'people':tracked,'raw_people':original,'people_inference_s':time.perf_counter()-tick})
        index+=1
    cap.release();del backend;gc.collect();torch.cuda.empty_cache()
    return rows,fps/stride


def gates(people,observations,config):
    output=[]
    for person in people:
        if person.track_id<=0:continue
        wrists=[k[:2] for k in person.keypoints[[9,10]] if k[2]>=.4] if person.keypoints is not None else []
        associated=[o for o in observations if o.label=='cell_phone' and o.confidence>=config.phone_confidence and person_for(o,people)==person.track_id]
        sized=[o for o in associated if max(o.box[2]-o.box[0],o.box[3]-o.box[1])<=person.height*.35]
        near=[o for o in sized if any(point_box_distance(w,o.box)<=person.height*.18 for w in wrists)]
        output.append({'track_id':person.track_id,'alert_eligible':person.alert_eligible,'person_height':person.height,
            'wrists':len(wrists),'associated_phones':len(associated),'size_valid':len(sized),'hand_valid':len(near)})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--private-job')
    parser.add_argument('--runtime',type=Path,default=ROOT/'data/models/phone-objects.onnx')
    parser.add_argument('--candidate',type=Path,default=ROOT/'data/cache/kaggle-phone-head-completed/phone-head-experiment/phone-head.onnx')
    parser.add_argument('--output',type=Path,default=ROOT/'data/cache/phone-comparison-20261025.json');args=parser.parse_args()
    cv2.setNumThreads(1)
    dataset=ROOT/'data/datasets/fpi-det/reorganized_dataset/val'
    manifest,pools=fixed_manifest(dataset)
    report={'status':'running','split':'FPI-Det supplied val','confidence':.3,'matching_iou':.5,'seed':42,
        'manifest':manifest,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'pool_sizes':pools,
        'limitations':['Validation, not an independent factory/event benchmark. Candidate trained with this source validation split.',
            'Negative means no annotated phone; head/hand annotation completeness and source/video independence are not newly certified.',
            'No thresholds are tuned. Phone-only decoding is an experimental ablation, not automatically enabled.',
            'Private webcam stays local and is never copied to Git or an external service. Partial alert clip cannot establish full-stream event recall.'],
        'models':{},'private':None}
    paths={'runtime':args.runtime.resolve(),'candidate':args.candidate.resolve()}
    save(args.output,report)
    for name,path in paths.items():
        detector=OnnxDetector(path,.3);ids=[i for i,label in detector.names.items() if label=='cell_phone']
        if ids!=[67]:raise ValueError(f'Expected phone=67, got {ids}')
        replay=ReplaySession(detector.session,67,len(detector.names));detector.session=replay
        totals={mode:{'tp':0,'fp':0,'fn':0,'small_tp':0,'small_truth':0,'negative_fp':0} for mode in ('all_class_argmax','phone_only_argmax')};rows=[]
        started=time.perf_counter()
        for n,item in enumerate(manifest):
            frame=cv2.imread(str(dataset/'images'/item['file']));h,w=frame.shape[:2]
            truth=[((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h) for cls,x,y,bw,bh in read_labels(dataset/'labels'/f"{Path(item['file']).stem}.txt") if cls==0]
            replay.record();old=[o for o in detector.predict(frame) if o.label=='cell_phone'];replay.replay();restricted=detector.predict(frame)
            results={}
            for mode,observations in [('all_class_argmax',old),('phone_only_argmax',restricted)]:
                counts=score(observations,truth);results[mode]={'counts':counts,'predictions':[asdict(o) for o in observations]}
                for key,value in counts.items():totals[mode][key]+=value
                if not truth:totals[mode]['negative_fp']+=counts['fp']
            rows.append({'file':item['file'],'phone_positive':item['phone_positive'],'truth':truth,'results':results})
            if (n+1)%32==0:print('PHONE',name,n+1,flush=True)
        for values in totals.values():
            values['precision']=values['tp']/max(1,values['tp']+values['fp']);values['recall']=values['tp']/max(1,values['tp']+values['fn'])
        report['models'][name]={'path':str(path),'sha256':sha(path),'provider':detector.provider,'class0':detector.names[0],'class67':detector.names[67],
            'elapsed_s':time.perf_counter()-started,'totals':totals,'rows':rows};save(args.output,report)
        print('RESULT',name,totals,flush=True)
        del detector,replay;gc.collect()
    if args.private_job:
        path,original,payload=private_source(args.private_job)
        measurements,effective_fps=private_measurements(path,original)
        config=RunConfig(**{**original,'fall_detection':False,'helmet_detection':False,'vest_detection':False,
            'phone_detection':True,'smoking_detection':False,'fire_detection':False,'smoke_detection':False,
            'proximity_detection':False,'load_posture_detection':False,'harness_detection':False,'mask_detection':False,'zones':[]})
        report['private']={'scope':'PRIVATE LOCAL USER-REQUESTED WEBCAM REGRESSION. Do not publish this section/images.',
            'clip_sha256':sha(path),'clip_start_s':payload.get('clip_start_s'),'config':config.model_dump(),'effective_fps':effective_fps,
            'source_recorded_event':{key:payload.get(key) for key in ('start_s','alert_s','end_s','episode_state')},'models':{}}
        for name,path in paths.items():
            detector=OnnxDetector(path,config.phone_confidence);detector.name='phone-objects'
            replay=ReplaySession(detector.session,67,len(detector.names));detector.session=replay
            models=ScenarioModels.__new__(ScenarioModels);models.person_confidence=config.confidence;models.phone_crops=True
            models.smoking_crops=False;models.ppe_crops=False;models.models=[detector];models.allowed={'cell_phone'};models.classifiers=[];models.max_people=config.ppe_max_people
            rules={mode:ScenarioRules(config) for mode in ('all_class_argmax','phone_only_argmax')};results={mode:{'events':[],'measurements':[]} for mode in rules}
            for sample in measurements:
                replay.record();old=models.predict(sample['frame'],sample['raw_people']);replay.replay();restricted=models.predict(sample['frame'],sample['raw_people'])
                for mode,objects in [('all_class_argmax',old),('phone_only_argmax',restricted)]:
                    people=deepcopy(sample['people']);events,_=rules[mode].step(people,objects,sample['at_s'],sample['frame'].shape[1],sample['frame'].shape[0])
                    results[mode]['events'].extend(events);results[mode]['measurements'].append({'at_s':sample['at_s'],
                        'phones':[asdict(o) for o in objects if o.label=='cell_phone'],'gates':gates(people,objects,config),
                        'states':deepcopy(rules[mode].state_rows)})
            report['private']['models'][name]=results;save(args.output,report)
            print('PRIVATE',name,{mode:len(value['events']) for mode,value in results.items()},flush=True)
            del detector,replay,models;gc.collect()
    report['status']='completed';save(args.output,report);print('DONE',args.output,flush=True)


if __name__=='__main__':main()
