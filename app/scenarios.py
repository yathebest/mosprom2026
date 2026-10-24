"""Evidence-based scenario candidates; unknown PPE never means missing PPE."""
from collections import deque
import math

import cv2
import numpy as np

from .core import Gate
from .scenario_models import iou

PPE={
 'helmet':({'helmet','hardhat'},{'no_helmet','no_hardhat'},'no_helmet','Каска'),
 'vest':({'vest','safety_vest'},{'no_vest','no_safety_vest'},'no_vest','Сигнальный жилет'),
 'mask':({'mask','face_mask'},{'no_mask','no_face_mask'},'no_mask','Защита лица'),
 'harness':({'harness','safety_harness'},{'no_harness'},'no_harness_at_height','Страховочная привязь'),
}
EQUIPMENT={'excavator','excavators','wheel_loader','dump_truck','truck','mini_van','vehicle','machinery','car','bus','motorcycle'}


class EvidenceGate(Gate):
    """Independent measurements confirm an episode; unknown is not clearance."""
    def __init__(self):
        super().__init__();self.samples=0;self.clear_since=None

    def update(self,now,condition,confirm,release,gap):
        if self.last_seen is not None and now<=self.last_seen:return None
        previous_seen=self.last_seen
        self.last_seen=now
        if condition is None:
            if self.last_true is not None and now-self.last_true>gap:
                self.since=None;self.samples=0
            self.clear_since=None
            return None
        if condition:
            if self.last_true is not None and now-self.last_true>gap:
                self.since=None;self.samples=0
            if self.since is None:self.since=now;self.samples=0
            self.samples+=1;self.last_true=now;self.clear_since=None
            if not self.active and self.samples>=3 and now-self.since>=confirm-1e-8:
                self.active=True;self.sequence+=1
                return self.since,self.sequence
        else:
            self.since=None;self.samples=0
            if self.clear_since is None or (previous_seen is not None and now-previous_seen>gap):self.clear_since=now
            if now-self.clear_since>=release:self.active=False
        return None


def ppe_measurement(matches,positive,negative,threshold,margin):
    # A specialist detector and independently trained proposal classifier can
    # corroborate each other. Do not compare uncalibrated softmax and YOLO
    # confidences as if they represented the same probability.
    # A low-score box is a proposal, not independent evidence for its class.
    # It must not veto a confident crop classifier as a "model conflict".
    detectors=[o for o in matches if o.model.split(':')[0] in ('ppe-head','ppe-vest') and o.confidence>=threshold]
    classifiers=[o for o in matches if o.model.endswith('-classifier') and o.confidence>=.8]
    d=max(detectors,key=lambda o:o.confidence,default=None)
    c=max(classifiers,key=lambda o:o.confidence,default=None)
    if d and c:
        if (d.label in negative)==(c.label in negative):
            return ('missing' if c.label in negative else 'worn'),'specialist_consensus'
        return 'unknown','conflicting_models'
    reliable=[o for o in matches if o.confidence>=threshold]
    assisted=[o for o in reliable if o.model.endswith(':pose_proposal')]
    if assisted:
        candidate=max(assisted,key=lambda o:o.confidence)
        if any((o.label in negative)!=(candidate.label in negative) for o in reliable if o not in assisted):
            return 'unknown','conflicting_models'
    worn=[o for o in reliable if o.label in positive];missing=[o for o in reliable if o.label in negative]
    if worn and missing:
        delta=max(o.confidence for o in worn)-max(o.confidence for o in missing)
        return (('worn' if delta>0 else 'missing'),'stronger_evidence') if abs(delta)>=margin else ('unknown','conflicting_models')
    assisted_reason='pose_assisted_torso' if assisted and assisted[0].region=='torso' else 'pose_assisted_head'
    return ('worn' if worn else 'missing' if missing else 'unknown'),(assisted_reason if assisted else 'measured' if reliable else 'no_reliable_evidence')


def contains(zone,point,width,height):
    polygon=np.asarray(zone.points,np.float32)*[width,height]
    return cv2.pointPolygonTest(polygon.astype(np.float32),point,False)>=0


def point_box_distance(point,box):
    x,y=point;a,b,c,d=box
    return math.hypot(max(a-x,0,x-c),max(b-y,0,y-d))


def person_for(observation,people,region='body'):
    if observation.owner_box is not None:
        owners=sorted(((iou(observation.owner_box,p.box),p.track_id) for p in people),reverse=True)
        if not owners or owners[0][0]<.5:return None
        if len(owners)>1 and owners[0][0]-owners[1][0]<.12:return None
        return owners[0][1]
    matches=[]
    a=np.asarray(observation.box,np.float32);area=max(1,(a[2]-a[0])*(a[3]-a[1]))
    for person in people:
        x1,y1,x2,y2=person.box;h=person.height
        region_box=np.array([x1,y1,x2,y1+h*.4] if region=='head' else [x1,y1,x2,y2])
        if point_box_distance(((a[0]+a[2])/2,(a[1]+a[3])/2),(x1-h*.1,y1-h*.1,x2+h*.1,y2+h*.1))>0:continue
        intersection=np.maximum(0,np.minimum(a[2:],region_box[2:])-np.maximum(a[:2],region_box[:2])).prod()
        relative=float(intersection/max(1,min(area,(region_box[2]-region_box[0])*(region_box[3]-region_box[1]))))
        if relative>=.3:
            center=(region_box[:2]+region_box[2:])/2
            distance=float(np.linalg.norm((a[:2]+a[2:])/2-center))/h
            matches.append((relative+.2*iou(a,person.box)-.1*distance,person.track_id))
    matches.sort(reverse=True)
    if not matches:return None
    if len(matches)>1 and matches[0][0]-matches[1][0]<.12:return None
    return matches[0][1]


class ScenarioRules:
    def __init__(self,config):
        self.config=config;self.gates={};self.outside=set();self.last_seen={}
        self.equipment={};self.next_equipment_id=1;self.ppe_status={}
        self.ppe_details={};self.ppe_memory={}
        self.incidents=[];self.boxes={}

    def update_equipment(self,objects,now):
        used=set();result=[]
        for obj in objects:
            matches=[(iou(obj.box,state['box']),tid) for tid,state in self.equipment.items()
                     if tid not in used and now-state['time']<.8 and obj.label==state['label']]
            best=max(matches,default=(0,None))
            tid=best[1] if best[0]>.1 else self.next_equipment_id
            if best[0]<=.1:self.next_equipment_id+=1
            state=self.equipment.setdefault(tid,{'history':deque(),'label':obj.label})
            history=state['history'];point=((obj.box[0]+obj.box[2])/2,obj.box[3])
            history.append((now,*point))
            while len(history)>1 and now-history[0][0]>1:history.popleft()
            dt=now-history[0][0];speed=math.hypot(point[0]-history[0][1],point[1]-history[0][2])/max(dt,1e-6) if dt>=.3 else 0
            state.update(box=obj.box,time=now);used.add(tid);result.append((obj,speed))
        self.equipment={tid:s for tid,s in self.equipment.items() if now-s['time']<15}
        return result

    def step(self,people,objects,now,width,height,base_details=None):
        cfg=self.config;base_details=base_details or {};events=[];active={}
        predicted={p.track_id for p in people if not p.observed}
        for key,gate in self.gates.items():
            if key[0] in predicted:gate.since=None
        proposals=list(objects)
        objects=[o for o in objects if o.confidence>=(cfg.phone_confidence if o.label=='cell_phone' else cfg.object_confidence)]
        people=[p for p in people if p.track_id>0 and p.observed]
        eligible={p.track_id for p in people if p.alert_eligible}
        assigned={}
        for key,(positive,negative,_,_) in PPE.items():
            assigned[key]={}
            for obj in proposals:
                if obj.label not in positive|negative:continue
                if obj.confidence<cfg.object_confidence and not (obj.model.split(':')[0] in ('ppe-head','ppe-vest') and obj.confidence>=.25):continue
                tid=person_for(obj,people,'head' if key in ('helmet','mask') else 'body')
                if tid is not None:assigned[key].setdefault(tid,[]).append(obj)
        self.ppe_status={p.track_id:{} for p in people}
        self.ppe_details={p.track_id:{} for p in people}
        visible={p.track_id for p in people}
        for p in people:self.boxes[p.track_id]=tuple(p.box)
        for item in self.incidents:
            if item['tid'] in visible:item.update(last=now,box=self.boxes[item['tid']])
        self.incidents=[x for x in self.incidents if now-x['last']<=15]
        def emit(tid,kind,zone,condition,hold,score,reason,obj=None,extra=None):
            evidence=kind in {'no_helmet','no_vest','no_mask','no_harness_at_height','possible_phone_use','possible_smoking'}
            gate=self.gates.setdefault((tid,kind,zone),EvidenceGate() if evidence else Gate())
            measured=condition if condition is None else bool(condition)
            if tid and tid not in eligible:measured=False
            release=max(3.,cfg.release_hold) if kind in {'possible_phone_use','possible_smoking'} else cfg.release_hold
            fired=gate.update(now,measured,hold,release,cfg.observation_gap)
            if gate.active:active.setdefault(tid,[]).append(kind)
            if fired:
                # An ID switch must not create a new incident for the same
                # spatially continuous worker. Never merge visible neighbours.
                duplicate=next((x for x in self.incidents if tid and x['tid'] not in visible
                    and x['kind']==kind and x['zone']==zone and now-x['last']<=3
                    and iou(x['box'],self.boxes[tid])>=.55),None)
                if duplicate:
                    duplicate.update(tid=tid,last=now,box=self.boxes[tid]);return
                if tid:self.incidents.append({'tid':tid,'kind':kind,'zone':zone,'last':now,'box':self.boxes[tid]})
                start,sequence=fired
                event={'track_id':tid,'kind':kind,'zone':zone,'start_s':round(start,3),'alert_s':round(now,3),
                       'sequence':sequence,'score':round(float(score),4),'reason':reason,
                       'confirmation_s':round(now-start,3),'source':'scenario_rules','scope':'person' if tid else 'scene',
                       'experimental':True}
                if obj:event.update(evidence_label=obj.label,evidence_model=obj.model,evidence_box=list(obj.box))
                if extra:event.update(extra)
                events.append(event)
        equipment=self.update_equipment([o for o in objects if o.label in EQUIPMENT],now) if cfg.proximity_detection else []
        homography=None
        if cfg.calibration:
            homography=cv2.getPerspectiveTransform((np.asarray(cfg.calibration.image_points)*[width,height]).astype(np.float32),np.asarray(cfg.calibration.world_points,np.float32))
        for person in people:
            tid=person.track_id;self.last_seen[tid]=now
            zones=[z for z in cfg.zones if contains(z,person.foot,width,height)]
            for key,(positive,negative,kind,title) in PPE.items():
                if not getattr(cfg,key+'_detection'):continue
                matches=assigned[key].get(tid,[])
                worn=[o for o in matches if o.label in positive];missing=[o for o in matches if o.label in negative]
                best_worn=max(worn,key=lambda o:o.confidence,default=None)
                best_missing=max(missing,key=lambda o:o.confidence,default=None)
                state,reason_code=ppe_measurement(matches,positive,negative,cfg.object_confidence,cfg.ppe_conflict_margin)
                head_pixels=round(min(person.box[2]-person.box[0],person.height*.22),1)
                if key in ('helmet','mask') and head_pixels<12:
                    state='unknown';reason_code='insufficient_pixels'
                fresh=state!='unknown'
                memory_key=(tid,key)
                if fresh:self.ppe_memory[memory_key]=(now,state)
                previous=self.ppe_memory.get(memory_key)
                if not matches and previous and now-previous[0]<=cfg.ppe_memory_s:
                    state=previous[1];reason_code='recent_observation'
                self.ppe_status[tid][key]=state
                self.ppe_details[tid][key]={'state':state,'reason':reason_code,'fresh':fresh,
                                          'head_pixels_estimate':head_pixels,
                                          'observed_at_s':now if fresh else previous[0] if previous else None}
                region=next((z.name for z in zones if z.kind=='height'),'') if key=='harness' else ''
                condition=(state=='missing' and (bool(region) if key=='harness' else True)) if fresh else None
                obj=max(missing,key=lambda o:o.confidence,default=None)
                reason=(f'{title}: модель явно обнаружила отсутствие экипировки; подтверждение {cfg.ppe_hold:g} с.'+
                        (' Человек находится в размеченной зоне работы на высоте.' if key=='harness' else ''))
                emit(tid,kind,region,condition,cfg.ppe_hold,obj.confidence if obj else 0,reason,obj,
                     {'ppe_assessment':dict(self.ppe_details[tid][key])})
            for zone in cfg.zones:
                if zone.kind not in ('barrier','under_load','under_conveyor'):continue
                inside=zone in zones;key=(tid,zone.name)
                if not inside:self.outside.add(key)
                kind={'barrier':'barrier_crossing','under_load':'under_load','under_conveyor':'under_conveyor'}[zone.kind]
                condition=inside and (key in self.outside if zone.kind=='barrier' else True)
                reason={'barrier':'Наблюдался переход в область за размеченной границей ограждения.',
                        'under_load':'Человек находится в размеченной запретной области под грузом.',
                        'under_conveyor':'Человек находится в размеченной запретной области под конвейером.'}[zone.kind]
                emit(tid,kind,zone.name,condition,cfg.zone_hold,person.confidence,reason,
                     extra={'context_source':'operator_polygon','limitation':'Physical barrier/load/conveyor is defined by operator; not independently recognized.'})
            if cfg.phone_detection:
                phones=[];wrists=[]
                if person.keypoints is not None:
                    wrists=[k[:2] for k in person.keypoints[[9,10]] if k[2]>=.4]
                    phones=[o for o in objects if o.label=='cell_phone' and person_for(o,people)==tid
                            and max(o.box[2]-o.box[0],o.box[3]-o.box[1])<=person.height*.35
                            and any(point_box_distance(w,o.box)<=person.height*.18 for w in wrists)]
                context=(not cfg.phone_work_context_only or base_details.get(tid,{}).get('speed',0)>=.12
                         or any(z.kind=='equipment' for z in zones))
                obj=max(phones,key=lambda o:o.confidence,default=None)
                emit(tid,'possible_phone_use','',(bool(obj) and context) if wrists else None,cfg.object_hold,obj.confidence if obj else 0,
                     'Телефон устойчиво обнаружен рядом с измеренной рукой. Использование требует проверки по клипу.',obj)
            if cfg.smoking_detection:
                # Smoke/dust alone is not smoking. Require a localized smoking
                # proposal near a sufficiently resolved person, not a distant
                # scene-level label amplified from a few pixels.
                smoking=[o for o in objects if o.label in ('smooking','smoking','cigarette')
                         and person_for(o,people)==tid and person.height>=cfg.smoking_min_person_height
                         and (o.box[1]+o.box[3])/2<=person.box[1]+person.height*.55
                         and not any(z.kind=='smoking_allowed' for z in zones)]
                obj=max(smoking,key=lambda o:o.confidence,default=None)
                emit(tid,'possible_smoking','',bool(obj),cfg.object_hold,obj.confidence if obj else 0,
                     'Повторно обнаружен визуальный признак курения рядом с человеком. Проверьте клип: жест сам по себе не доказывает курение.',obj)
            if cfg.load_posture_detection:
                bent=False;carried=[]
                k=person.keypoints
                if k is not None and all(k[i,2]>=.4 for i in (5,6,11,12)):
                    shoulders=k[[5,6],:2].mean(0);hips=k[[11,12],:2].mean(0)
                    delta=hips-shoulders;angle=math.degrees(math.atan2(abs(delta[0]),abs(delta[1])))
                    bent=40<=angle<=75
                    wrists=[k[i,:2] for i in (9,10) if k[i,2]>=.4]
                    carried=[o for o in objects if o.label in ('suitcase','handbag') and person_for(o,people)==tid
                             and any(point_box_distance(w,o.box)<person.height*.2 for w in wrists)]
                obj=max(carried,key=lambda o:o.confidence,default=None)
                moving=base_details.get(tid,{}).get('speed',0)>.1
                emit(tid,'unsafe_load_posture','',bent and bool(obj) and moving,cfg.object_hold,obj.confidence if obj else 0,
                     'Сильный наклон туловища при движении с распознанной ручной кладью. Экспериментальный признак; масса и произвольные коробки не определяются.',obj)
            if cfg.proximity_detection:
                candidates=[]
                for obj,motion_px_s in equipment:
                    if motion_px_s/person.height<.05:continue
                    a=np.array([obj.box[0],obj.box[3]],np.float32);b=np.array([obj.box[2],obj.box[3]],np.float32);point=np.asarray(person.foot,np.float32)
                    if homography is not None:
                        transformed=cv2.perspectiveTransform(np.array([[point,a,b]],np.float32),homography)[0];point,a,b=transformed
                    ab=b-a;t=np.clip(np.dot(point-a,ab)/max(float(ab@ab),1e-6),0,1)
                    distance=float(np.linalg.norm(point-(a+t*ab)))
                    normalized_distance=distance if homography is not None else distance/person.height
                    limit=cfg.proximity_m if homography is not None else cfg.proximity_heights
                    if normalized_distance<=limit:candidates.append((normalized_distance,obj))
                zone=next((z for z in zones if z.kind=='equipment'),None)
                best=min(candidates,key=lambda x:x[0],default=None);obj=best[1] if best else None
                metric=homography is not None
                reason='Человек вошёл в размеченную опасную область работающего оборудования.' if zone else 'Оценено сближение с движущимся распознанным транспортом/механизмом; требуется проверка перспективы.'
                emit(tid,'equipment_proximity',zone.name if zone else '',bool(best or zone),cfg.object_hold,
                     obj.confidence if obj else person.confidence,reason,obj,
                     {'distance_m':best[0] if best and metric else None,'distance_heights':best[0] if best and not metric else None,
                      'context_source':'operator_polygon' if zone else 'detected_moving_equipment'})
        scene_specs=[('fire',cfg.fire_detection,{'fire','flame','flames'},'fire_allowed','Обнаружены признаки открытого огня вне разрешённой области.'),
                     ('smoke',cfg.smoke_detection,{'smoke'},'fire_allowed','Обнаружены признаки дыма. Пар/пыль могут давать ошибки.'),
                     ]
        for kind,enabled,labels,allowed_kind,reason in scene_specs:
            if not enabled:continue
            candidates=[o for o in objects if o.label in labels and not any(z.kind==allowed_kind and contains(z,((o.box[0]+o.box[2])/2,(o.box[1]+o.box[3])/2),width,height) for z in cfg.zones)]
            obj=max(candidates,key=lambda o:o.confidence,default=None)
            emit(0,kind,'',bool(obj),cfg.object_hold,obj.confidence if obj else 0,reason,obj)
        expired={tid for tid,last in self.last_seen.items() if now-last>15}
        self.last_seen={tid:t for tid,t in self.last_seen.items() if tid not in expired}
        self.outside={key for key in self.outside if key[0] not in expired}
        self.gates={key:gate for key,gate in self.gates.items() if key[0] not in expired}
        self.ppe_memory={key:value for key,value in self.ppe_memory.items() if now-value[0]<=max(2,cfg.ppe_memory_s)}
        return events,active


def base_config(config):
    return config.model_copy(update={'zones':[z for z in config.zones if z.kind=='danger_zone']})


def assessment_rows(people,details):
    return [{'track_id':int(p.track_id),'tracking':'predicted' if not p.observed else 'measured' if p.alert_eligible else 'weak',
             'visibility':{'person_height_px':round(p.height,1),'small_objects':'limited' if p.height<120 else 'requires_review'},
             'ppe':details.get(p.track_id,{})} for p in people[:32]]


def annotate_objects(frame,objects,ppe_status,people,ppe_details=None):
    for obj in objects:
        if obj.label in ('person','fall_detected'):continue
        color=(65,65,240) if obj.label.startswith('no_') or obj.label in ('fire','smoke','smooking','smoking') else (60,205,220)
        x1,y1,x2,y2=map(int,obj.box)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f'{obj.label} {obj.confidence:.2f}',(max(0,x1),max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    for person in people:
        status=ppe_status.get(person.track_id,{})
        known=[f'{key}:{value}'+(' (recent)' if (ppe_details or {}).get(person.track_id,{}).get(key,{}).get('reason')=='recent_observation' else '') for key,value in status.items()]
        if known:
            x,y=map(int,person.box[:2]);cv2.putText(frame,' '.join(known),(max(0,x),max(32,y+17)),cv2.FONT_HERSHEY_SIMPLEX,.4,(80,235,230),1,cv2.LINE_AA)
    return frame
