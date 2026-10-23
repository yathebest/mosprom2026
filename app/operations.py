"""Operator workflow. Priorities are explicit rules, not model probabilities."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

from . import storage as db

PRIORITIES={'possible_fall':'high','person_down':'high','danger_zone':'medium','fast_motion':'medium',
            'fire':'high','smoke':'high','possible_smoking':'medium','no_helmet':'medium','no_vest':'medium',
            'no_mask':'medium','no_harness_at_height':'high','possible_phone_use':'medium',
            'equipment_proximity':'high','barrier_crossing':'high','under_load':'high','under_conveyor':'high',
            'unsafe_load_posture':'medium'}


def triage(events,now=None):
    now=now or datetime.now(timezone.utc)
    result=[]
    for event in events:
        if event['status'] not in ('new','forwarded'):continue
        age=max(0,(now-datetime.fromisoformat(event['created'].replace('Z','+00:00'))).total_seconds())
        priority=event['payload'].get('priority') or PRIORITIES.get(event['kind'],'low')
        target=60 if priority=='high' else 300
        result.append({**event,'priority':priority,'waiting_s':round(age),'review_target_s':target,
                       'overdue':age>target,'priority_reason':'Приоритет задан типом события / настройкой зоны, не вероятностью модели.'})
    return sorted(result,key=lambda e:({'high':0,'medium':1,'low':2}[e['priority']],e['created']))


def review_archive():
    records=[e for e in db.events() if e['status'] in ('confirmed','false')]
    output=io.BytesIO();manifest=[]
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for event in records:
            item={key:event[key] for key in ('id','job_id','kind','camera','zone','start_s','alert_s','status','payload')}
            item['supervision']='operator_confirmed_event' if event['status']=='confirmed' else 'operator_rejected_alert'
            item['split']='unassigned; group by source video/camera/person before training'
            image=(db.DATA/event['image']).resolve()
            if image.is_relative_to((db.DATA/'runs').resolve()) and image.is_file():
                relative=f"evidence/{event['id']}.jpg";archive.write(image,relative);item['evidence']=relative
            clip=(db.DATA/(event['clip'] or '')).resolve()
            if clip.is_relative_to((db.DATA/'runs').resolve()) and clip.is_file():
                relative=f"clips/{event['id']}.mp4";archive.write(clip,relative);item['clip']=relative
            manifest.append(item)
        archive.writestr('annotations.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        archive.writestr('README.txt','Operator review export. Rejected alerts are not automatically safe/negative frames.\n'
                         'Evidence JPEGs have overlays; extract unmodified frames from original clips for model training.\n'
                         'No automatic retraining. Review labels and split by source/camera/person.\n'
                         'Respect original video/dataset permissions and attribution.\n')
    return output.getvalue()
